"""HSI-PatchCore — nonparametric memory-bank anomaly detection on dense per-location features.

PatchCore (Roth et al., CVPR 2022) scores a location by the distance between its feature vector
and the nearest entry of a coreset-subsampled memory bank of *normal* features. The node takes any
``[B, H, W, C]`` grid of features:

- a hyperspectral cube, whose spectrum is already a dense, physically meaningful per-pixel feature
  (``standardize=True``: every band is z-scored with statistics fitted on normal data);
- a deep feature grid, e.g. ViT patch tokens laid out on their patch grid
  (``standardize=False``: the features are used as they are).

Pipeline per frame:

1. optionally z-score every channel with the fitted ``mu`` / ``sd`` buffers;
2. average over a ``pool_size`` x ``pool_size`` window (PatchCore's "locally aware" aggregation);
3. sample the pooled map on a ``stride`` grid (``bank_stride`` while filling the bank);
4. score = Euclidean distance to the nearest ``coreset`` row (k = 1);
5. bilinearly upsample the coarse distance map to the input resolution, or to the resolution of
   the optional ``reference`` input (the cube, when the input is a coarse feature grid).

Because z-scoring is affine per channel and average pooling is linear, the bank is built in ONE
pass over the fit stream: window sums of the raw input (plus the number of in-image cells per
window, so the zero-padded border is standardised exactly like an inference-time pooled z-score)
are collected while the channel statistics accumulate, then standardised once, capped to
``max_bank_size`` rows, and reduced to ``coreset_size`` rows with k-center greedy
(:func:`cuvis_ai_patchcore.sampling.k_center_greedy`).

The node is nonparametric (no gradient-trainable state, no ``TRAINABLE_BUFFERS``); Phase 1
(``StatisticalTrainer`` / ``restore-trainrun``) is its whole training. All fitted state lives in
eagerly-sized buffers, so ``state_dict`` round-trips through the pipeline ``.pt``; with
``standardize=False`` the ``mu`` / ``sd`` buffers stay at their identity values.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai_patchcore.node._common import check_topk_frac, random_cap, require_fitted, topk_mean
from cuvis_ai_patchcore.sampling import k_center_greedy
from cuvis_ai_patchcore.streaming_stats import StreamingMeanVar

_CORESET_PROJECTIONS = {"sparse_random"}
_AUTOCAST_DTYPES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}
# Reduced-precision guard. ``torch.cdist`` expands |a-b|^2 = |a|^2 + |b|^2 - 2ab; with 61-band
# z-scores a single far-out-of-distribution pixel (|z| in the hundreds after a session drift) pushes
# that expansion past float16's 65504 -> inf distance -> a NaN/inf frame. Clamping |z| to
# ``_HALF_CLIP`` and scaling by ``_HALF_SCALE`` keeps every intermediate in range (max squared
# distance 4 * C * (clip * scale)^2 ~ 3.9e3 for C=61); the distance is rescaled afterwards, so
# only pixels beyond 64 sigma (already extreme anomalies) differ from the float32 path. The clamp
# is a z-unit assumption and is applied only when the node standardizes; raw feature grids get the
# pre-scaling alone.
_HALF_CLIP = 64.0
_HALF_SCALE = 1.0 / 16.0


class PatchCoreDetector(Node):
    """Nearest-coreset distance anomaly detector on spectra or dense feature grids (PatchCore)."""

    _category = NodeCategory.MODEL
    _tags = frozenset({NodeTag.HYPERSPECTRAL, NodeTag.ANOMALY, NodeTag.TORCH, NodeTag.STATEFUL})

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Dense feature grid [B, H, W, C] float32: a hyperspectral cube (spectra "
            "as features) or any per-location features such as ViT patch tokens on their patch "
            "grid; C must equal input_channels.",
        ),
        "reference": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            optional=True,
            description="Optional [B, H_ref, W_ref, *] tensor whose spatial size sets the "
            "resolution of `scores` (e.g. the cube when the input is a coarse feature grid). "
            "Defaults to the input's own H, W.",
        ),
    }

    OUTPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Pixel-wise anomaly scores [B, H, W, 1]: distance to the nearest coreset "
            "entry, bilinearly upsampled from the stride grid to the input (or reference) size.",
        ),
        "anomaly_score": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Image-level anomaly score [B] (mean of the top topk_frac pixel scores).",
        ),
    }

    def __init__(
        self,
        input_channels: int,
        coreset_size: int = 8000,
        stride: int = 4,
        bank_stride: int = 10,
        pool_size: int = 3,
        max_bank_size: int = 80000,
        topk_frac: float = 0.001,
        chunk_size: int = 4096,
        autocast_dtype: str | None = None,
        standardize: bool = True,
        seed: int = 0,
        eps: float = 1e-6,
        coreset_projection: str | None = None,
        projection_eps: float = 0.9,
        **kwargs: Any,
    ) -> None:
        """Create an unfitted detector; buffers are sized from the constructor arguments.

        Parameters
        ----------
        input_channels : number of channels ``C`` of the input grid (spectral bands, or the
            feature dimension of a deep feature grid).
        coreset_size : rows kept in the memory bank after k-center greedy. Fixed buffer size —
            a smaller bank is cycled to fill it (duplicates never change a nearest distance).
        stride : sampling stride of the scoring grid (latency scales ~1/stride**2).
        bank_stride : sampling stride used while collecting normal features in Phase 1.
        pool_size : odd side of the local averaging window; ``1`` disables pooling.
        max_bank_size : random (seeded) cap on the collected normal features before coreset
            selection; bounds Phase-1 memory and time.
        topk_frac : fraction of output pixels averaged into the image-level ``anomaly_score``.
        chunk_size : query rows per ``torch.cdist`` call (memory / speed trade-off).
        autocast_dtype : ``None`` (float32), ``"float16"`` or ``"bfloat16"`` — reduced-precision
            nearest-neighbour search, applied on CUDA inputs only (CPU always runs float32).
        standardize : z-score every channel with statistics fitted in Phase 1 (``True``, the
            spectral setting) or use the input features unchanged (``False``, for deep feature
            grids that are already on a common scale).
        seed : RNG seed for the bank cap, the k-center-greedy start point and the projection.
        eps : floor for the per-channel standard deviation.
        coreset_projection : ``None`` (default) selects the coreset on exact feature distances;
            ``"sparse_random"`` first maps the bank through anomalib's sparse random projection
            (Johnson-Lindenstrauss dimension for ``projection_eps``) and selects on the projected
            distances, i.e. anomalib's ``KCenterGreedy`` recipe. Scoring always uses the original
            features; only which rows enter the bank changes.
        projection_eps : JL distortion parameter of the projection (anomalib default ``0.9``).
        """
        if input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {input_channels}")
        if coreset_size <= 0:
            raise ValueError(f"coreset_size must be positive, got {coreset_size}")
        if stride < 1 or bank_stride < 1:
            raise ValueError("stride and bank_stride must be >= 1")
        if pool_size < 1 or pool_size % 2 == 0:
            raise ValueError(f"pool_size must be an odd positive integer, got {pool_size}")
        if max_bank_size < coreset_size:
            raise ValueError("max_bank_size must be >= coreset_size")
        topk_frac = check_topk_frac(topk_frac)
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if autocast_dtype is not None and autocast_dtype not in _AUTOCAST_DTYPES:
            raise ValueError(
                f"autocast_dtype must be None or one of {sorted(_AUTOCAST_DTYPES)}, "
                f"got {autocast_dtype!r}"
            )
        if coreset_projection is not None and coreset_projection not in _CORESET_PROJECTIONS:
            raise ValueError(
                f"coreset_projection must be None or one of {sorted(_CORESET_PROJECTIONS)}, "
                f"got {coreset_projection!r}"
            )
        if not 0.0 < projection_eps < 1.0:
            raise ValueError(f"projection_eps must be in (0, 1), got {projection_eps}")

        self.input_channels = int(input_channels)
        self.coreset_size = int(coreset_size)
        self.stride = int(stride)
        self.bank_stride = int(bank_stride)
        self.pool_size = int(pool_size)
        self.max_bank_size = int(max_bank_size)
        self.topk_frac = float(topk_frac)
        self.chunk_size = int(chunk_size)
        self.autocast_dtype = autocast_dtype
        self.standardize = bool(standardize)
        self.seed = int(seed)
        self.eps = float(eps)
        self.coreset_projection = coreset_projection
        self.projection_eps = float(projection_eps)

        super().__init__(
            input_channels=self.input_channels,
            coreset_size=self.coreset_size,
            stride=self.stride,
            bank_stride=self.bank_stride,
            pool_size=self.pool_size,
            max_bank_size=self.max_bank_size,
            topk_frac=self.topk_frac,
            chunk_size=self.chunk_size,
            autocast_dtype=self.autocast_dtype,
            standardize=self.standardize,
            seed=self.seed,
            eps=self.eps,
            coreset_projection=self.coreset_projection,
            projection_eps=self.projection_eps,
            **kwargs,
        )

        c = self.input_channels
        self.register_buffer("mu", torch.zeros(c, dtype=torch.float32))
        self.register_buffer("sd", torch.ones(c, dtype=torch.float32))
        self.register_buffer("coreset", torch.zeros(self.coreset_size, c, dtype=torch.float32))
        self._nn_dtype = _AUTOCAST_DTYPES.get(autocast_dtype) if autocast_dtype else None

    # ------------------------------------------------------------------ features
    def _pooled(self, x_bchw: Tensor, stride: int) -> Tensor:
        """[B, C, H, W] -> locally averaged map (zero-padded) sampled on a ``stride`` grid."""
        if self.pool_size > 1:
            x_bchw = F.avg_pool2d(x_bchw, self.pool_size, stride=1, padding=self.pool_size // 2)
        return x_bchw[:, :, ::stride, ::stride]

    def _features(self, cube: Tensor, stride: int) -> Tensor:
        """BHWC grid -> (optionally z-scored) pooled features [B, C, gh, gw]."""
        z = (cube - self.mu) / self.sd if self.standardize else cube
        return self._pooled(z.permute(0, 3, 1, 2), stride)

    def _window_sums(self, cube: Tensor, stride: int) -> tuple[Tensor, Tensor]:
        """Raw window sums [B, C, gh, gw] and in-image cell counts [B, 1, gh, gw] on the grid.

        With zero padding, ``pool(z)[p] = (sum_valid(raw) - n_valid[p] * mu) / (k**2 * sd)``, so
        collecting the sums and counts lets Phase 1 standardise AFTER the pass and still
        reproduce the inference-time feature exactly, border windows included.
        """
        k2 = float(self.pool_size * self.pool_size)
        sums = self._pooled(cube.permute(0, 3, 1, 2), stride) * k2
        ones = torch.ones(1, 1, cube.shape[1], cube.shape[2], dtype=cube.dtype, device=cube.device)
        counts = self._pooled(ones, stride) * k2
        return sums, counts.expand(cube.shape[0], -1, -1, -1)

    def _nearest_distance(self, flat: Tensor) -> Tensor:
        """Euclidean distance from each ``(N, C)`` query row to its nearest coreset row."""
        bank = self.coreset
        rescale = 1.0
        if self._nn_dtype is not None and flat.is_cuda:
            if self.standardize:  # the clamp is a z-unit assumption
                flat = flat.clamp(-_HALF_CLIP, _HALF_CLIP)
                bank = bank.clamp(-_HALF_CLIP, _HALF_CLIP)
            flat = (flat * _HALF_SCALE).to(self._nn_dtype)
            bank = (bank * _HALF_SCALE).to(self._nn_dtype)
            rescale = 1.0 / _HALF_SCALE
        out = torch.empty(flat.shape[0], dtype=torch.float32, device=flat.device)
        for i in range(0, flat.shape[0], self.chunk_size):
            chunk = flat[i : i + self.chunk_size]
            out[i : i + chunk.shape[0]] = torch.cdist(chunk, bank).min(dim=1).values.float()
        return out * rescale if rescale != 1.0 else out

    # ------------------------------------------------------------------ phase 1
    @torch.no_grad()
    def statistical_initialization(self, input_stream) -> None:
        """Fit channel statistics (if standardizing) and the coreset from a stream of normal grids.

        Single pass: ``StreamingMeanVar`` accumulates per-channel mean / variance over all pixels
        (skipped when ``standardize=False``) while raw window sums (and in-image cell counts) on
        the ``bank_stride`` grid are collected; the samples are standardised afterwards into
        exactly the inference feature, capped to ``max_bank_size`` rows and reduced to
        ``coreset_size`` rows by k-center greedy.
        """
        self._statistically_initialized = False
        stats = StreamingMeanVar(self.input_channels) if self.standardize else None
        n_pixels = 0
        sums: list[Tensor] = []
        counts: list[Tensor] = []
        for batch in input_stream:
            cube = batch.get("cube") if isinstance(batch, dict) else None
            if cube is None:
                continue
            if cube.shape[-1] != self.input_channels:
                raise ValueError(
                    f"{type(self).__name__}: input has {cube.shape[-1]} channels, "
                    f"input_channels={self.input_channels}"
                )
            cube = cube.float()
            n_pixels += cube.shape[0] * cube.shape[1] * cube.shape[2]
            if stats is not None:
                # Row chunks keep the float64 temporaries of the accumulator small (a 1000x1080x61
                # cube would otherwise need ~1 GB of transient GPU memory next to the SDK's pools).
                for rows in cube.reshape(-1, self.input_channels).split(262144, dim=0):
                    stats.update(rows)
            s, n = self._window_sums(cube, self.bank_stride)  # [B,C,gh,gw], [B,1,gh,gw]
            sums.append(s.permute(0, 2, 3, 1).reshape(-1, self.input_channels))
            counts.append(n.permute(0, 2, 3, 1).reshape(-1, 1))
        if n_pixels < 2 or not sums:
            raise RuntimeError(
                f"{type(self).__name__}.statistical_initialization() received insufficient "
                "normal data (need at least 2 pixels)."
            )

        device = sums[0].device
        if stats is not None:
            mu = stats.mean.to(device)
            sd = stats.var.sqrt().clamp_min(self.eps).to(device)
        else:  # identity standardisation: the bank holds the pooled raw features
            mu = torch.zeros(self.input_channels, dtype=torch.float32, device=device)
            sd = torch.ones(self.input_channels, dtype=torch.float32, device=device)
        k2 = float(self.pool_size * self.pool_size)
        bank = (torch.cat(sums, dim=0) - torch.cat(counts, dim=0) * mu) / (k2 * sd)

        gen = torch.Generator().manual_seed(self.seed)
        bank = random_cap(bank, self.max_bank_size, gen)
        idx = k_center_greedy(
            bank,
            self.coreset_size,
            generator=gen,
            projection_eps=self.projection_eps if self.coreset_projection else None,
        ).to(bank.device)
        core = bank[idx]
        if core.shape[0] < self.coreset_size:  # tiny bank: cycle rows to fill the fixed buffer
            core = core[torch.arange(self.coreset_size, device=core.device) % core.shape[0]]

        self.mu.copy_(mu)
        self.sd.copy_(sd)
        self.coreset.copy_(core)
        self._statistically_initialized = True

    # ------------------------------------------------------------------ inference
    def forward(self, cube: Tensor, reference: Tensor | None = None, **_: Any) -> dict[str, Tensor]:
        """Score a BHWC grid against the fitted coreset; ``reference`` sets the output size."""
        require_fitted(self)
        b, h, w, c = cube.shape
        if reference is not None:
            h, w = int(reference.shape[1]), int(reference.shape[2])
        q = self._features(cube, self.stride)  # [B, C, gh, gw]
        gh, gw = q.shape[-2:]
        flat = q.permute(0, 2, 3, 1).reshape(-1, c)
        dist = self._nearest_distance(flat).reshape(b, 1, gh, gw)
        up = F.interpolate(dist, size=(h, w), mode="bilinear", align_corners=False)  # [B,1,H,W]
        scores = up.permute(0, 2, 3, 1).contiguous()
        return {"scores": scores, "anomaly_score": topk_mean(up, self.topk_frac)}
