"""Score-range calibration — one anomaly map onto its NORMAL range, unclamped above.

Fusing detectors needs their maps on a common scale, and the scale that works is each detector's
normal range: the 1st and 99th percentile of its scores over normal frames map to 0 and 1, so "1"
means "as high as the top percent of normal pixels" for every detector alike, while an anomaly that
scores far above the normal range keeps its rank because nothing is clamped above. A min-max range
is set by the single most extreme normal pixel (one specular highlight squashes a whole channel);
a clamped percentile range saturates every anomaly, and everything else, on a drifted session.
Both were measured to break the two-bank walnut fusion; this node is the calibration that
reproduces the validated result, fitted on normal frames only (Phase 1).
"""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor


class ScoreRangeNormalizer(Node):
    """Map score maps onto their fitted normal range, (x - p_low) / (p_high - p_low), unclamped."""

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.ANOMALY, NodeTag.NORMALIZATION, NodeTag.TORCH, NodeTag.STATEFUL})

    INPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Score maps [B, H, W, C] (anomaly maps: C = 1); C must equal n_channels.",
        ),
    }
    OUTPUT_SPECS = {
        "normalized": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Same shape: (x - p_low) / (p_high - p_low) per channel with the fitted "
            "normal-range percentiles, floored at 0 when `floor` is set, never clamped above.",
        ),
    }

    def __init__(
        self,
        n_channels: int = 1,
        low: float = 1.0,
        high: float = 99.0,
        floor: bool = True,
        fit_subsample: int = 4,
        max_fit_values: int = 4_000_000,
        seed: int = 0,
        eps: float = 1e-9,
        **kwargs: Any,
    ) -> None:
        """Create an unfitted calibrator; the bound buffers are sized from ``n_channels``.

        Parameters
        ----------
        n_channels : channels ``C`` of the score maps (1 for an anomaly map).
        low, high : percentiles in ``[0, 100]`` of the pooled normal scores mapped to 0 and 1.
        floor : clip the output below at 0 (values under the low percentile are "normal").
        fit_subsample : spatial stride while collecting scores in Phase 1 (every ``s``-th pixel).
        max_fit_values : seeded random cap on the collected values before the percentiles; bounds
            Phase-1 memory (``torch.quantile`` handles up to 16M values).
        seed : RNG seed of the cap.
        eps : floor for the ``(p_high - p_low)`` denominator.
        """
        if isinstance(n_channels, bool) or int(n_channels) < 1:
            raise ValueError(f"n_channels must be a positive integer, got {n_channels}")
        if not 0.0 <= float(low) < float(high) <= 100.0:
            raise ValueError(f"require 0 <= low < high <= 100, got {low}, {high}")
        if int(fit_subsample) < 1:
            raise ValueError(f"fit_subsample must be >= 1, got {fit_subsample}")
        if int(max_fit_values) < 2:
            raise ValueError(f"max_fit_values must be >= 2, got {max_fit_values}")
        self.n_channels = int(n_channels)
        self.low = float(low)
        self.high = float(high)
        self.floor = bool(floor)
        self.fit_subsample = int(fit_subsample)
        self.max_fit_values = int(max_fit_values)
        self.seed = int(seed)
        self.eps = float(eps)
        super().__init__(
            n_channels=self.n_channels,
            low=self.low,
            high=self.high,
            floor=self.floor,
            fit_subsample=self.fit_subsample,
            max_fit_values=self.max_fit_values,
            seed=self.seed,
            eps=self.eps,
            **kwargs,
        )
        self.register_buffer("lo", torch.zeros(self.n_channels, dtype=torch.float32))
        self.register_buffer("hi", torch.ones(self.n_channels, dtype=torch.float32))

    def _thin(self, vals: Tensor, gen: torch.Generator) -> Tensor:
        """Seeded random cap of the collected values to ``max_fit_values`` rows."""
        if vals.shape[0] <= self.max_fit_values:
            return vals
        keep = torch.randperm(vals.shape[0], generator=gen)[: self.max_fit_values]
        return vals[keep.to(vals.device)]

    # ------------------------------------------------------------------ phase 1
    @torch.no_grad()
    def statistical_initialization(self, input_stream) -> None:
        """Pool the (subsampled, capped) normal scores; store their low / high percentiles."""
        self._statistically_initialized = False
        gen = torch.Generator().manual_seed(self.seed)
        s = self.fit_subsample
        chunks: list[Tensor] = []
        total = 0
        for batch in input_stream:
            x = batch.get("scores") if isinstance(batch, dict) else None
            if x is None:
                continue
            if x.shape[-1] != self.n_channels:
                raise ValueError(
                    f"{type(self).__name__}: scores have {x.shape[-1]} channels, "
                    f"n_channels={self.n_channels}"
                )
            v = x[:, ::s, ::s, :].reshape(-1, self.n_channels).float()
            chunks.append(v)
            total += v.shape[0]
            if total > 2 * self.max_fit_values:  # bound memory while streaming
                chunks = [self._thin(torch.cat(chunks, dim=0), gen)]
                total = chunks[0].shape[0]
        if not chunks:
            raise RuntimeError(
                f"{type(self).__name__}.statistical_initialization() received no score maps."
            )
        vals = self._thin(torch.cat(chunks, dim=0), gen)
        if vals.shape[0] < 2:
            raise RuntimeError(
                f"{type(self).__name__}.statistical_initialization() needs at least 2 values."
            )
        q = torch.tensor(
            [self.low / 100.0, self.high / 100.0], dtype=vals.dtype, device=vals.device
        )
        pct = torch.quantile(vals, q, dim=0)  # [2, C], linear interpolation (numpy's default)
        self.lo.copy_(pct[0])
        self.hi.copy_(pct[1])
        self._statistically_initialized = True

    # ------------------------------------------------------------------ inference
    def forward(self, scores: Tensor, **_: Any) -> dict[str, Tensor]:
        """Calibrate the maps to the fitted normal range."""
        if not self._statistically_initialized:
            raise RuntimeError(
                f"{type(self).__name__} requires statistical_initialization() (Phase 1) or "
                "loaded weights before forward()."
            )
        out = (scores - self.lo) / (self.hi - self.lo).clamp_min(self.eps)
        if self.floor:
            out = out.clamp_min(0.0)
        return {"normalized": out}
