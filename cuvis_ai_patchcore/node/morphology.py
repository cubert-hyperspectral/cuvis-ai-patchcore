"""Mask morphology: drop the connected components of a boolean mask below a pixel count.

An anomaly mask thresholded from a patch-level score map marks a real object with a blob of the
object's size plus a halo of a patch or more, while texture and noise on an empty background cross
the threshold in specks of a few dozen pixels. ``MaskMinArea`` keeps a blob only if it has at least
``min_area`` pixels (8-connected), so the specks go and the objects stay. Use it after a gate's
``decisions`` and before the viewers of the mask; the frame-level alarm is not changed by it.
``MaskBlobGate`` keeps a blob only if it holds at least ``min_px`` pixels of a second mask, e.g. a
``SpectralObjectMask``: an anomaly blob on the empty belt goes, one around an object stays.
``MaskPeakGate`` keeps a blob only if its highest score reaches ``ratio`` x the highest score of the
reference blob it lies in: after an anomaly mask is cut to the objects (``DecisionFusion("all")``
with a grown ``SpectralObjectMask``), the piece holding the mark's peak stays and halo pieces left
on neighbouring objects go. All work on a grid of ``cell`` x ``cell`` pixel cells (default 4): the
pixel counts (or score maxima) per cell are computed on the GPU, only the small cell grid is
labelled (8-connected, OpenCV on the CPU) and the decision per blob comes back as a cell mask. Areas
and gate counts are exact pixel counts; the only difference to labelling every pixel is that marks
whose cells touch form one blob (gaps of up to 2 x cell - 1 pixels). ``cell=1`` labels every pixel.
The nodes are not differentiable; the masks are boolean anyway.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor


def _cell_sums(x: Tensor, cell: int) -> Tensor:
    """Per-cell sums of a [N, C, H, W] float tensor (the last partial cells included)."""
    if cell == 1:
        return x
    return F.avg_pool2d(x, kernel_size=cell, stride=cell, ceil_mode=True, divisor_override=1)


def _label(occupied: np.ndarray) -> tuple[int, np.ndarray]:
    """8-connected labels of a 2-D bool grid; 16-bit labels (3x faster) whenever they cannot
    overflow: an h x w grid has at most ceil(h / 2) * ceil(w / 2) blobs."""
    h, w = occupied.shape
    ltype = cv2.CV_16U if ((h + 1) // 2) * ((w + 1) // 2) < 65535 else cv2.CV_32S
    return cv2.connectedComponents(occupied.astype(np.uint8), connectivity=8, ltype=ltype)


def _keep_blobs(occupied: np.ndarray, tests: list[tuple[np.ndarray, float]]) -> np.ndarray:
    """Cells of the 8-connected blobs of ``occupied`` that pass every (weights, threshold) test:
    the blob's summed weights reach the threshold. Sums run over the occupied cells only."""
    n, lab = _label(occupied)
    out = np.zeros(occupied.size, bool)
    if n <= 1:
        return out.reshape(occupied.shape)
    sel = occupied.ravel()
    ls = lab.ravel()[sel]  # the blob of each occupied cell (labels >= 1)
    keep = np.ones(n, bool)
    for weights, threshold in tests:
        keep &= np.bincount(ls, weights=weights.ravel()[sel], minlength=n) >= threshold
    out[sel] = keep[ls]
    return out.reshape(occupied.shape)


def _cell_max(x: Tensor, cell: int) -> Tensor:
    """Per-cell maxima of a [N, C, H, W] float tensor (the last partial cells included)."""
    if cell == 1:
        return x
    return F.max_pool2d(x, kernel_size=cell, stride=cell, ceil_mode=True)


def _segment_max(labels: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    """Per-label maxima of values (labels in [0, n)); -inf for labels without values. A sort and a
    reduceat: much faster than np.maximum.at."""
    out = np.full(n, -np.inf)
    if labels.size:
        order = np.argsort(labels, kind="stable")
        lab, val = labels[order], values[order]
        starts = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])
        out[lab[starts]] = np.maximum.reduceat(val, starts)
    return out


def _peak_keep(
    occupied: np.ndarray, ref: np.ndarray, peak: np.ndarray, ref_peak: np.ndarray, ratio: float
) -> np.ndarray:
    """Cells of the 8-connected blobs of ``occupied`` whose peak reaches ``ratio`` x the highest
    peak of the ``ref`` blobs they touch; a blob that touches no ``ref`` cell stays."""
    n, lab = _label(occupied)
    out = np.zeros(occupied.size, bool)
    if n <= 1:
        return out.reshape(occupied.shape)
    m, rlab = _label(ref)
    sel = occupied.ravel()
    ls = lab.ravel()[sel].astype(np.int64)
    rls = rlab.ravel()[sel].astype(np.int64)
    blob_peak = _segment_max(ls, peak.ravel()[sel].astype(np.float64), n)
    rsel = ref.ravel()
    ref_blob_peak = _segment_max(
        rlab.ravel()[rsel].astype(np.int64), ref_peak.ravel()[rsel].astype(np.float64), max(m, 1)
    )
    inside = rls > 0
    compare = _segment_max(ls[inside], ref_blob_peak[rls[inside]], n)
    keep = np.isneginf(compare) | (blob_peak >= ratio * compare)
    out[sel] = keep[ls]
    return out.reshape(occupied.shape)


def _expand_cells(keep: Tensor, x: Tensor, cell: int) -> Tensor:
    """x [N, H, W, C] (bool) where its cell of ``keep`` [N, h, w, C] is set."""
    if cell == 1:
        return x & keep
    n, h, w, c = x.shape
    if h % cell == 0 and w % cell == 0:  # a broadcast over the cells: one kernel, no copy of keep
        cells = x.reshape(n, h // cell, cell, w // cell, cell, c)
        return (cells & keep[:, :, None, :, None, :]).reshape(n, h, w, c)
    keep = keep.repeat_interleave(cell, dim=1).repeat_interleave(cell, dim=2)
    return x & keep[:, :h, :w]


def _filter_blobs(x: Tensor, weights: Tensor | None, threshold: float, cell: int) -> Tensor:
    """The blobs of x [N, C, H, W] (bool) whose summed weights [N, C, H, W] (None: x's own pixel
    counts) reach the threshold, labelled on the cell grid; one copy to the host."""
    counts = _cell_sums(x.to(torch.float32), cell)
    if weights is None:
        occ_np = w_np = counts.cpu().numpy()
    else:
        host = torch.stack([counts, _cell_sums(weights, cell)]).cpu().numpy()
        occ_np, w_np = host[0], host[1]
    occ_np = occ_np > 0
    keep = np.zeros(occ_np.shape, bool)
    for i in range(occ_np.shape[0]):
        for c in range(occ_np.shape[1]):
            if occ_np[i, c].any():
                keep[i, c] = _keep_blobs(occ_np[i, c], [(w_np[i, c], threshold)])
    k = torch.from_numpy(keep).to(device=x.device).permute(0, 2, 3, 1)
    return _expand_cells(k, x.permute(0, 2, 3, 1), cell).permute(0, 3, 1, 2)


class MaskMinArea(Node):
    """Drop the 8-connected blobs of a boolean mask [B, H, W, C] below ``min_area`` pixels."""

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.MASK, NodeTag.POSTPROCESSING, NodeTag.NUMPY})

    INPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, -1),
            description="Boolean mask [B, H, W, C], e.g. a FrameScoreGate's decisions; each frame "
            "and channel is filtered on its own.",
        ),
    }
    OUTPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, -1),
            description="Same shape: the blobs of `decisions` with at least `min_area` pixels "
            "(8-connected).",
        ),
    }

    def __init__(self, min_area: int = 250, cell: int = 4, **kwargs: Any) -> None:
        """Create a minimum-area mask filter.

        Parameters
        ----------
        min_area : smallest blob that is kept, in pixels (8-connected, default 250); ``0`` or ``1``
            keep every blob. Choose it below the smallest real object's blob (the object plus the
            score map's halo) and above the specks of the background.
        cell : side of the cells the blobs are labelled on (default 4; ``1`` labels every pixel).
        """
        if isinstance(min_area, bool) or not isinstance(min_area, int) or min_area < 0:
            raise ValueError(f"MaskMinArea: min_area must be an integer >= 0, got {min_area!r}.")
        if isinstance(cell, bool) or not isinstance(cell, int) or cell < 1:
            raise ValueError(f"MaskMinArea: cell must be an integer >= 1, got {cell!r}.")
        self.min_area = int(min_area)
        self.cell = int(cell)
        super().__init__(min_area=self.min_area, cell=self.cell, **kwargs)

    def forward(self, decisions: Tensor, **_: Any) -> dict[str, Tensor]:
        """Return the mask without its blobs smaller than ``min_area`` pixels."""
        if self.min_area <= 1 or decisions.numel() == 0 or not bool(decisions.any()):
            return {"decisions": decisions.clone()}
        x = decisions.permute(0, 3, 1, 2)
        out = _filter_blobs(x, None, float(self.min_area), self.cell)
        return {"decisions": out.permute(0, 2, 3, 1)}


class MaskBlobGate(Node):
    """Keep the blobs of a boolean mask that hold enough pixels of a second (gating) mask."""

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.MASK, NodeTag.POSTPROCESSING, NodeTag.NUMPY})

    INPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, -1),
            description="Boolean mask [B, H, W, C] whose blobs are gated, e.g. an anomaly mask.",
        ),
        "mask": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, -1),
            description="Gating mask [B, H, W, C'] of the same batch, height and width (a pixel "
            "counts where any channel is set), e.g. a SpectralObjectMask: where objects are.",
        ),
    }
    OUTPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, -1),
            description="Same shape as `decisions`: its 8-connected blobs that hold at least "
            "`min_px` pixels of `mask`.",
        ),
    }

    def __init__(self, min_px: int = 16, cell: int = 4, **kwargs: Any) -> None:
        """Create a blob gate.

        Parameters
        ----------
        min_px : pixels of the gating mask a blob needs to stay (default 16; ``0`` keeps every
            blob). With a SpectralObjectMask as the gate, a blob on the empty background goes and a
            blob around an object (its halo included) stays.
        cell : side of the cells the blobs are labelled on (default 4; ``1`` labels every pixel).
        """
        if isinstance(min_px, bool) or not isinstance(min_px, int) or min_px < 0:
            raise ValueError(f"MaskBlobGate: min_px must be an integer >= 0, got {min_px!r}.")
        if isinstance(cell, bool) or not isinstance(cell, int) or cell < 1:
            raise ValueError(f"MaskBlobGate: cell must be an integer >= 1, got {cell!r}.")
        self.min_px = int(min_px)
        self.cell = int(cell)
        super().__init__(min_px=self.min_px, cell=self.cell, **kwargs)

    def forward(self, decisions: Tensor, mask: Tensor, **_: Any) -> dict[str, Tensor]:
        """Return the blobs of ``decisions`` that overlap ``mask`` by at least ``min_px`` pixels."""
        gate = mask.any(dim=-1)
        if gate.shape != decisions.shape[:3]:
            raise ValueError(
                f"MaskBlobGate: mask is [B, H, W] = {tuple(gate.shape)}, decisions "
                f"{tuple(decisions.shape[:3])}."
            )
        if self.min_px == 0 or decisions.numel() == 0 or not bool(decisions.any()):
            return {"decisions": decisions.clone()}
        x = decisions.permute(0, 3, 1, 2)
        w = (x & gate[:, None]).to(torch.float32)
        out = _filter_blobs(x, w, float(self.min_px), self.cell)
        return {"decisions": out.permute(0, 2, 3, 1)}


class MaskPeakGate(Node):
    """Keep the blobs of a mask whose peak score reaches a share of their reference blob's peak."""

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.MASK, NodeTag.POSTPROCESSING, NodeTag.NUMPY})

    INPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, -1),
            description="Boolean mask [B, H, W, C] whose blobs are kept or dropped, e.g. the "
            "pieces of a mask after a cut.",
        ),
        "reference": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, -1),
            description="Boolean mask [B, H, W, C'] of the same batch, height and width (a pixel "
            "counts where any channel is set) whose blobs set the peak to reach, e.g. the mask "
            "before the cut.",
        ),
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Score map [B, H, W, C''] the peaks are read from (the first channel), "
            "e.g. the anomaly map the mask was thresholded from; finite and positive where marked.",
        ),
    }
    OUTPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, -1),
            description="Same shape as `decisions`: the blobs whose highest score reaches `ratio` "
            "times the highest score of the reference blob they lie in; blobs outside every "
            "reference blob stay.",
        ),
    }

    def __init__(self, ratio: float = 0.8, cell: int = 4, **kwargs: Any) -> None:
        """Create a peak gate.

        Parameters
        ----------
        ratio : share of the reference blob's peak a blob must reach to stay (default 0.8; ``0``
            keeps every blob). After a cut of an anomaly mask to the objects, the piece holding the
            mark's peak stays and halo pieces on neighbouring objects, far below it, go.
        cell : side of the cells the blobs are labelled on (default 4; ``1`` labels every pixel).
        """
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not 0.0 <= ratio <= 1.0
        ):
            raise ValueError(f"MaskPeakGate: ratio must be in [0, 1], got {ratio!r}.")
        if isinstance(cell, bool) or not isinstance(cell, int) or cell < 1:
            raise ValueError(f"MaskPeakGate: cell must be an integer >= 1, got {cell!r}.")
        self.ratio = float(ratio)
        self.cell = int(cell)
        super().__init__(ratio=self.ratio, cell=self.cell, **kwargs)

    def forward(
        self, decisions: Tensor, reference: Tensor, scores: Tensor, **_: Any
    ) -> dict[str, Tensor]:
        """Return the blobs of ``decisions`` whose peak reaches ``ratio`` x their reference peak."""
        ref = reference.any(dim=-1)
        if ref.shape != decisions.shape[:3] or tuple(scores.shape[:3]) != tuple(
            decisions.shape[:3]
        ):
            raise ValueError(
                f"MaskPeakGate: reference {tuple(ref.shape)} / scores {tuple(scores.shape[:3])} do "
                f"not match decisions [B, H, W] = {tuple(decisions.shape[:3])}."
            )
        if self.ratio == 0.0 or decisions.numel() == 0 or not bool(decisions.any()):
            return {"decisions": decisions.clone()}
        x = decisions.permute(0, 3, 1, 2)
        s = scores[..., :1].permute(0, 3, 1, 2).to(torch.float32)
        r = ref[:, None]
        low = torch.full_like(s, float("-inf"))
        peak = _cell_max(torch.where(x, s, low), self.cell)
        ref_peak = _cell_max(torch.where(r, s, low), self.cell)
        host = torch.cat([peak, ref_peak], dim=1).cpu().numpy()  # one copy to the host
        keep = np.zeros(peak.shape, bool)
        for i in range(keep.shape[0]):
            rp = host[i, -1]
            for c in range(keep.shape[1]):
                occ = host[i, c] > -np.inf  # a cell holds a mark iff its maximum is finite
                if occ.any():
                    keep[i, c] = _peak_keep(occ, rp > -np.inf, host[i, c], rp, self.ratio)
        kk = torch.from_numpy(keep).to(device=decisions.device).permute(0, 2, 3, 1)
        return {"decisions": _expand_cells(kk, decisions, self.cell)}
