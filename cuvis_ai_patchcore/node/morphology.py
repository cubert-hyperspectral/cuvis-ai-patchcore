"""Mask morphology: drop the connected components of a boolean mask below a pixel count.

An anomaly mask thresholded from a patch-level score map marks a real object with a blob of the
object's size plus a halo of a patch or more, while texture and noise on an empty background cross
the threshold in specks of a few dozen pixels. ``MaskMinArea`` keeps a blob only if it has at least
``min_area`` pixels (8-connected), so the specks go and the objects stay. Use it after a gate's
``decisions`` and before the viewers of the mask; the frame-level alarm is not changed by it.
``MaskBlobGate`` keeps a blob only if it holds at least ``min_px`` pixels of a second mask, e.g. a
``SpectralObjectMask``: an anomaly blob on the empty belt goes, one around an object stays.
Connected components are labelled with OpenCV on the CPU (the numpy path of cuvis-ai's mask_ops), so
the nodes are not differentiable; the masks are boolean anyway.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor


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

    def __init__(self, min_area: int = 250, **kwargs: Any) -> None:
        """Create a minimum-area mask filter.

        Parameters
        ----------
        min_area : smallest blob that is kept, in pixels (8-connected, default 250); ``0`` or ``1``
            keep every blob. Choose it below the smallest real object's blob (the object plus the
            score map's halo) and above the specks of the background.
        """
        if isinstance(min_area, bool) or not isinstance(min_area, int) or min_area < 0:
            raise ValueError(f"MaskMinArea: min_area must be an integer >= 0, got {min_area!r}.")
        self.min_area = int(min_area)
        super().__init__(min_area=self.min_area, **kwargs)

    def forward(self, decisions: Tensor, **_: Any) -> dict[str, Tensor]:
        """Return the mask without its blobs smaller than ``min_area`` pixels."""
        if self.min_area <= 1 or decisions.numel() == 0:
            return {"decisions": decisions.clone()}
        masks = decisions.detach().cpu().numpy()
        out = np.zeros_like(masks)
        for b in range(masks.shape[0]):
            for c in range(masks.shape[3]):
                m = masks[b, :, :, c]
                if not m.any():
                    continue
                _, lab, stats, _ = cv2.connectedComponentsWithStats(
                    m.astype(np.uint8), connectivity=8
                )
                keep = stats[:, cv2.CC_STAT_AREA] >= self.min_area
                keep[0] = False  # the background component
                out[b, :, :, c] = keep[lab]
        return {"decisions": torch.from_numpy(out).to(device=decisions.device)}


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

    def __init__(self, min_px: int = 16, **kwargs: Any) -> None:
        """Create a blob gate.

        Parameters
        ----------
        min_px : pixels of the gating mask a blob needs to stay (default 16; ``0`` keeps every
            blob). With a SpectralObjectMask as the gate, a blob on the empty background goes and a
            blob around an object (its halo included) stays.
        """
        if isinstance(min_px, bool) or not isinstance(min_px, int) or min_px < 0:
            raise ValueError(f"MaskBlobGate: min_px must be an integer >= 0, got {min_px!r}.")
        self.min_px = int(min_px)
        super().__init__(min_px=self.min_px, **kwargs)

    def forward(self, decisions: Tensor, mask: Tensor, **_: Any) -> dict[str, Tensor]:
        """Return the blobs of ``decisions`` that overlap ``mask`` by at least ``min_px`` pixels."""
        gate = mask.any(dim=-1)
        if gate.shape != decisions.shape[:3]:
            raise ValueError(
                f"MaskBlobGate: mask is [B, H, W] = {tuple(gate.shape)}, decisions "
                f"{tuple(decisions.shape[:3])}."
            )
        if self.min_px == 0 or decisions.numel() == 0:
            return {"decisions": decisions.clone()}
        masks = decisions.detach().cpu().numpy()
        g = gate.detach().cpu().numpy()
        out = np.zeros_like(masks)
        for b in range(masks.shape[0]):
            for c in range(masks.shape[3]):
                m = masks[b, :, :, c]
                if not m.any():
                    continue
                n, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
                hits = np.bincount(
                    lab.ravel(), weights=g[b].ravel().astype(np.float64), minlength=n
                )
                keep = hits >= self.min_px
                keep[0] = False  # the background component
                out[b, :, :, c] = keep[lab]
        return {"decisions": torch.from_numpy(out).to(device=decisions.device)}
