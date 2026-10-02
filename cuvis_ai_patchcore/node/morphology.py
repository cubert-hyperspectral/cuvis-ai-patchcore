"""Mask morphology: drop the connected components of a boolean mask below a pixel count.

An anomaly mask thresholded from a patch-level score map marks a real object with a blob of the
object's size plus a halo of a patch or more, while texture and noise on an empty background cross
the threshold in specks of a few dozen pixels. ``MaskMinArea`` keeps a blob only if it has at least
``min_area`` pixels (8-connected), so the specks go and the objects stay. Use it after a gate's
``decisions`` and before the viewers of the mask; the frame-level alarm is not changed by it.
Connected components are labelled with OpenCV on the CPU (the numpy path of cuvis-ai's mask_ops), so
the node is not differentiable; the masks are boolean anyway.
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
