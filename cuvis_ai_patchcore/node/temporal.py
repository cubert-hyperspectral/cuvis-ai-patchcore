"""Frame-to-frame mask persistence: keep a mask pixel only where the previous frame's mask was near.

A per-frame detector on a moving scene (objects on a turntable or a belt) marks two kinds of blobs:
real objects, which stay in view and move a bounded distance from one frame to the next, and
single-frame flickers (sensor noise, motion blur, a reflection). ``MaskPersistence`` keeps a pixel
of the current mask only if the previous frame's mask has a set pixel within ``radius_px`` of it
(square window), so an object shows from its second frame on while a one-frame flicker never
shows. Set ``radius_px`` to at least the largest distance an object moves between two frames.

The node keeps runtime state: the last frame's input mask (not serialized, so a saved pipeline
holds no extra weights). Frames of a batch are taken in batch order, after the frames of earlier
calls. The first frame after construction or ``reset()``, and the first frame after the mask's
height, width, channel count or device change, show nothing. Torch-native; the masks are boolean,
so the node is not differentiable.
"""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor


class MaskPersistence(Node):
    """Keep the pixels of a boolean mask [B, H, W, C] that the previous frame's mask has nearby."""

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.MASK, NodeTag.POSTPROCESSING, NodeTag.STATEFUL, NodeTag.TORCH})

    INPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, -1),
            description="Boolean mask [B, H, W, C] of consecutive frames (batch order is time "
            "order), e.g. a FrameScoreGate's decisions; each channel is filtered on its own.",
        ),
    }
    OUTPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, -1),
            description="Same shape: the pixels of `decisions` with a set pixel of the previous "
            "frame's mask (same channel) within `radius_px`; all False on the first frame.",
        ),
    }

    def __init__(self, radius_px: int = 40, **kwargs: Any) -> None:
        """Create a mask persistence filter.

        Parameters
        ----------
        radius_px : how far, in pixels, the previous frame's mask may lie from a pixel that is kept
            (square window of side ``2 * radius_px + 1``, default 40). At least the largest distance
            an object moves between two frames; larger lets more flickers next to real objects
            through. ``0`` keeps only the pixels set in both frames.
        """
        if isinstance(radius_px, bool) or not isinstance(radius_px, int) or radius_px < 0:
            raise ValueError(
                f"MaskPersistence: radius_px must be an integer >= 0, got {radius_px!r}."
            )
        self.radius_px = int(radius_px)
        super().__init__(radius_px=self.radius_px, **kwargs)
        self._previous: Tensor | None = None  # the last frame's input mask (runtime only)

    def reset(self) -> None:
        """Forget the previous frame, so the next frame shows nothing (e.g. a new recording)."""
        self._previous = None

    def _near(self, masks: Tensor) -> Tensor:
        """Square dilation by ``radius_px`` of boolean masks [N, H, W, C] (separable max-pool)."""
        if self.radius_px == 0:
            return masks
        k, r = 2 * self.radius_px + 1, self.radius_px
        x = masks.permute(0, 3, 1, 2).to(torch.float32)
        # max over the square window = max over its rows, then over its columns; the implicit
        # padding never wins the max, so the image border adds nothing
        x = torch.nn.functional.max_pool2d(x, kernel_size=(1, k), stride=1, padding=(0, r))
        x = torch.nn.functional.max_pool2d(x, kernel_size=(k, 1), stride=1, padding=(r, 0))
        return (x > 0).permute(0, 2, 3, 1)

    def forward(self, decisions: Tensor, **_: Any) -> dict[str, Tensor]:
        """Return each frame's mask pixels near the previous frame's mask; remember the last."""
        if decisions.shape[0] == 0:
            return {"decisions": decisions.clone()}
        prev = self._previous
        if prev is None or prev.shape != decisions.shape[1:] or prev.device != decisions.device:
            prev = torch.zeros_like(decisions[0])
        # each frame's predecessor: the remembered frame for the first, the batch's own after that
        before = torch.cat([prev.unsqueeze(0), decisions[:-1]], dim=0)
        self._previous = decisions[-1].clone()
        return {"decisions": decisions & self._near(before)}
