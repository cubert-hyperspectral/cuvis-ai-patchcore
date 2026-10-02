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
        self._previous_any = False  # whether it has a set pixel (saves a GPU sync per frame)

    def reset(self) -> None:
        """Forget the previous frame, so the next frame shows nothing (e.g. a new recording)."""
        self._previous = None
        self._previous_any = False

    def _near(self, masks: Tensor) -> Tensor:
        """Square dilation by ``radius_px`` of boolean masks [N, H, W, C]: a box sum over the
        integral image (exact, one pass whatever the radius; int32 sums, frames < 2**31 px)."""
        if self.radius_px == 0:
            return masks
        r, k = self.radius_px, 2 * self.radius_px + 1
        x = masks.permute(0, 3, 1, 2).to(torch.int32)
        s = torch.nn.functional.pad(x, (r + 1, r, r + 1, r))
        s = s.cumsum(2, dtype=torch.int32).cumsum(3, dtype=torch.int32)  # int64 is 2x slower
        box = s[..., k:, k:] - s[..., :-k, k:] - s[..., k:, :-k] + s[..., :-k, :-k]
        return (box > 0).permute(0, 2, 3, 1)

    def forward(self, decisions: Tensor, **_: Any) -> dict[str, Tensor]:
        """Return each frame's mask pixels near the previous frame's mask; remember the last."""
        if decisions.shape[0] == 0:
            return {"decisions": decisions.clone()}
        prev, prev_any = self._previous, self._previous_any
        if prev is None or prev.shape != decisions.shape[1:] or prev.device != decisions.device:
            prev, prev_any = torch.zeros_like(decisions[0]), False
        if decisions.shape[0] == 1:
            now_any = bool(decisions.any())
            if not (now_any and prev_any):  # this frame or the one before is empty
                self._previous, self._previous_any = decisions[0].clone(), now_any
                return {"decisions": torch.zeros_like(decisions)}
        # each frame's predecessor: the remembered frame for the first, the batch's own after that
        if decisions.shape[0] == 1:
            before = prev.unsqueeze(0)
        else:
            before = torch.cat([prev.unsqueeze(0), decisions[:-1]], dim=0)
        self._previous = decisions[-1].clone()
        self._previous_any = True if decisions.shape[0] == 1 else bool(self._previous.any())
        return {"decisions": decisions & self._near(before)}
