"""Spectral objectness: which pixels of a hyperspectral frame are not the background material.

On a conveyor or a turntable the background (the belt) covers most of every frame, so the frame's
median spectrum is the belt's. ``SpectralObjectMask`` marks a pixel as an object where its spectral
angle to that median exceeds ``min_angle_deg``. The angle ignores brightness, so shadows and
vignetting stay background, and it compares every pixel with the same frame's belt, so an error in
the white reference or a light change shifts both alike. It is class-agnostic: any material that
differs from the belt counts, known or not; a foreign object with the belt's own spectrum (a piece
of the belt) does not. The angle is computed on a strided grid (``stride``, default every 4th pixel
in both axes) and expanded to the full size by nearest neighbour.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor


class SpectralObjectMask(Node):
    """Mark the pixels whose spectral angle to the frame's median spectrum exceeds a threshold."""

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.HYPERSPECTRAL, NodeTag.MASK, NodeTag.SEGMENTATION, NodeTag.TORCH})

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Hyperspectral cube [B, H, W, C] (e.g. reflectance) whose background "
            "covers most of the frame.",
        ),
    }
    OUTPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Object mask [B, H, W, 1]: spectral angle to the frame's median spectrum "
            "above `min_angle_deg` (on the stride grid, nearest neighbour to H, W).",
        ),
        "angle": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="The angle in degrees on the stride grid [B, ceil(H / s), ceil(W / s), 1].",
        ),
    }

    def __init__(self, min_angle_deg: float = 6.0, stride: int = 4, **kwargs: Any) -> None:
        """Create a spectral objectness mask.

        Parameters
        ----------
        min_angle_deg : smallest angle to the median spectrum that counts as an object, in degrees
            (default 6.0). On the walnut stand the belt lies within ~2-3 deg of its median and every
            labelled foreign-object material at 10.8 deg or more.
        stride : grid step in pixels on which the angle is computed (default 4, the frame cache's
            cube4 grid); 1 computes it on every pixel.
        """
        if (
            isinstance(min_angle_deg, bool)
            or not isinstance(min_angle_deg, (int, float))
            or not 0.0 <= float(min_angle_deg) <= 180.0
        ):
            raise ValueError(
                f"SpectralObjectMask: min_angle_deg must be in [0, 180], got {min_angle_deg!r}."
            )
        if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
            raise ValueError(f"SpectralObjectMask: stride must be an integer >= 1, got {stride!r}.")
        self.min_angle_deg = float(min_angle_deg)
        self.stride = int(stride)
        super().__init__(min_angle_deg=self.min_angle_deg, stride=self.stride, **kwargs)

    def forward(self, cube: Tensor, **_: Any) -> dict[str, Tensor]:
        """Return the object mask at full size and the angle map on the stride grid."""
        b, h, w, c = cube.shape
        grid = cube[:, :: self.stride, :: self.stride, :].to(torch.float32)
        gh, gw = grid.shape[1], grid.shape[2]
        x = grid.reshape(b, gh * gw, c)
        xs = x.sort(dim=1).values  # the median of each band over the grid: the belt's spectrum
        n = xs.shape[1]
        ref = 0.5 * (xs[:, (n - 1) // 2] + xs[:, n // 2])  # [B, C], numpy's median
        cos = (x * ref[:, None, :]).sum(-1) / (
            x.norm(dim=-1) * ref.norm(dim=-1, keepdim=True) + 1e-6
        )
        angle = torch.rad2deg(torch.arccos(cos.clamp(-1.0, 1.0))).reshape(b, 1, gh, gw)
        hot = (angle > self.min_angle_deg).to(torch.float32)
        full = F.interpolate(hot, size=(h, w), mode="nearest")
        return {"decisions": full.permute(0, 2, 3, 1) > 0.5, "angle": angle.permute(0, 2, 3, 1)}
