"""Spectral objectness: which pixels of a hyperspectral frame are not the background material.

On a conveyor or a turntable the background (the belt) covers most of every frame, so the frame's
median spectrum is the belt's. ``SpectralObjectMask`` marks a pixel as an object where its spectral
angle to that median exceeds ``min_angle_deg``. The angle ignores brightness, so shadows and
vignetting stay background, and it compares every pixel with the same frame's belt, so an error in
the white reference or a light change shifts both alike. It is class-agnostic: any material that
differs from the belt counts, known or not; a foreign object with the belt's own spectrum (a piece
of the belt) does not. The angle is computed on a strided grid (``stride``, default every 4th pixel
in both axes) and expanded to the full size by nearest neighbour.

``MaskBlobFilter`` is MaskMinArea, SpectralObjectMask and MaskBlobGate in one pass, for a live
pipeline: one labelling of the cell grid, both tests per blob, nothing done when the mask is empty,
and the spectral angle computed only at the cells that hold a mark.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai_patchcore.node.morphology import _cell_sums, _expand_cells, _keep_blobs


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

    def __init__(
        self, min_angle_deg: float = 6.0, stride: int = 4, median_stride: int = 8, **kwargs: Any
    ) -> None:
        """Create a spectral objectness mask.

        Parameters
        ----------
        min_angle_deg : smallest angle to the median spectrum that counts as an object, in degrees
            (default 6.0). On the walnut stand the belt lies within ~2-3 deg of its median and every
            labelled foreign-object material at 10.8 deg or more.
        stride : grid step in pixels on which the angle is computed (default 4, the frame cache's
            cube4 grid); 1 computes it on every pixel.
        median_stride : grid step of the pixels whose per-band median is the background spectrum
            (default 8: 16 875 pixels of a 1000 x 1080 frame, plenty for a median).
        """
        if (
            isinstance(min_angle_deg, bool)
            or not isinstance(min_angle_deg, (int, float))
            or not 0.0 <= float(min_angle_deg) <= 180.0
        ):
            raise ValueError(
                f"SpectralObjectMask: min_angle_deg must be in [0, 180], got {min_angle_deg!r}."
            )
        for name, val in (("stride", stride), ("median_stride", median_stride)):
            if isinstance(val, bool) or not isinstance(val, int) or val < 1:
                raise ValueError(
                    f"SpectralObjectMask: {name} must be an integer >= 1, got {val!r}."
                )
        self.min_angle_deg = float(min_angle_deg)
        self.stride = int(stride)
        self.median_stride = int(median_stride)
        super().__init__(
            min_angle_deg=self.min_angle_deg,
            stride=self.stride,
            median_stride=self.median_stride,
            **kwargs,
        )

    def forward(self, cube: Tensor, **_: Any) -> dict[str, Tensor]:
        """Return the object mask at full size and the angle map on the stride grid."""
        b, h, w, c = cube.shape
        grid = cube[:, :: self.stride, :: self.stride, :].to(torch.float32)
        gh, gw = grid.shape[1], grid.shape[2]
        x = grid.reshape(b, gh * gw, c)
        sample = cube[:, :: self.median_stride, :: self.median_stride, :].to(torch.float32)
        # [B, C]: the background's spectrum (median along the last, contiguous axis: 4x faster)
        ref = sample.reshape(b, -1, c).transpose(1, 2).contiguous().median(dim=2).values
        cos = (x * ref[:, None, :]).sum(-1) / (
            x.norm(dim=-1) * ref.norm(dim=-1, keepdim=True) + 1e-6
        )
        angle = torch.rad2deg(torch.arccos(cos.clamp(-1.0, 1.0))).reshape(b, 1, gh, gw)
        hot = (angle > self.min_angle_deg).to(torch.float32)
        full = F.interpolate(hot, size=(h, w), mode="nearest")
        return {"decisions": full.permute(0, 2, 3, 1) > 0.5, "angle": angle.permute(0, 2, 3, 1)}


class MaskBlobFilter(Node):
    """Keep the blobs of a boolean mask that are large enough and lie on non-background material."""

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.MASK, NodeTag.POSTPROCESSING, NodeTag.HYPERSPECTRAL, NodeTag.NUMPY})

    INPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, -1),
            description="Boolean mask [B, H, W, C] whose blobs are filtered, e.g. a "
            "FrameScoreGate's decisions; each frame and channel on its own.",
        ),
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            optional=True,
            description="Hyperspectral cube [B, H, W, K] of the same frames, for the object test; "
            "without it only the size test runs.",
        ),
    }
    OUTPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, -1),
            description="Same shape: the blobs of `decisions` with at least `min_area` pixels that "
            "hold at least `min_object_px` pixels of non-background material.",
        ),
    }

    def __init__(
        self,
        min_area: int = 250,
        min_object_px: int = 16,
        min_angle_deg: float = 6.0,
        cell: int = 4,
        median_stride: int = 8,
        **kwargs: Any,
    ) -> None:
        """Create a blob filter: MaskMinArea, SpectralObjectMask and MaskBlobGate in one pass.

        Parameters
        ----------
        min_area : smallest blob that is kept, in pixels (default 250; 0 or 1 keep every size).
        min_object_px : pixels of the blob that must be objects (default 16; 0 skips the test).
        min_angle_deg : spectral angle to the frame's median spectrum above which a pixel is an
            object (default 6.0).
        cell : the blobs are labelled on cells of cell x cell pixels (default 4) and the object test
            reads each cell's first pixel (the stride-``cell`` grid of SpectralObjectMask).
        median_stride : grid step of the pixels whose per-band median is the background spectrum
            (default 8).
        """
        for name, val, lo in (
            ("min_area", min_area, 0),
            ("min_object_px", min_object_px, 0),
            ("cell", cell, 1),
            ("median_stride", median_stride, 1),
        ):
            if isinstance(val, bool) or not isinstance(val, int) or val < lo:
                raise ValueError(f"MaskBlobFilter: {name} must be an integer >= {lo}, got {val!r}.")
        if (
            isinstance(min_angle_deg, bool)
            or not isinstance(min_angle_deg, (int, float))
            or not 0.0 <= float(min_angle_deg) <= 180.0
        ):
            raise ValueError(
                f"MaskBlobFilter: min_angle_deg must be in [0, 180], got {min_angle_deg!r}."
            )
        self.min_area = int(min_area)
        self.min_object_px = int(min_object_px)
        self.min_angle_deg = float(min_angle_deg)
        self.cell = int(cell)
        self.median_stride = int(median_stride)
        super().__init__(
            min_area=self.min_area,
            min_object_px=self.min_object_px,
            min_angle_deg=self.min_angle_deg,
            cell=self.cell,
            median_stride=self.median_stride,
            **kwargs,
        )

    def _object_cells(self, cube: Tensor) -> Tensor:
        """[B, h, w] bool on the cell grid: the cells whose first pixel is not background, i.e. has
        a spectral angle above ``min_angle_deg`` to the frame's median spectrum."""
        b, k = cube.shape[0], cube.shape[-1]
        m = self.median_stride
        sample = cube[:, ::m, ::m, :].to(torch.float32).permute(0, 3, 1, 2).reshape(b, k, -1)
        ref = sample.median(dim=2).values  # [B, K]: along the contiguous axis, 4x faster on a GPU
        px = cube[:, :: self.cell, :: self.cell, :].to(torch.float32)  # [B, h, w, K]
        dot = torch.einsum("bhwk,bk->bhw", px, ref)
        norm = torch.linalg.vector_norm(px, dim=-1) * ref.norm(dim=-1)[:, None, None]
        cos = (dot / (norm + 1e-6)).clamp(-1.0, 1.0)
        return cos < math.cos(math.radians(self.min_angle_deg))  # angle > a <=> cos < cos(a)

    def forward(self, decisions: Tensor, cube: Tensor | None = None, **_: Any) -> dict[str, Tensor]:
        """Return the blobs that pass the size test and, with a cube, the object test."""
        if decisions.numel() == 0 or not bool(decisions.any()):
            return {"decisions": decisions.clone()}
        counts = _cell_sums(decisions.permute(0, 3, 1, 2).to(torch.float32), self.cell)
        small = torch.uint8 if self.cell * self.cell < 256 else torch.int32
        planes = [counts.to(small)]  # marked pixels per cell, [B, C, h, w]
        if cube is not None and self.min_object_px > 0:
            obj = self._object_cells(cube)
            planes.append((counts * obj[:, None]).to(small))  # marked object pixels per cell
        host = torch.stack(planes).cpu().numpy()  # the one copy to the host
        keep = np.zeros(host.shape[1:], bool)
        for i in range(keep.shape[0]):
            for c in range(keep.shape[1]):
                occ = host[0, i, c] > 0
                if not occ.any():
                    continue
                tests = [(host[0, i, c], max(self.min_area, 1))]
                if len(planes) > 1:
                    tests.append((host[1, i, c], self.min_object_px))
                keep[i, c] = _keep_blobs(occ, tests)
        kk = torch.from_numpy(keep).to(device=decisions.device).permute(0, 2, 3, 1)
        return {"decisions": _expand_cells(kk, decisions, self.cell)}
