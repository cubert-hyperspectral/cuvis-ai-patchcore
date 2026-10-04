"""Spectral objectness: which pixels of a hyperspectral frame are not the background material.

On a conveyor or a turntable the background (the belt) covers most of every frame, so the frame's
median spectrum is the belt's. ``SpectralObjectMask`` marks a pixel as an object where its spectral
angle to that median exceeds ``min_angle_deg``. The angle ignores brightness, so shadows and
vignetting stay background, and it compares every pixel with the same frame's belt, so an error in
the white reference or a light change shifts both alike. It is class-agnostic: any material that
differs from the belt counts, known or not; a foreign object with the belt's own spectrum (a piece
of the belt) does not. The angle is computed on a strided grid (``stride``, default every 4th pixel
in both axes) and expanded to the full size by nearest neighbour. Instead of the fixed
``min_angle_deg``, ``threshold="otsu"`` takes each frame's Otsu level of the angle map (in 0.1 deg
steps, clipped to ``[otsu_floor_deg, otsu_ceiling_deg]``), so a frame with darker or more objects
finds its own split; ``fill`` closes small gaps and fills holes of the objects on the stride grid,
and ``dilate_px`` grows the full-size mask, e.g. to keep a margin around each object.

``MaskBlobFilter`` is MaskMinArea, SpectralObjectMask and MaskBlobGate in one pass, for a live
pipeline: one labelling of the cell grid, both tests per blob, nothing done when the mask is empty,
and the spectral angle computed only at the cells that hold a mark.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai_patchcore.node.morphology import _cell_sums, _expand_cells, _keep_blobs

_THRESHOLDS = ("fixed", "otsu")


def _otsu_deg(angle: np.ndarray) -> float:
    """OpenCV's Otsu level of an angle map quantised to 0.1 deg (0-25.5 deg), in degrees."""
    q = np.clip(angle * 10.0, 0, 255).astype(np.uint8)
    level, _ = cv2.threshold(q, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(level) / 10.0


def _fill_holes(m: np.ndarray) -> np.ndarray:
    """m [h, w] (bool) with its holes filled: background not 4-connected to the border is set."""
    h, w = m.shape
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = m
    flood = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(pad, flood, (0, 0), 1, flags=4)  # the outside background becomes 1
    return m | (pad[1:-1, 1:-1] == 0)


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
            "above the threshold (on the stride grid, nearest neighbour to H, W), filled and grown "
            "when `fill` / `dilate_px` are set.",
        ),
        "angle": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="The angle in degrees on the stride grid [B, ceil(H / s), ceil(W / s), 1].",
        ),
    }

    def __init__(
        self,
        min_angle_deg: float = 6.0,
        stride: int = 4,
        median_stride: int = 8,
        threshold: str = "fixed",
        otsu_floor_deg: float = 3.0,
        otsu_ceiling_deg: float = 12.0,
        fill: bool = False,
        dilate_px: int = 0,
        **kwargs: Any,
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
        threshold : ``"fixed"`` (``min_angle_deg``, default) or ``"otsu"``: each frame's Otsu level
            of the angle map, clipped to ``[otsu_floor_deg, otsu_ceiling_deg]`` (defaults 3 / 12).
        fill : close 1-cell gaps (3 x 3) and fill the holes of the objects on the stride grid
            (default False).
        dilate_px : grow the full-size object mask by this many pixels (square window, default 0).
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
        if threshold not in _THRESHOLDS:
            raise ValueError(
                f"SpectralObjectMask: threshold must be one of {_THRESHOLDS}, got {threshold!r}."
            )
        for name, val in (
            ("otsu_floor_deg", otsu_floor_deg),
            ("otsu_ceiling_deg", otsu_ceiling_deg),
        ):
            if (
                isinstance(val, bool)
                or not isinstance(val, (int, float))
                or not 0.0 <= val <= 180.0
            ):
                raise ValueError(f"SpectralObjectMask: {name} must be in [0, 180], got {val!r}.")
        if otsu_floor_deg > otsu_ceiling_deg:
            raise ValueError("SpectralObjectMask: otsu_floor_deg must not exceed otsu_ceiling_deg.")
        if not isinstance(fill, bool):
            raise ValueError(f"SpectralObjectMask: fill must be a bool, got {fill!r}.")
        if isinstance(dilate_px, bool) or not isinstance(dilate_px, int) or dilate_px < 0:
            raise ValueError(
                f"SpectralObjectMask: dilate_px must be an integer >= 0, got {dilate_px!r}."
            )
        self.min_angle_deg = float(min_angle_deg)
        self.stride = int(stride)
        self.median_stride = int(median_stride)
        self.threshold = threshold
        self.otsu_floor_deg = float(otsu_floor_deg)
        self.otsu_ceiling_deg = float(otsu_ceiling_deg)
        self.fill = fill
        self.dilate_px = int(dilate_px)
        super().__init__(
            min_angle_deg=self.min_angle_deg,
            stride=self.stride,
            median_stride=self.median_stride,
            threshold=self.threshold,
            otsu_floor_deg=self.otsu_floor_deg,
            otsu_ceiling_deg=self.otsu_ceiling_deg,
            fill=self.fill,
            dilate_px=self.dilate_px,
            **kwargs,
        )

    def _objects(self, angle: Tensor) -> Tensor:
        """[B, 1, h, w] float (0 / 1) objects on the stride grid from the angle map [B, 1, h, w]."""
        if self.threshold == "fixed" and not self.fill:
            return (angle > self.min_angle_deg).to(torch.float32)
        a = angle[:, 0].detach().cpu().numpy()  # small grid: threshold, closing, holes on the CPU
        out = np.zeros(a.shape, np.float32)
        for i in range(a.shape[0]):
            thr = self.min_angle_deg
            if self.threshold == "otsu":
                thr = min(max(_otsu_deg(a[i]), self.otsu_floor_deg), self.otsu_ceiling_deg)
            o = a[i] > thr
            if self.fill:
                k = np.ones((3, 3), np.uint8)
                o = cv2.morphologyEx(o.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
                o = _fill_holes(o)
            out[i] = o
        return torch.from_numpy(out).to(device=angle.device)[:, None]

    def forward(self, cube: Tensor, **_: Any) -> dict[str, Tensor]:
        """Return the object mask at full size and the angle map on the stride grid."""
        b, h, w, c = cube.shape
        grid = cube[:, :: self.stride, :: self.stride, :].to(torch.float32)  # a strided view
        sample = cube[:, :: self.median_stride, :: self.median_stride, :].to(torch.float32)
        # [B, C]: the background's spectrum (the median along a contiguous axis: 4x faster on a GPU)
        ref = sample.permute(0, 3, 1, 2).reshape(b, c, -1).median(dim=2).values
        # per frame a matrix-vector product reads the strided grid once, without copying it
        dot = torch.stack([grid[i] @ ref[i] for i in range(b)])
        norm = torch.linalg.vector_norm(grid, dim=-1) * ref.norm(dim=-1)[:, None, None]
        cos = (dot / (norm + 1e-6)).clamp(-1.0, 1.0)
        angle = torch.rad2deg(torch.arccos(cos))[:, None]  # [B, 1, h / s, w / s]
        objects = self._objects(angle)
        r = self.dilate_px
        if r and r % self.stride == 0 and h % self.stride == 0 and w % self.stride == 0:
            # cell-aligned objects grown by r = k x stride pixels are the grid grown by k cells
            k = r // self.stride
            objects = F.max_pool2d(objects, kernel_size=2 * k + 1, stride=1, padding=k)
            r = 0
        full = F.interpolate(objects, size=(h, w), mode="nearest")
        if r:
            full = F.max_pool2d(full, kernel_size=2 * r + 1, stride=1, padding=r)
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
