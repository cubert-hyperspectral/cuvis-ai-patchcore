"""Score-map fusion — one score map from several, by a fixed elementwise rule.

Averaging the calibrated maps of complementary detectors (e.g. a PatchCore bank on raw spectra and
a PatchCore bank on ViT features) cancels each detector's private noise while keeping the signal
both agree on; ``min`` is the hard AND, ``max`` the OR, ``wmean`` a weighted average, ``softmin`` a
soft AND between the minimum and the mean (a pixel stays hot only where the detectors agree, e.g. a
spectral model and an image model, which removes each one's private false alarms). ``first`` is a
priority rule for gated maps: per frame it takes the first inbound map that is not all zero, e.g.
one detector's display map whenever its gate opens and a second detector's only on frames the first
misses. The node is stateless, torch-native and differentiable. Feed it maps on a common scale (a
fitted normalizer per detector), otherwise the detector with the widest range dominates.

``DecisionFusion`` is the counterpart for boolean masks (e.g. the ``decisions`` of several
``FrameScoreGate`` nodes): ``any`` / ``all`` combine them pixel-wise, ``first`` takes per frame the
first inbound mask with a set pixel, the mask of the map a ``first`` score fusion displays.

``MaskComposite`` merges several boolean masks for display: a label map (e.g. 1 = shell, 2 = foreign
object) and a level map, so that one output shows what separate masks showed one at a time.
"""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

_MODES = ("mean", "min", "max", "wmean", "first", "softmin")


class ScoreMapFusion(Node):
    """Fuse N score maps [B, H, W, 1] into one by a fixed elementwise rule (``mode``).

    Modes: mean | min | max | wmean | first | softmin; one map per inbound connection, identical
    shapes; see ``__init__`` for the rules.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.ANOMALY, NodeTag.TORCH})

    INPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            variadic=True,
            description="Score maps [B, H, W, 1] of identical shape, one per inbound connection "
            "(fan-in); connect every detector's normalized map to this port.",
        ),
    }
    OUTPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Fused score map [B, H, W, 1] per ``mode``.",
        ),
    }

    def __init__(
        self,
        mode: str = "mean",
        weights: list[float] | None = None,
        beta: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a fusion node.

        Parameters
        ----------
        mode : ``"mean"`` (arithmetic average), ``"min"`` (AND), ``"max"`` (OR), ``"wmean"``
            (weighted average with ``weights``, normalised to sum to one), ``"first"`` (per frame,
            the first inbound map in connection order that is not all zero; zeros if all are) or
            ``"softmin"`` (soft minimum ``-(1/beta) log(sum_i w_i exp(-beta x_i))`` with the weights
            normalised to sum to one, equal weights by default: the minimum plus at most
            ``log(N) / beta`` for equal weights, the (weighted) mean as ``beta`` goes to 0).
        weights : one non-negative weight per inbound map, required for ``"wmean"``, optional for
            ``"softmin"`` and rejected for the other modes; the count is checked against the inbound
            maps at run time.
        beta : the soft minimum's sharpness (> 0), in the units of the inbound maps; required for
            ``"softmin"`` and rejected for the other modes.
        """
        if mode not in _MODES:
            raise ValueError(f"ScoreMapFusion: mode must be one of {_MODES}, got {mode!r}.")
        if mode == "wmean" and not weights:
            raise ValueError("ScoreMapFusion: mode 'wmean' requires weights.")
        if mode in ("wmean", "softmin") and weights is not None:
            if any(float(x) < 0.0 for x in weights) or float(sum(weights)) <= 0.0:
                raise ValueError(
                    "ScoreMapFusion: weights must be non-negative with a positive sum."
                )
        elif weights is not None:
            raise ValueError(
                f"ScoreMapFusion: weights are only used with mode 'wmean' or 'softmin', "
                f"not {mode!r}."
            )
        if mode == "softmin":
            if beta is None or not 0.0 < float(beta) < float("inf"):
                raise ValueError(f"ScoreMapFusion: mode 'softmin' requires beta > 0, got {beta!r}.")
        elif beta is not None:
            raise ValueError(
                f"ScoreMapFusion: beta is only used with mode 'softmin', not {mode!r}."
            )
        self.mode = mode
        self.weights = [float(x) for x in weights] if weights is not None else None
        self.beta = float(beta) if beta is not None else None
        super().__init__(mode=self.mode, weights=self.weights, beta=self.beta, **kwargs)

    def forward(self, scores: list[Tensor] | Tensor, **_: Any) -> dict[str, Tensor]:
        """Return the fused map; every inbound map must share one shape."""
        maps = list(scores) if isinstance(scores, (list, tuple)) else [scores]
        if not maps:
            raise ValueError("ScoreMapFusion: no score maps connected.")
        shape = maps[0].shape
        for i, m in enumerate(maps[1:], start=1):
            if m.shape != shape:
                raise ValueError(
                    f"ScoreMapFusion: map {i} has shape {tuple(m.shape)}, expected {tuple(shape)}."
                )
        stack = torch.stack(maps, dim=0)  # [N, B, H, W, 1]
        if self.mode == "mean":
            out = stack.mean(dim=0)
        elif self.mode == "min":
            out = stack.amin(dim=0)
        elif self.mode == "max":
            out = stack.amax(dim=0)
        elif self.mode == "first":  # priority: per frame, the first map that is not all zero
            live = stack.flatten(start_dim=2).ne(0).any(dim=2)  # [N, B]
            pick = live.to(torch.int64).argmax(dim=0)  # [B]: first live map (0 when none is live)
            out = stack[pick, torch.arange(stack.shape[1], device=stack.device)]
        elif self.mode == "softmin":  # lo - log(sum_i w_i exp(-beta (x_i - lo))) / beta, stable
            if self.weights is not None and len(self.weights) != len(maps):
                raise ValueError(
                    f"ScoreMapFusion: {len(self.weights)} weights for {len(maps)} score maps."
                )
            w = torch.tensor(
                self.weights or [1.0] * len(maps), dtype=stack.dtype, device=stack.device
            )
            w = (w / w.sum()).view(-1, 1, 1, 1, 1)
            lo = stack.amin(dim=0)
            out = lo - torch.log((w * torch.exp(-self.beta * (stack - lo))).sum(dim=0)) / self.beta
        else:  # wmean
            if len(self.weights) != len(maps):
                raise ValueError(
                    f"ScoreMapFusion: {len(self.weights)} weights for {len(maps)} score maps."
                )
            w = torch.tensor(self.weights, dtype=stack.dtype, device=stack.device)
            w = (w / w.sum()).view(-1, 1, 1, 1, 1)
            out = (stack * w).sum(dim=0)
        return {"scores": out}


_DECISION_MODES = ("any", "all", "first")


class DecisionFusion(Node):
    """Fuse N boolean masks [B, H, W, C] into one by ``mode`` (any | all | first)."""

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.MASK, NodeTag.TORCH})

    INPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, -1),
            variadic=True,
            description="Boolean masks [B, H, W, C] of identical shape, one per inbound connection "
            "(fan-in), e.g. the `decisions` of several gates.",
        ),
    }
    OUTPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, -1),
            description="Fused mask [B, H, W, C] per ``mode``.",
        ),
    }

    def __init__(self, mode: str = "any", **kwargs: Any) -> None:
        """Create a mask fusion node.

        Parameters
        ----------
        mode : ``"any"`` (pixel-wise OR), ``"all"`` (pixel-wise AND) or ``"first"`` (per frame, the
            first inbound mask in connection order with a set pixel; all False if none has one).
        """
        if mode not in _DECISION_MODES:
            raise ValueError(
                f"DecisionFusion: mode must be one of {_DECISION_MODES}, got {mode!r}."
            )
        self.mode = mode
        super().__init__(mode=self.mode, **kwargs)

    def forward(self, decisions: list[Tensor] | Tensor, **_: Any) -> dict[str, Tensor]:
        """Return the fused mask; every inbound mask must share one shape."""
        masks = list(decisions) if isinstance(decisions, (list, tuple)) else [decisions]
        if not masks:
            raise ValueError("DecisionFusion: no masks connected.")
        shape = masks[0].shape
        for i, m in enumerate(masks[1:], start=1):
            if m.shape != shape:
                raise ValueError(
                    f"DecisionFusion: mask {i} has shape {tuple(m.shape)}, expected {tuple(shape)}."
                )
        stack = torch.stack(masks, dim=0)  # [N, B, H, W, C]
        if self.mode == "any":
            out = stack.any(dim=0)
        elif self.mode == "all":
            out = stack.all(dim=0)
        else:  # first: per frame, the first mask with a set pixel
            live = stack.flatten(start_dim=2).any(dim=2)  # [N, B]
            pick = live.to(torch.int64).argmax(dim=0)  # [B]: first live mask (0 when none is live)
            out = stack[pick, torch.arange(stack.shape[1], device=stack.device)]
        return {"decisions": out}


class MaskComposite(Node):
    """Merge N boolean masks [B, H, W, C] into one label map and one level map for display."""

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.MASK, NodeTag.TORCH})

    INPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, -1),
            variadic=True,
            description="Boolean masks [B, H, W, C] of one batch, height and width, one per "
            "inbound connection (fan-in), in the order of `labels` / `levels`; a pixel of a mask "
            "counts as set where any of its channels is.",
        ),
    }
    OUTPUT_SPECS = {
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Label map [B, H, W]: 0 where no mask is set, else the largest `labels` "
            "entry among the masks set there.",
        ),
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Level map [B, H, W, 1]: 0 where no mask is set, else the largest `levels` "
            "entry among the masks set there.",
        ),
    }

    def __init__(
        self, labels: list[int] | None = None, levels: list[float] | None = None, **kwargs: Any
    ) -> None:
        """Create a mask composite.

        Parameters
        ----------
        labels : one positive integer label per inbound mask, in connection order (default
            ``[1, 2]``). Where masks overlap the largest label wins, so give the mask that should
            stay visible on top (e.g. a foreign object on a shell) the largest one.
        levels : one non-negative level per inbound mask, in connection order (default
            ``[0.5, 1.0]``); where masks overlap the largest level wins.
        """
        labels = [1, 2] if labels is None else list(labels)
        levels = [0.5, 1.0] if levels is None else list(levels)
        if not labels or len(labels) != len(levels):
            raise ValueError(
                "MaskComposite: labels and levels need one entry per mask, "
                f"got {labels!r} / {levels!r}."
            )
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in labels):
            raise ValueError(f"MaskComposite: labels must be integers >= 1, got {labels!r}.")
        if any(
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not 0.0 <= float(x) < float("inf")
            for x in levels
        ):
            raise ValueError(f"MaskComposite: levels must be finite numbers >= 0, got {levels!r}.")
        self.labels = [int(x) for x in labels]
        self.levels = [float(x) for x in levels]
        super().__init__(labels=self.labels, levels=self.levels, **kwargs)

    def forward(self, decisions: list[Tensor] | Tensor, **_: Any) -> dict[str, Tensor]:
        """Return the label map and the level map of the inbound masks."""
        masks = list(decisions) if isinstance(decisions, (list, tuple)) else [decisions]
        if len(masks) != len(self.labels):
            raise ValueError(
                f"MaskComposite: {len(masks)} masks connected, "
                f"{len(self.labels)} labels configured."
            )
        bhw = masks[0].shape[:3]
        for i, m in enumerate(masks[1:], start=1):
            if m.shape[:3] != bhw:
                raise ValueError(
                    f"MaskComposite: mask {i} is [B, H, W] = {tuple(m.shape[:3])}, "
                    f"expected {tuple(bhw)}."
                )
        stack = torch.stack([m.any(dim=-1) for m in masks], dim=0)  # [N, B, H, W]
        dev = stack.device
        labels = torch.tensor(self.labels, dtype=torch.int32, device=dev).view(-1, 1, 1, 1)
        levels = torch.tensor(self.levels, dtype=torch.float32, device=dev).view(-1, 1, 1, 1)
        label_map = (stack.to(torch.int32) * labels).amax(dim=0)
        level_map = (stack.to(torch.float32) * levels).amax(dim=0).unsqueeze(-1)
        return {"mask": label_map, "scores": level_map}
