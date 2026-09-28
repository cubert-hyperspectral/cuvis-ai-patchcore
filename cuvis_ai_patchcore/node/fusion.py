"""Score-map fusion — one score map from several, by a fixed elementwise rule.

Averaging the calibrated maps of complementary detectors (e.g. a PatchCore bank on raw spectra and
a PatchCore bank on ViT features) cancels each detector's private noise while keeping the signal
both agree on; ``min`` is the hard AND, ``max`` the OR, ``wmean`` a weighted average. ``first`` is a
priority rule for gated maps: per frame it takes the first inbound map that is not all zero, e.g.
one detector's display map whenever its gate opens and a second detector's only on frames the first
misses. The node is stateless, torch-native and differentiable. Feed it maps on a common scale (a
fitted normalizer per detector), otherwise the detector with the widest range dominates.
"""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

_MODES = ("mean", "min", "max", "wmean", "first")


class ScoreMapFusion(Node):
    """Fuse N score maps [B, H, W, 1] into one by ``mode`` (mean | min | max | wmean | first)."""

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
        self, mode: str = "mean", weights: list[float] | None = None, **kwargs: Any
    ) -> None:
        """Create a fusion node.

        Parameters
        ----------
        mode : ``"mean"`` (arithmetic average), ``"min"`` (AND), ``"max"`` (OR), ``"wmean"``
            (weighted average with ``weights``, normalised to sum to one) or ``"first"`` (per frame,
            the first inbound map in connection order that is not all zero; zeros if all are).
        weights : one non-negative weight per inbound map, required for ``"wmean"`` and rejected
            for the other modes; the count is checked against the inbound maps at run time.
        """
        if mode not in _MODES:
            raise ValueError(f"ScoreMapFusion: mode must be one of {_MODES}, got {mode!r}.")
        if mode == "wmean":
            if not weights:
                raise ValueError("ScoreMapFusion: mode 'wmean' requires weights.")
            if any(float(x) < 0.0 for x in weights) or float(sum(weights)) <= 0.0:
                raise ValueError(
                    "ScoreMapFusion: weights must be non-negative with a positive sum."
                )
        elif weights is not None:
            raise ValueError(
                f"ScoreMapFusion: weights are only used with mode 'wmean', not {mode!r}."
            )
        self.mode = mode
        self.weights = [float(x) for x in weights] if weights is not None else None
        super().__init__(mode=self.mode, weights=self.weights, **kwargs)

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
        else:  # wmean
            if len(self.weights) != len(maps):
                raise ValueError(
                    f"ScoreMapFusion: {len(self.weights)} weights for {len(maps)} score maps."
                )
            w = torch.tensor(self.weights, dtype=stack.dtype, device=stack.device)
            w = (w / w.sum()).view(-1, 1, 1, 1, 1)
            out = (stack * w).sum(dim=0)
        return {"scores": out}
