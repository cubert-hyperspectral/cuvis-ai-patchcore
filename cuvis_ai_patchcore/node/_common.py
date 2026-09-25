"""Helpers shared by the plugin's nodes (private: not part of the plugin's API).

- :func:`check_topk_frac` and :func:`topk_mean` define the image-level score every node reports,
  so a detector's ``anomaly_score`` and a gate's ``frame_score`` on the same map agree exactly.
- :func:`random_cap` is the seeded row cap both Phase-1 fits use to bound memory.
- :func:`require_fitted` is the Phase-1 guard of the fitted nodes' ``forward``.
"""

from __future__ import annotations

import torch
from cuvis_ai_core.node.node import Node
from torch import Tensor


def check_topk_frac(topk_frac: float) -> float:
    """Return ``topk_frac`` as a float; raise ``ValueError`` unless it is in ``(0, 1]``."""
    value = float(topk_frac)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"topk_frac must be in (0, 1], got {topk_frac}")
    return value


def topk_mean(values: Tensor, topk_frac: float) -> Tensor:
    """Per-frame mean of the top ``topk_frac`` values of a ``[B, ...]`` tensor -> ``[B]`` float32.

    ``k = max(1, floor(topk_frac * n))`` over the ``n`` values of one frame, so a small map still
    averages at least its maximum.
    """
    flat = values.reshape(values.shape[0], -1).float()
    k = max(1, int(topk_frac * flat.shape[1]))
    return torch.topk(flat, k, dim=1).values.mean(dim=1)


def random_cap(rows: Tensor, max_rows: int, generator: torch.Generator) -> Tensor:
    """Seeded random subset of at most ``max_rows`` rows (``rows`` itself when already within).

    The generator is drawn from only when rows are dropped.
    """
    if rows.shape[0] <= max_rows:
        return rows
    keep = torch.randperm(rows.shape[0], generator=generator)[:max_rows]
    return rows[keep.to(rows.device)]


def require_fitted(node: Node) -> None:
    """Raise ``RuntimeError`` when a fitted node runs before Phase 1 or a weights load."""
    if not node._statistically_initialized:
        raise RuntimeError(
            f"{type(node).__name__} requires statistical_initialization() (Phase 1) or "
            "loaded weights before forward()."
        )
