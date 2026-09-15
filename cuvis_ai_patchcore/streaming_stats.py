"""Streaming per-feature mean / variance (Chan et al. parallel Welford merge, float64 state).

cuvis-ai-core ships no Welford utility (it lives in the high-level ``cuvis_ai`` package, which
plugins must not depend on), so this small accumulator covers the Phase-1 z-score statistics.
"""

from __future__ import annotations

import torch
from torch import Tensor


class StreamingMeanVar:
    """Accumulate mean and (unbiased) variance of ``(N, F)`` batches in float64."""

    def __init__(self, n_features: int) -> None:
        if n_features <= 0:
            raise ValueError(f"n_features must be positive, got {n_features}")
        self.n_features = int(n_features)
        self.count = 0
        self._mean = torch.zeros(self.n_features, dtype=torch.float64)
        self._m2 = torch.zeros(self.n_features, dtype=torch.float64)

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        """Merge a batch of shape ``(N, n_features)``."""
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"expected (N, {self.n_features}), got {tuple(x.shape)}")
        m = int(x.shape[0])
        if m == 0:
            return
        x64 = x.detach().to(torch.float64)
        if self._mean.device != x64.device:
            self._mean = self._mean.to(x64.device)
            self._m2 = self._m2.to(x64.device)
        batch_mean = x64.mean(dim=0)
        batch_m2 = ((x64 - batch_mean) ** 2).sum(dim=0)
        n_new = self.count + m
        delta = batch_mean - self._mean
        self._mean = self._mean + delta * (m / n_new)
        self._m2 = self._m2 + batch_m2 + delta**2 * (self.count * m / n_new)
        self.count = n_new

    @property
    def mean(self) -> Tensor:
        """Running mean, float32, shape ``(n_features,)``."""
        return self._mean.to(torch.float32)

    @property
    def var(self) -> Tensor:
        """Unbiased running variance, float32, shape ``(n_features,)`` (zeros until 2 samples)."""
        if self.count < 2:
            return torch.zeros_like(self._mean, dtype=torch.float32)
        return (self._m2 / (self.count - 1)).to(torch.float32)
