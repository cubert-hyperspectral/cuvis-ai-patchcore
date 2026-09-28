"""k-center greedy (farthest-first) coreset selection.

Mirrors the selection order of anomalib's ``KCenterGreedy`` (Sener & Savarese, 2018, as used by
PatchCore): a random seed point initialises the distance field, then each step picks the point
with the largest distance to its nearest already-selected centre. Differences from anomalib:

* exact distances by default — anomalib's sparse random projection (Johnson–Lindenstrauss) only
  pays off for very high-dimensional CNN patch features; hyperspectral spectra (tens to a few
  hundred bands) are scored directly. The projection is available as an option
  (``projection_eps``) to reproduce anomalib's selection recipe;
* seeded through an explicit ``torch.Generator`` so a Phase-1 fit is reproducible;
* distances use the expanded ``|a|² - 2a·b + |b|²`` form (one mat-vec per step, no ``(N, F)``
  temporary).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


@torch.no_grad()
def k_center_greedy(
    features: Tensor,
    n_select: int,
    *,
    generator: torch.Generator | None = None,
    start_index: int | None = None,
    projection_eps: float | None = None,
) -> Tensor:
    """Return ``LongTensor`` indices of a farthest-first coreset of ``features`` (``(N, F)``).

    ``n_select`` is clipped to ``N``. As in anomalib the random seed point itself is only
    selected when every point is requested (``n_select >= N``), in which case the result is a
    farthest-first ordering of all ``N`` rows. With ``projection_eps`` set, the greedy selection
    runs on anomalib's sparse random projection of the features (see
    :func:`sparse_random_projection`); the returned indices still address the original rows.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be (N, F), got {tuple(features.shape)}")
    n = int(features.shape[0])
    if n == 0:
        raise ValueError("features is empty")
    if n_select <= 0:
        raise ValueError(f"n_select must be positive, got {n_select}")
    n_select = min(int(n_select), n)

    f = features.detach().to(torch.float32)
    if projection_eps is not None:
        f = sparse_random_projection(f, projection_eps, generator=generator)
    if start_index is None:
        start_index = int(torch.randint(high=n, size=(1,), generator=generator).item())
    if not 0 <= start_index < n:
        raise ValueError(f"start_index {start_index} out of range for N={n}")

    sq = (f * f).sum(dim=1)
    min_d = torch.full((n,), float("inf"), dtype=torch.float32, device=f.device)
    picks: list[int] = []
    centre = start_index
    n_greedy = n_select if n_select < n else n - 1
    for _ in range(n_greedy):
        d = (sq - 2.0 * (f @ f[centre]) + sq[centre]).clamp_min_(0.0)
        min_d = torch.minimum(min_d, d)
        min_d[centre] = 0.0
        centre = int(torch.argmax(min_d).item())
        picks.append(centre)
    if n_select >= n:
        picks.append(start_index)
    return torch.tensor(picks, dtype=torch.long)


# -------------------------------------------------------------- anomalib-style option
def johnson_lindenstrauss_min_dim(n_samples: int, eps: float) -> int:
    """anomalib's Johnson-Lindenstrauss bound, truncated to int.

    ``4 ln(n) / (eps**2/2 - eps**3/3)``. For a low-dimensional input (61-band spectra, hundreds
    of thousands of samples) this is LARGER than the feature dimension: anomalib then projects
    *up*, which randomises the selection without reducing anything.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2")
    if not 0.0 < eps < 1.0:
        raise ValueError(f"eps must be in (0, 1), got {eps}")
    return int(4.0 * math.log(n_samples) / (eps**2 / 2.0 - eps**3 / 3.0))


@torch.no_grad()
def sparse_random_projection(
    features: Tensor, eps: float = 0.9, *, generator: torch.Generator | None = None
) -> Tensor:
    """Very sparse random projection (Li, Hastie & Church 2006), anomalib's step before k-center.

    ``density = 1/sqrt(F)``; each of the ``k`` output rows has ``Binomial(F, density)`` non-zeros
    at uniformly drawn columns with values ``+-1``, scaled by ``sqrt(1/density)/sqrt(k)``; ``k`` is
    the JL dimension for ``eps``. Returns ``features @ R.T`` of shape ``(N, k)``. Draws come from
    ``generator`` (anomalib uses the unseeded global torch + numpy RNGs, so its selections are not
    reproducible run to run; a seeded generator makes ours reproducible while keeping the recipe).
    """
    if features.ndim != 2:
        raise ValueError(f"features must be (N, F), got {tuple(features.shape)}")
    n, f = int(features.shape[0]), int(features.shape[1])
    k = johnson_lindenstrauss_min_dim(n, eps)
    density = 1.0 / math.sqrt(f)
    if density >= 1.0:  # F == 1: dense +-1 matrix
        r = (torch.randint(0, 2, (k, f), generator=generator).float() * 2.0 - 1.0) / math.sqrt(k)
    else:
        r = torch.zeros((k, f), dtype=torch.float32)
        nnz = torch.binomial(
            torch.full((k,), float(f)), torch.full((k,), density), generator=generator
        ).long()
        for i in range(k):
            m = int(nnz[i])
            if m == 0:
                continue
            cols = torch.randperm(f, generator=generator)[:m]
            r[i, cols] = torch.randint(0, 2, (m,), generator=generator).float() * 2.0 - 1.0
        r *= math.sqrt(1.0 / density) / math.sqrt(k)
    return features.detach().to(torch.float32) @ r.to(features.device).T
