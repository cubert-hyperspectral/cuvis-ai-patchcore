"""k-center greedy (farthest-first) coreset selection.

Mirrors the selection order of anomalib's ``KCenterGreedy`` (Sener & Savarese, 2018, as used by
PatchCore): a random seed point initialises the distance field, then each step picks the point
with the largest distance to its nearest already-selected centre. Differences from anomalib:

* no sparse random projection — the Johnson–Lindenstrauss trick only pays off for very
  high-dimensional CNN patch features; hyperspectral spectra (tens to a few hundred bands) are
  scored directly, so distances are exact;
* seeded through an explicit ``torch.Generator`` so a Phase-1 fit is reproducible;
* distances use the expanded ``|a|² - 2a·b + |b|²`` form (one mat-vec per step, no ``(N, F)``
  temporary).
"""

from __future__ import annotations

import torch
from torch import Tensor


@torch.no_grad()
def k_center_greedy(
    features: Tensor,
    n_select: int,
    *,
    generator: torch.Generator | None = None,
    start_index: int | None = None,
) -> Tensor:
    """Return ``LongTensor`` indices of a farthest-first coreset of ``features`` (``(N, F)``).

    ``n_select`` is clipped to ``N``. As in anomalib the random seed point itself is only
    selected when every point is requested (``n_select >= N``), in which case the result is a
    farthest-first ordering of all ``N`` rows.
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
