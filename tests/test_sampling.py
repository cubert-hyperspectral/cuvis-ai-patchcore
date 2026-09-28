"""k-center greedy: parity with a naive farthest-first reference, distinctness, edge cases."""

from __future__ import annotations

import pytest
import torch

from cuvis_ai_patchcore.sampling import (
    johnson_lindenstrauss_min_dim,
    k_center_greedy,
    sparse_random_projection,
)

pytestmark = pytest.mark.unit


def _naive_farthest_first(features: torch.Tensor, n_select: int, start: int) -> list[int]:
    """O(N * k * N) reference: explicit min over all selected centres at every step.

    Mirrors anomalib's KCenterGreedy semantics: the start point seeds the distance field but is
    itself never appended; each step appends the argmax of the min-distance-to-centres field.
    """
    centres = [start]
    picks: list[int] = []
    for _ in range(n_select):
        d = torch.stack([(features - features[c]).norm(dim=1) for c in centres]).min(dim=0).values
        for c in centres:
            d[c] = 0.0
        nxt = int(torch.argmax(d).item())
        picks.append(nxt)
        centres.append(nxt)
    return picks


def test_matches_naive_reference():
    torch.manual_seed(0)
    feats = torch.randn(200, 7)
    for start in (0, 17, 199):
        ref = _naive_farthest_first(feats, 25, start)
        out = k_center_greedy(feats, 25, start_index=start)
        assert out.dtype == torch.long
        assert out.tolist() == ref


def test_indices_distinct_and_in_range():
    torch.manual_seed(1)
    feats = torch.rand(500, 3)
    idx = k_center_greedy(feats, 120, generator=torch.Generator().manual_seed(3))
    assert idx.shape == (120,)
    assert len(set(idx.tolist())) == 120
    assert int(idx.min()) >= 0 and int(idx.max()) < 500


def test_seeded_generator_is_reproducible():
    feats = torch.rand(300, 4, generator=torch.Generator().manual_seed(5))
    a = k_center_greedy(feats, 40, generator=torch.Generator().manual_seed(11))
    b = k_center_greedy(feats, 40, generator=torch.Generator().manual_seed(11))
    c = k_center_greedy(feats, 40, generator=torch.Generator().manual_seed(12))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_select_everything_returns_permutation():
    feats = torch.rand(30, 2, generator=torch.Generator().manual_seed(0))
    idx = k_center_greedy(feats, 30, start_index=4)
    assert sorted(idx.tolist()) == list(range(30))
    assert idx[-1].item() == 4  # the seed point is appended last
    more = k_center_greedy(feats, 1000, start_index=4)  # n_select clipped to N
    assert torch.equal(more, idx)


def test_first_pick_is_farthest_from_start():
    feats = torch.tensor([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [3.0, 0.0]])
    idx = k_center_greedy(feats, 2, start_index=0)
    assert idx.tolist() == [2, 3]  # farthest from 0 is 10.0, then 3.0 (midway between 0 and 10)


@pytest.mark.parametrize("bad", [0, -3])
def test_rejects_bad_n_select(bad):
    with pytest.raises(ValueError):
        k_center_greedy(torch.rand(5, 2), bad)


def test_rejects_non_2d_and_empty():
    with pytest.raises(ValueError):
        k_center_greedy(torch.rand(5, 2, 2), 2)
    with pytest.raises(ValueError):
        k_center_greedy(torch.empty(0, 2), 2)
    with pytest.raises(ValueError):
        k_center_greedy(torch.rand(5, 2), 2, start_index=9)


# ----------------------------------------------------------------- anomalib-style projection option
@pytest.mark.parametrize(
    ("n", "eps", "expected"),
    [(300_000, 0.9, 311), (28_800, 0.9, 253), (1_000, 0.5, 331), (50, 0.9, 96)],
)
def test_jl_dim_matches_anomalib_formula(n, eps, expected):
    """Values cross-checked against anomalib 2.1 ``SparseRandomProjection``."""
    assert johnson_lindenstrauss_min_dim(n, eps) == expected
    anomalib = pytest.importorskip("anomalib.models.components.dimensionality_reduction")
    assert int(anomalib.SparseRandomProjection._johnson_lindenstrauss_min_dim(n, eps)) == expected


@pytest.mark.parametrize("bad", [(1, 0.9), (100, 0.0), (100, 1.0), (100, -0.1)])
def test_jl_dim_rejects_bad_arguments(bad):
    with pytest.raises(ValueError):
        johnson_lindenstrauss_min_dim(*bad)


def test_projection_shape_and_determinism():
    x = torch.randn(500, 61)
    p1 = sparse_random_projection(x, 0.9, generator=torch.Generator().manual_seed(3))
    p2 = sparse_random_projection(x, 0.9, generator=torch.Generator().manual_seed(3))
    p3 = sparse_random_projection(x, 0.9, generator=torch.Generator().manual_seed(4))
    assert p1.shape == (500, johnson_lindenstrauss_min_dim(500, 0.9))
    assert torch.equal(p1, p2)
    assert not torch.equal(p1, p3)


def test_projection_matrix_follows_anomalib_recipe():
    """Non-zeros are +-sqrt(1/density)/sqrt(k) and the fill matches density = 1/sqrt(F)."""
    f = 64
    r_t = sparse_random_projection(torch.eye(f), 0.9, generator=torch.Generator().manual_seed(0))
    k = johnson_lindenstrauss_min_dim(f, 0.9)
    assert r_t.shape == (f, k)  # projecting the identity exposes R.T
    density = 1.0 / f**0.5
    scale = (1.0 / density) ** 0.5 / k**0.5
    nz = r_t[r_t != 0]
    assert torch.allclose(nz.abs(), torch.full_like(nz, scale), atol=1e-6)
    assert abs(float((r_t != 0).float().mean()) - density) < 0.03


def test_projection_preserves_pairwise_distances_approximately():
    """JL guarantee: every squared distance is distorted by at most (1 +- eps), on average ~1."""
    g = torch.Generator().manual_seed(11)
    x = torch.randn(300, 768, generator=g) * torch.logspace(-1, 1, 300)[:, None]  # scale spread
    p = sparse_random_projection(x, 0.9, generator=torch.Generator().manual_seed(0))
    d0 = torch.cdist(x[:50], x[50:100]).flatten() ** 2
    d1 = torch.cdist(p[:50], p[50:100]).flatten() ** 2
    ratio = d1 / d0
    assert float(ratio.min()) > 0.1 and float(ratio.max()) < 1.9  # the (1 +- eps) bound
    assert 0.8 < float(ratio.mean()) < 1.2
    assert float(torch.corrcoef(torch.stack([d0, d1]))[0, 1]) > 0.95


def test_k_center_with_projection_is_valid_and_reproducible():
    x = torch.randn(1_000, 61)
    a = k_center_greedy(x, 50, generator=torch.Generator().manual_seed(1), projection_eps=0.9)
    b = k_center_greedy(x, 50, generator=torch.Generator().manual_seed(1), projection_eps=0.9)
    c = k_center_greedy(x, 50, generator=torch.Generator().manual_seed(1))
    assert torch.equal(a, b)
    assert a.numel() == 50 and a.unique().numel() == 50 and int(a.max()) < 1_000
    assert not torch.equal(a, c)  # the projection changes the selection (that is its purpose)


def test_k_center_with_projection_still_spreads_out():
    """Projected selection is still farthest-first: separated clusters are all covered early."""
    g = torch.Generator().manual_seed(5)
    centres = torch.tensor(
        [
            [0, 0, 0, 0],
            [40, 0, 0, 0],
            [0, 40, 0, 0],
            [0, 0, 40, 0],
            [0, 0, 0, 40],
            [40, 40, 0, 0],
            [0, 0, 40, 40],
            [40, 0, 40, 0],
        ],
        dtype=torch.float32,
    )
    labels = torch.arange(8).repeat_interleave(250)
    x = centres[labels] + torch.randn(2_000, 4, generator=g)
    start = 3  # cluster 0 seeds the distance field, so the greedy never has to pick from it
    others = set(range(1, 8))
    for eps in (None, 0.9):
        idx = k_center_greedy(
            x, 7, generator=torch.Generator().manual_seed(0), start_index=start, projection_eps=eps
        )
        assert set(labels[idx].tolist()) == others  # one pick per remaining cluster, in 7 picks
