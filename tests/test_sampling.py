"""k-center greedy: parity with a naive farthest-first reference, distinctness, edge cases."""

from __future__ import annotations

import pytest
import torch

from cuvis_ai_patchcore.sampling import k_center_greedy

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
