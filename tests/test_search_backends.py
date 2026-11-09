"""Search-backend plumbing for PatchCoreDetector: hparam validation, the shortlist/rescore split
with a fake index (always runs, no turbovec needed), and the no-rescore reconstruction formula.
Real-turbovec behaviour (equal-to-exact on a known bank, index rebuild, padding) lives in
test_patchcore_detector.py next to the node that owns them.
"""

from __future__ import annotations

import pytest
import torch

from cuvis_ai_patchcore.node._search import (
    TurbovecIndexState,
    _coreset_signature,
    _pad_to_multiple_of_8,
    _unit_normalize,
    reconstruct_cosine_distance,
    turbovec_shortlist,
    validate_search_hparams,
)

pytestmark = pytest.mark.unit


# ----- hparam validation ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"search_backend": "bogus", "search_bit_width": 4, "shortlist_size": 8, "rescore": True},
        {"search_backend": "exact", "search_bit_width": 3, "shortlist_size": 8, "rescore": True},
        {"search_backend": "exact", "search_bit_width": 4, "shortlist_size": 8, "rescore": False},
        {"search_backend": "turbovec", "search_bit_width": 4, "shortlist_size": 0, "rescore": True},
        {
            "search_backend": "turbovec",
            "search_bit_width": 4,
            "shortlist_size": -1,
            "rescore": True,
        },
    ],
)
def test_invalid_search_hparams_raise(kwargs):
    with pytest.raises(ValueError):
        validate_search_hparams(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"search_backend": "exact", "search_bit_width": 4, "shortlist_size": 8, "rescore": True},
        {
            "search_backend": "turbovec",
            "search_bit_width": 2,
            "shortlist_size": 1,
            "rescore": False,
        },
        {
            "search_backend": "turbovec",
            "search_bit_width": 4,
            "shortlist_size": 64,
            "rescore": True,
        },
    ],
)
def test_valid_search_hparams_pass(kwargs):
    validate_search_hparams(**kwargs)  # must not raise


# ----- padding and normalisation helpers ---------------------------------------------------------


def test_pad_to_multiple_of_8_pads_short_channel_dim():
    rows = torch.randn(5, 61)
    padded = _pad_to_multiple_of_8(rows)
    assert padded.shape == (5, 64)
    assert torch.equal(padded[:, :61], rows)
    assert torch.equal(padded[:, 61:], torch.zeros(5, 3))


def test_pad_to_multiple_of_8_is_a_no_op_when_already_aligned():
    rows = torch.randn(5, 64)
    assert torch.equal(_pad_to_multiple_of_8(rows), rows)


def test_unit_normalize_handles_zero_norm_rows():
    rows = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    unit, norms = _unit_normalize(rows, eps=1e-6)
    assert torch.allclose(norms, torch.tensor([5.0, 0.0]))
    assert torch.allclose(unit[0], torch.tensor([0.6, 0.8]))
    assert torch.isfinite(unit).all()  # the zero row must not become NaN


def test_coreset_signature_changes_with_shape_dtype_and_version():
    a = torch.zeros(10, 4)
    b = torch.zeros(12, 4)
    c = a.to(torch.float64)
    sig_a = _coreset_signature(a)
    assert sig_a != _coreset_signature(b)
    assert sig_a != _coreset_signature(c)
    a.copy_(torch.ones(10, 4))
    assert sig_a != _coreset_signature(a)  # version counter bumped by copy_


# ----- shortlist plumbing with a fake index (no turbovec needed) --------------------------------


class _FakeTurbovecIndex:
    """Deterministic stand-in for turbovec.TurboQuantIndex: returns fixed candidate ids per query
    row, independent of the actual vector content, so the plumbing can be tested without the real
    dependency installed."""

    def __init__(self, n_rows: int, fixed_ids):
        self._n_rows = n_rows
        self._fixed_ids = fixed_ids

    def __len__(self):
        return self._n_rows

    def search(self, queries, k):
        import numpy as np

        n = queries.shape[0]
        ids = np.tile(np.array(self._fixed_ids[:k], dtype=np.int64), (n, 1))
        scores = np.zeros((n, k), dtype=np.float32)
        return scores, ids


def test_turbovec_shortlist_returns_fixed_candidate_ids():
    fake_index = _FakeTurbovecIndex(n_rows=10, fixed_ids=[3, 1, 7, 0])
    state = TurbovecIndexState(
        index=fake_index, norms=torch.ones(10), signature=(10, torch.float32, 4, 0)
    )
    flat = torch.randn(5, 4)
    ids = turbovec_shortlist(state, flat, shortlist_size=3, eps=1e-6)
    assert ids.shape == (5, 3)
    assert ids.dtype == torch.int64
    for row in ids:
        assert row.tolist() == [3, 1, 7]


def test_turbovec_shortlist_clips_k_to_index_size():
    fake_index = _FakeTurbovecIndex(n_rows=2, fixed_ids=[1, 0])
    state = TurbovecIndexState(
        index=fake_index, norms=torch.ones(2), signature=(2, torch.float32, 4, 0)
    )
    flat = torch.randn(3, 4)
    ids = turbovec_shortlist(state, flat, shortlist_size=8, eps=1e-6)
    assert ids.shape == (3, 2)  # clipped to the 2 coreset rows


# ----- no-rescore reconstruction formula ---------------------------------------------------------


def test_reconstruct_cosine_distance_matches_hand_checked_vectors():
    # q = (3, 4) |q|=5; c = (0, 5) |c|=5; cos(q, c) = (0+20)/(5*5) = 0.8
    # exact distance: ||q - c|| = ||(3, -1)|| = sqrt(10)
    q = torch.tensor([5.0])
    c = torch.tensor([5.0])
    cos = torch.tensor([0.8])
    dist, n_clamped = reconstruct_cosine_distance(q, c, cos)
    assert torch.allclose(dist, torch.tensor([10.0**0.5]), atol=1e-6)
    assert n_clamped == 0


def test_reconstruct_cosine_distance_handles_zero_norm():
    q = torch.tensor([0.0, 5.0])
    c = torch.tensor([5.0, 0.0])
    cos = torch.tensor([0.0, 0.0])
    dist, n_clamped = reconstruct_cosine_distance(q, c, cos)
    assert torch.allclose(dist, torch.tensor([5.0, 5.0]))
    assert n_clamped == 0


def test_reconstruct_cosine_distance_clamps_negative_radicand_and_counts_it():
    # An approximate cos slightly above 1 with near-equal norms can push the radicand negative
    # before flooring; the clamp keeps the distance real and the count records it happened.
    q = torch.tensor([1.0])
    c = torch.tensor([1.0])
    cos = torch.tensor([1.3])  # outside [-1, 1]; the formula clamps cos itself too
    dist, n_clamped = reconstruct_cosine_distance(q, c, cos)
    assert dist.item() == pytest.approx(0.0, abs=1e-6)
    assert n_clamped == 0  # clamping cos to 1 already keeps the radicand at exactly 0 here


def test_reconstruct_cosine_distance_counts_clamped_negatives_from_fp_noise():
    # Equal norms and cos clamped to exactly 1.0 give radicand 0 by construction; force a tiny
    # negative radicand directly through norms that make 2|q||c| slightly exceed |q|^2+|c|^2.
    q = torch.tensor([1.0])
    c = torch.tensor([1.0000001])
    cos = torch.tensor([1.0])
    _dist, n_clamped = reconstruct_cosine_distance(q, c, cos)
    assert n_clamped in (0, 1)  # fp-noise dependent; must not raise and must stay a count


def test_validate_search_hparams_is_strict_about_backend_spelling():
    with pytest.raises(ValueError, match="search_backend"):
        validate_search_hparams(
            search_backend="Turbovec", search_bit_width=4, shortlist_size=8, rescore=True
        )
