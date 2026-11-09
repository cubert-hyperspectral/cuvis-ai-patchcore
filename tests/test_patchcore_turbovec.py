"""PatchCoreDetector: the turbovec search backend.

Kept out of test_patchcore_detector.py so that module's exact-backend tests (golden parity, port
contract, Phase-1 statistics, serialization, hparam validation) still collect and run in an
environment without the optional ``turbovec`` dependency installed; this module's own
module-scope ``pytest.importorskip("turbovec")`` only skips the tests that actually need it.
"""

from __future__ import annotations

import pytest
import torch

from cuvis_ai_patchcore.node.patchcore import PatchCoreDetector
from tests.test_patchcore_detector import FIT_KW, C, H, W, _fitted, _stream

pytestmark = pytest.mark.unit

turbovec = pytest.importorskip("turbovec")


def test_state_dict_keys_unchanged_with_turbovec_backend():
    """No new persisted keys: a published .pt (exact backend) must load into a turbovec-configured
    node and vice versa, since the index is rebuilt from the coreset, never stored."""
    exact = _fitted()
    tv = _fitted(search_backend="turbovec")
    assert set(exact.state_dict()) == set(tv.state_dict()) == {"mu", "sd", "coreset"}


def test_turbovec_rescore_matches_exact_on_a_small_synthetic_bank():
    """rescore=True must reproduce the exact backend's distance (not merely its argmin): both
    recompute the fp32 difference-form norm over the shortlisted rows, and with shortlist_size
    covering the whole coreset the two backends shortlist a superset/equal set of the true
    nearest rows."""
    kw = {**FIT_KW, "coreset_size": 16, "max_bank_size": 200, "shortlist_size": 16}
    exact = _fitted(**kw, search_backend="exact")
    tv = _fitted(**kw, search_backend="turbovec", rescore=True)
    assert torch.equal(exact.coreset, tv.coreset)
    cube = torch.rand(1, H, W, C, generator=torch.Generator().manual_seed(42)) * 3 + 1
    out_exact = exact(cube=cube)["scores"]
    out_tv = tv(cube=cube)["scores"]
    assert torch.allclose(out_exact, out_tv, atol=1e-4), (out_exact - out_tv).abs().max()


def test_turbovec_no_rescore_uses_hand_checked_formula_on_known_vectors():
    """With rescore=False, the reported distance comes from the norm-reconstruction formula fed
    by the index's own quantised cosine score, not the true Euclidean distance and not a cosine
    recomputed from the coreset rows. This test pins it on a tiny bank where the shortlist always
    covers every row (k == n_coreset_rows), so the only approximation left is the index's own
    cosine versus the exact one, checked here with a tolerance that catches a wrong formula while
    allowing turbovec's own quantisation error."""
    node = PatchCoreDetector(
        input_channels=2,
        coreset_size=4,
        stride=1,
        bank_stride=1,
        pool_size=1,
        max_bank_size=4,
        standardize=False,
        search_backend="turbovec",
        search_bit_width=4,
        shortlist_size=4,
        rescore=False,
    )
    bank = torch.tensor([[3.0, 4.0], [0.0, 5.0], [-3.0, 4.0], [1.0, 0.0]])
    node.statistical_initialization(iter([{"cube": bank.reshape(1, 4, 1, 2)}]))
    query = torch.tensor([[5.0, 0.0]]).reshape(1, 1, 1, 2)
    out = node(cube=query)["scores"][0, 0, 0, 0]
    exact = torch.cdist(torch.tensor([[5.0, 0.0]]), bank).min().item()
    assert out.item() == pytest.approx(exact, abs=0.25)  # quantised cosine, not exact


class _FixedScoreIndex:
    """Wraps a real turbovec index but returns caller-chosen scores for every query, so a test
    can tell whether the node used the index's own score or recomputed one from gathered rows."""

    def __init__(self, real_index, fixed_scores):
        self._real_index = real_index
        self._fixed_scores = fixed_scores

    def __len__(self):
        return len(self._real_index)

    def search(self, queries, k):
        import numpy as np

        _real_scores, ids = self._real_index.search(queries, k)
        n = queries.shape[0]
        scores = np.tile(np.array(self._fixed_scores[:k], dtype=np.float32), (n, 1))
        return scores, ids


def test_turbovec_no_rescore_follows_the_index_own_score_not_the_exact_cosine():
    """rescore=False must use the index's own (possibly approximate) score, not an exact cosine
    recomputed from the gathered coreset rows: a fake score that disagrees with the true cosine
    must show up in the reported distance."""
    node = PatchCoreDetector(
        input_channels=2,
        coreset_size=4,
        stride=1,
        bank_stride=1,
        pool_size=1,
        max_bank_size=4,
        standardize=False,
        search_backend="turbovec",
        search_bit_width=4,
        shortlist_size=4,
        rescore=False,
    )
    bank = torch.tensor([[3.0, 4.0], [0.0, 5.0], [-3.0, 4.0], [1.0, 0.0]])
    node.statistical_initialization(iter([{"cube": bank.reshape(1, 4, 1, 2)}]))
    query = torch.tensor([[5.0, 0.0]]).reshape(1, 1, 1, 2)

    real_index = node._turbovec_state.index
    node._turbovec_state.index = _FixedScoreIndex(real_index, fixed_scores=[1.0, 1.0, 1.0, 1.0])
    out_perfect_cos = node(cube=query)["scores"][0, 0, 0, 0].item()

    node._turbovec_state.index = _FixedScoreIndex(real_index, fixed_scores=[-1.0, -1.0, -1.0, -1.0])
    out_worst_cos = node(cube=query)["scores"][0, 0, 0, 0].item()

    assert out_perfect_cos != pytest.approx(out_worst_cos)
    # cos=1 reconstructs to |q - c|; cos=-1 reconstructs to |q| + |c|, strictly larger here.
    assert out_perfect_cos < out_worst_cos


def test_turbovec_no_rescore_never_gathers_coreset_rows():
    """Poisoning the coreset rows with NaN after the index is built must not reach a rescore=False
    result: that path has no reason to read self.coreset at all."""
    node = PatchCoreDetector(
        input_channels=2,
        coreset_size=4,
        stride=1,
        bank_stride=1,
        pool_size=1,
        max_bank_size=4,
        standardize=False,
        search_backend="turbovec",
        search_bit_width=4,
        shortlist_size=4,
        rescore=False,
    )
    bank = torch.tensor([[3.0, 4.0], [0.0, 5.0], [-3.0, 4.0], [1.0, 0.0]])
    node.statistical_initialization(iter([{"cube": bank.reshape(1, 4, 1, 2)}]))
    query = torch.tensor([[5.0, 0.0]]).reshape(1, 1, 1, 2)
    node(cube=query)  # build the index once over the real bank

    # Poison the buffer's content without bumping the version counter or changing its identity
    # (.data bypasses autograd tracking), so the cache still considers the index fresh: a
    # rescore=False forward must not gather these rows regardless.
    node.coreset.data.copy_(torch.full_like(node.coreset, float("nan")))
    out = node(cube=query)["scores"]
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("rescore", [True, False])
def test_turbovec_rescore_is_chunked_like_the_exact_backend(rescore):
    """The turbovec path must bound its candidate gather by chunk_size, the same way the exact
    path already does, instead of gathering the whole batch's shortlist at once."""
    kw = {**FIT_KW, "coreset_size": 16, "max_bank_size": 200, "shortlist_size": 4}
    big = _fitted(**kw, search_backend="turbovec", rescore=rescore, chunk_size=4096)
    small = _fitted(**kw, search_backend="turbovec", rescore=rescore, chunk_size=7)
    assert torch.equal(big.coreset, small.coreset)
    cube = torch.rand(1, H, W, C, generator=torch.Generator().manual_seed(7)) * 3 + 1

    out_big = big(cube=cube)["scores"]
    max_gather_rows = 0
    original_getitem = torch.Tensor.__getitem__

    def spying_getitem(self, index):
        nonlocal max_gather_rows
        result = original_getitem(self, index)
        if self is small.coreset and isinstance(index, torch.Tensor) and index.ndim == 2:
            max_gather_rows = max(max_gather_rows, index.shape[0])
        return result

    torch.Tensor.__getitem__ = spying_getitem
    try:
        out_small = small(cube=cube)["scores"]
    finally:
        torch.Tensor.__getitem__ = original_getitem

    assert torch.allclose(out_big, out_small, atol=1e-4), (out_big - out_small).abs().max()
    assert max_gather_rows <= small.chunk_size
    if rescore:
        assert max_gather_rows > 0  # the spy actually observed a coreset[ids]-style gather


def test_turbovec_pads_61_channel_bank_to_64():
    node = PatchCoreDetector(
        input_channels=61,
        coreset_size=10,
        stride=1,
        bank_stride=4,
        pool_size=1,
        max_bank_size=100,
        search_backend="turbovec",
    )
    cube = torch.rand(1, 12, 12, 61, generator=torch.Generator().manual_seed(1)) * 2 + 1
    node.statistical_initialization(iter([{"cube": cube}]))
    out = node(cube=cube)["scores"]
    assert torch.isfinite(out).all()  # the node itself keeps C=61; padding is internal to the index


def test_turbovec_index_rebuilt_after_load_state_dict():
    node = _fitted(search_backend="turbovec", coreset_size=16, max_bank_size=200)
    cube = _stream(1, seed=3)[0]["cube"]
    node(cube=cube)  # build the index lazily
    first_index = node._turbovec_state.index

    fresh = PatchCoreDetector(
        **{**FIT_KW, "coreset_size": 16, "max_bank_size": 200, "search_backend": "turbovec"}
    )
    fresh.load_state_dict(node.state_dict())
    fresh(cube=cube)
    assert fresh._turbovec_state.index is not first_index
    assert torch.allclose(fresh(cube=cube)["scores"], node(cube=cube)["scores"], atol=1e-4)


def test_turbovec_index_rebuilt_after_coreset_content_change():
    """An in-place content change (a refit, or a weights load into the same buffer) bumps the
    tensor's version counter; the cache keys on it, not just shape, so a same-shape refit still
    triggers a rebuild instead of silently searching the stale rows."""
    node = _fitted(search_backend="turbovec", coreset_size=16, max_bank_size=200)
    cube = _stream(1, seed=3)[0]["cube"]
    node(cube=cube)
    first_index = node._turbovec_state.index
    node.coreset.copy_(node.coreset + 1.0)  # same shape and dtype, new content
    node(cube=cube)
    assert node._turbovec_state.index is not first_index


def test_turbovec_index_rebuilds_after_assign_true_load_with_matching_version():
    """load_state_dict(..., assign=True) installs a brand new coreset tensor object whose version
    counter can coincidentally match the cached one (both fresh buffers mutated exactly once).
    The signature must still tell the two apart, so the index is rebuilt over the new bank
    instead of silently searching the old one."""
    kw = {**FIT_KW, "coreset_size": 16, "max_bank_size": 200}
    node_a = _fitted(**kw, search_backend="turbovec", seed=0)
    node_b = _fitted(**kw, search_backend="turbovec", seed=99)
    assert not torch.allclose(node_a.coreset, node_b.coreset)

    cube = _stream(1, seed=3)[0]["cube"]
    node_a(cube=cube)  # builds the index over bank A

    node_a.load_state_dict(node_b.state_dict(), assign=True)
    assert node_a.coreset is not node_b.coreset  # a third, freshly assigned tensor
    assert node_a.coreset._version == node_b.coreset._version  # the collision this guards against

    node_a.prepare_inference()
    out_a = node_a(cube=cube)["scores"]

    reference = PatchCoreDetector(**kw, search_backend="turbovec")
    reference.load_state_dict(node_b.state_dict())
    out_ref = reference(cube=cube)["scores"]
    assert torch.allclose(out_a, out_ref, atol=1e-4)


def test_turbovec_detector_builds_and_scores_under_inference_mode():
    """Constructing and loading a turbovec detector inside torch.inference_mode() must not make
    the coreset signature unreadable: coreset._version raises on an inference tensor."""
    kw = {**FIT_KW, "coreset_size": 16, "max_bank_size": 200}
    fitted = _fitted(**kw, search_backend="turbovec")
    cube = _stream(1, seed=3)[0]["cube"]

    with torch.inference_mode():
        node = PatchCoreDetector(**kw, search_backend="turbovec")
        node.load_state_dict(fitted.state_dict())
        node.prepare_inference()
        out = node(cube=cube)["scores"]
    assert torch.isfinite(out).all()


def test_turbovec_backend_builds_eagerly_after_statistical_initialization():
    """statistical_initialization() builds the index right away (Phase 1 is interactive; the
    first inference call should not pay the index-build cost)."""
    node = PatchCoreDetector(
        **{**FIT_KW, "coreset_size": 16, "max_bank_size": 200, "search_backend": "turbovec"}
    )
    assert node._turbovec_state is None
    node.statistical_initialization(iter(_stream(seed=3)))
    assert node._turbovec_state is not None


def test_turbovec_backend_builds_lazily_as_a_fallback_without_prepare_inference():
    """A node whose coreset arrives through load_state_dict alone (no fit, no prepare_inference
    call: the restore path a Node base without the hook would leave it in) still builds the
    index the first time it is needed, as a fallback."""
    fitted = _fitted(search_backend="turbovec", coreset_size=16, max_bank_size=200)
    restored = PatchCoreDetector(
        **{**FIT_KW, "coreset_size": 16, "max_bank_size": 200, "search_backend": "turbovec"}
    )
    restored.load_state_dict(fitted.state_dict())
    assert restored._turbovec_state is None
    cube = _stream(1, seed=1)[0]["cube"]
    out = restored(cube=cube)
    assert torch.isfinite(out["scores"]).all()
    assert restored._turbovec_state is not None


def test_turbovec_prepare_inference_builds_the_index_eagerly():
    node = _fitted(search_backend="turbovec", coreset_size=16, max_bank_size=200)
    node.prepare_inference()
    assert node._turbovec_state is not None
