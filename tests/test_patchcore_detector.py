"""PatchCoreDetector: golden parity with the reference scoring formula, port contract, Phase-1
statistics, coreset membership, state-dict round-trip, and hparam validation.

The reference implementation is a verbatim port of the scoring used to validate HSI-PatchCore
before the node existed (per-band z-score -> 3x3 avg-pool -> stride grid -> k=1 ``torch.cdist``
-> bilinear upsample); the node must reproduce it to float32 precision.
"""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F
from cuvis_ai_schemas.enums import ExecutionStage

from cuvis_ai_patchcore.node.patchcore import PatchCoreDetector

pytestmark = pytest.mark.unit

C, H, W = 5, 23, 17  # H, W deliberately not multiples of the strides
FIT_KW = {
    "input_channels": C,
    "coreset_size": 40,
    "stride": 3,
    "bank_stride": 2,
    "max_bank_size": 300,
}


def _stream(n_cubes: int = 4, seed: int = 0) -> list[dict[str, torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    scale = torch.tensor([1.0, 4.0, 0.5, 20.0, 2.0])
    offset = torch.tensor([0.0, 10.0, -3.0, 100.0, 1.0])
    return [{"cube": torch.rand(1, H, W, C, generator=g) * scale + offset} for _ in range(n_cubes)]


def _fitted(**overrides) -> PatchCoreDetector:
    kw = {**FIT_KW, **overrides}
    node = PatchCoreDetector(**kw)
    node.statistical_initialization(iter(_stream()))
    return node


def _reference_scores(
    cube_hwc: torch.Tensor,
    mu: torch.Tensor,
    sd: torch.Tensor,
    coreset: torch.Tensor,
    stride: int,
    pool: int,
) -> torch.Tensor:
    """Verbatim reference: feats() + score_frame() of the validation script."""
    x = (cube_hwc - mu) / sd
    x = x.permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
    if pool > 1:
        x = F.avg_pool2d(x, pool, stride=1, padding=pool // 2)
    g = x[:, :, ::stride, ::stride]
    gh, gw = g.shape[-2:]
    f = g.squeeze(0).reshape(x.shape[1], -1).t().contiguous()
    d = torch.empty(f.shape[0])
    for i in range(0, f.shape[0], 4096):
        chunk = f[i : i + 4096]
        d[i : i + chunk.shape[0]] = torch.cdist(chunk, coreset).min(1).values
    a = d.reshape(1, 1, gh, gw)
    h, w = cube_hwc.shape[:2]
    return F.interpolate(a, size=(h, w), mode="bilinear", align_corners=False)[0, 0]


# ----- 1. golden reference --------------------------------------------------------------------


def test_golden_parity_with_reference_scoring():
    node = _fitted()
    cube = torch.rand(2, H, W, C, generator=torch.Generator().manual_seed(7)) * 3.0 + 1.0
    out = node(cube=cube)["scores"]
    for i in range(cube.shape[0]):
        ref = _reference_scores(
            cube[i], node.mu, node.sd, node.coreset, node.stride, node.pool_size
        )
        assert torch.allclose(out[i, ..., 0], ref, atol=1e-5), (out[i, ..., 0] - ref).abs().max()


def _reference_distance_f64(flat: torch.Tensor, coreset: torch.Tensor) -> torch.Tensor:
    """Exact nearest-coreset distance in float64, no matmul expansion."""
    return torch.cdist(flat.double(), coreset.double()).min(dim=1).values


def _old_matmul_cdist_distance(
    flat: torch.Tensor, coreset: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    """The distance formula ``_nearest_distance`` used before the numerics fix (pre-ALL-6819):
    plain ``torch.cdist`` (matmul expansion) over the query chunk and the whole coreset, mirrored
    here so the characterization test does not depend on the current implementation."""
    out = torch.empty(flat.shape[0], dtype=torch.float32)
    for i in range(0, flat.shape[0], chunk_size):
        chunk = flat[i : i + chunk_size]
        out[i : i + chunk.shape[0]] = torch.cdist(chunk, coreset).min(dim=1).values
    return out


def test_distance_matches_float64_reference_closely():
    """Pins ``_nearest_distance`` against an exact float64 reference and against the pre-fix
    matmul-cdist formula, on seeded data (the golden-parity fixture, seed 7).

    Measured on this fixture: the pre-fix matmul-expansion distance (``_old_matmul_cdist_distance``)
    is within ``8.94e-06`` of the float64 reference over both frames (96 queries total against a
    40-row coreset). The current ``_nearest_distance`` must be at least as close to the float64
    reference (asserted at ``atol=2e-5``, a safety margin above the measured pre-fix error) and
    within ``atol=1e-4`` of the pre-fix values, since the fix changes only how the exact distance
    is derived, not the feature pipeline that produces the query rows.
    """
    node = _fitted()
    cube = torch.rand(2, H, W, C, generator=torch.Generator().manual_seed(7)) * 3.0 + 1.0
    z = (cube - node.mu) / node.sd
    pooled = node._pooled(z.permute(0, 3, 1, 2), node.stride)
    flat = pooled.permute(0, 2, 3, 1).reshape(-1, C)

    got = node._nearest_distance(flat)
    ref = _reference_distance_f64(flat, node.coreset)
    old = _old_matmul_cdist_distance(flat, node.coreset, node.chunk_size)

    assert (got.double() - ref).abs().max() < 2e-5
    assert torch.allclose(got, old, atol=1e-4), (got - old).abs().max()


def test_pool_size_one_is_plain_per_pixel_nearest_distance():
    node = _fitted(pool_size=1, stride=1)
    cube = torch.rand(1, 9, 8, C, generator=torch.Generator().manual_seed(3))
    out = node(cube=cube)["scores"][0, ..., 0]
    z = ((cube[0] - node.mu) / node.sd).reshape(-1, C)
    expected = torch.cdist(z, node.coreset).min(1).values.reshape(9, 8)
    assert torch.allclose(out, expected, atol=1e-5)


def test_anomaly_score_is_topk_mean_of_pixel_scores():
    node = _fitted(topk_frac=0.01)
    cube = torch.rand(2, H, W, C, generator=torch.Generator().manual_seed(9))
    out = node(cube=cube)
    k = max(1, int(0.01 * H * W))
    expected = torch.topk(out["scores"].reshape(2, -1), k, dim=1).values.mean(dim=1)
    assert torch.allclose(out["anomaly_score"], expected)


# ----- 2. port contract -----------------------------------------------------------------------


def test_port_contract():
    node = _fitted()
    out = node(cube=torch.rand(3, H, W, C))
    assert set(out) == set(node.OUTPUT_SPECS)
    assert out["scores"].shape == (3, H, W, 1)
    assert out["scores"].dtype == node.OUTPUT_SPECS["scores"].dtype
    assert out["anomaly_score"].shape == (3,)
    assert out["anomaly_score"].dtype == node.OUTPUT_SPECS["anomaly_score"].dtype
    assert torch.isfinite(out["scores"]).all()


def test_node_metadata_and_stages():
    node = PatchCoreDetector(**FIT_KW)
    assert node.requires_initial_fit is True
    assert node.TRAINABLE_BUFFERS == ()
    assert ExecutionStage.ALWAYS in node.execution_stages
    assert "cube" in node.INPUT_SPECS


def test_forward_before_fit_raises():
    node = PatchCoreDetector(**FIT_KW)
    with pytest.raises(RuntimeError, match="statistical_initialization"):
        node(cube=torch.rand(1, H, W, C))


def test_batch_matches_per_sample():
    node = _fitted()
    cube = torch.rand(3, H, W, C, generator=torch.Generator().manual_seed(11))
    batched = node(cube=cube)
    for i in range(3):
        single = node(cube=cube[i : i + 1])
        assert torch.allclose(batched["scores"][i], single["scores"][0], atol=1e-6)
        assert torch.allclose(batched["anomaly_score"][i], single["anomaly_score"][0], atol=1e-6)


def test_chunking_does_not_change_scores():
    a = _fitted(chunk_size=4096)
    b = _fitted(chunk_size=7)
    cube = torch.rand(1, H, W, C, generator=torch.Generator().manual_seed(2))
    assert torch.equal(a.coreset, b.coreset)
    assert torch.allclose(a(cube=cube)["scores"], b(cube=cube)["scores"], atol=1e-6)


# ----- 3. Phase 1 -----------------------------------------------------------------------------


def test_fit_statistics_match_stream():
    node = _fitted()
    pixels = torch.cat([b["cube"].reshape(-1, C) for b in _stream()])
    assert torch.allclose(node.mu, pixels.mean(0), atol=1e-4)
    assert torch.allclose(node.sd, pixels.std(0, unbiased=True).clamp_min(node.eps), rtol=1e-4)


def test_coreset_rows_are_distinct_bank_members():
    node = _fitted()
    bank = []
    for b in _stream():
        x = ((b["cube"] - node.mu) / node.sd).permute(0, 3, 1, 2)
        x = F.avg_pool2d(x, node.pool_size, stride=1, padding=node.pool_size // 2)
        x = x[:, :, :: node.bank_stride, :: node.bank_stride]
        bank.append(x.permute(0, 2, 3, 1).reshape(-1, C))
    bank = torch.cat(bank)
    assert bank.shape[0] > node.coreset_size  # no cycling in this configuration
    nearest = torch.cdist(node.coreset, bank).min(1).values
    # Phase 1 standardises window SUMS after the pass; the test standardises first and then
    # averages. Same math, different float32 accumulation order -> ~1e-4 z-units, not zero.
    assert nearest.max() < 5e-3, nearest.max()
    assert torch.unique(node.coreset, dim=0).shape[0] == node.coreset_size


def test_fit_is_reproducible_and_seed_dependent():
    a, b = _fitted(seed=1), _fitted(seed=1)
    c = _fitted(seed=2)
    assert torch.equal(a.coreset, b.coreset)
    assert not torch.equal(a.coreset, c.coreset)


def test_bank_cap_is_applied():
    node = _fitted(max_bank_size=60, coreset_size=50)
    assert node.coreset.shape == (50, C)
    assert torch.unique(node.coreset, dim=0).shape[0] == 50


def test_small_bank_is_cycled_to_fill_the_buffer():
    node = PatchCoreDetector(
        input_channels=C, coreset_size=10, stride=2, bank_stride=3, max_bank_size=10
    )
    cube = torch.rand(1, 6, 6, C, generator=torch.Generator().manual_seed(0))  # 4 pooled samples
    node.statistical_initialization(iter([{"cube": cube}]))
    assert node.coreset.shape == (10, C)
    assert torch.unique(node.coreset, dim=0).shape[0] == 4
    out = node(cube=cube)["scores"]
    assert torch.isfinite(out).all()


def test_fit_rejects_channel_mismatch_and_empty_stream():
    node = PatchCoreDetector(**FIT_KW)
    with pytest.raises(ValueError, match="channels"):
        node.statistical_initialization(iter([{"cube": torch.rand(1, 4, 4, C + 1)}]))
    with pytest.raises(RuntimeError, match="insufficient"):
        node.statistical_initialization(iter([]))
    assert node._statistically_initialized is False


# ----- 4. serialization -----------------------------------------------------------------------


def test_state_dict_round_trip_marks_node_fitted():
    fitted = _fitted()
    fresh = PatchCoreDetector(**FIT_KW)
    assert set(fitted.state_dict()) == {"mu", "sd", "coreset"}
    fresh.load_state_dict(fitted.state_dict())
    assert fresh._statistically_initialized is True
    cube = torch.rand(1, H, W, C, generator=torch.Generator().manual_seed(5))
    assert torch.equal(fresh(cube=cube)["scores"], fitted(cube=cube)["scores"])


def test_hparams_are_json_serializable_and_complete():
    node = PatchCoreDetector(**FIT_KW, autocast_dtype="float16", name="pc")
    hp = node.hparams
    for key in (
        "input_channels",
        "coreset_size",
        "stride",
        "bank_stride",
        "pool_size",
        "max_bank_size",
        "topk_frac",
        "chunk_size",
        "autocast_dtype",
        "tf32",
        "standardize",
        "seed",
        "eps",
        "search_backend",
        "search_bit_width",
        "shortlist_size",
        "rescore",
    ):
        assert key in hp, key
    json.dumps(hp)  # must not raise
    assert hp["autocast_dtype"] == "float16"
    assert hp["standardize"] is True


# ----- 5. validation --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"input_channels": 0},
        {"coreset_size": 0},
        {"stride": 0},
        {"bank_stride": 0},
        {"pool_size": 2},
        {"pool_size": 0},
        {"max_bank_size": 10},  # < coreset_size
        {"topk_frac": 0.0},
        {"topk_frac": 1.5},
        {"chunk_size": 0},
        {"autocast_dtype": "float64"},
        {"search_backend": "bogus"},
        {"search_bit_width": 3},
        {"search_bit_width": 8},
        {"search_backend": "exact", "rescore": False},
        {"shortlist_size": 0},
        {"shortlist_size": -1},
    ],
)
def test_invalid_hparams_raise(bad):
    with pytest.raises(ValueError):
        PatchCoreDetector(**{**FIT_KW, **bad})


def test_search_hparams_default_to_the_unchanged_exact_path():
    node = PatchCoreDetector(**FIT_KW)
    assert node.search_backend == "exact"
    assert node.search_bit_width == 4
    assert node.shortlist_size == 8
    assert node.rescore is True


def test_cpu_ignores_autocast_and_matches_fp32():
    node16 = _fitted(autocast_dtype="float16")
    node32 = _fitted()
    cube = torch.rand(1, H, W, C, generator=torch.Generator().manual_seed(4))
    assert torch.equal(node16(cube=cube)["scores"], node32(cube=cube)["scores"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for autocast path")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_cuda_autocast_close_to_fp32(dtype):
    node32 = _fitted().cuda()
    nodelp = _fitted(autocast_dtype=dtype).cuda()
    cube = (torch.rand(1, H, W, C, generator=torch.Generator().manual_seed(4)) * 3 + 1).cuda()
    s32 = node32(cube=cube)["scores"]
    slp = nodelp(cube=cube)["scores"]
    assert slp.dtype == torch.float32
    tol = 2e-2 if dtype == "float16" else 6e-2  # bf16 keeps 8 mantissa bits
    assert torch.allclose(slp, s32, rtol=tol, atol=tol), (slp - s32).abs().max()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for autocast path")
def test_cuda_fp16_survives_extreme_pixels():
    """A hot pixel hundreds of sigma out must score finite and highest, not poison the frame."""
    node16 = _fitted(autocast_dtype="float16").cuda()
    node32 = _fitted().cuda()
    cube = (torch.rand(1, H, W, C, generator=torch.Generator().manual_seed(4)) * 3 + 1).cuda()
    cube[0, H // 2, W // 2, :] = 1e4  # ~ thousands of sigma on every band
    s16 = node16(cube=cube)["scores"][0, ..., 0]
    s32 = node32(cube=cube)["scores"][0, ..., 0]
    assert torch.isfinite(s16).all()
    assert s16.argmax() == s32.argmax()
    # Away from the hot pixel the clamped path still matches float32. The 3x3 pool spreads the
    # pixel to neighbouring stride-3 grid cells and the bilinear upsample reaches one more cell,
    # so exclude a generous +-8 px window.
    far = torch.ones_like(s16, dtype=torch.bool)
    far[H // 2 - 8 : H // 2 + 9, W // 2 - 8 : W // 2 + 9] = False
    assert far.any()
    assert torch.allclose(s16[far], s32[far], rtol=2e-2, atol=2e-2)


# ----- 6. feature-grid mode: standardize=False and the reference input -------------------------

D = 6  # feature dimension of the synthetic "patch-token" grids
GRID_KW = {
    "input_channels": D,
    "coreset_size": 30,
    "stride": 1,
    "bank_stride": 1,
    "pool_size": 1,
    "max_bank_size": 300,
    "standardize": False,
}


def _grids(n: int = 3, seed: int = 21) -> list[dict[str, torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    return [{"cube": torch.randn(1, 9, 8, D, generator=g) * 3.0 + 1.5} for _ in range(n)]


def test_feature_mode_keeps_identity_stats_and_scores_raw_distance():
    node = PatchCoreDetector(**GRID_KW)
    node.statistical_initialization(iter(_grids()))
    assert torch.equal(node.mu, torch.zeros(D)) and torch.equal(node.sd, torch.ones(D))
    q = torch.randn(1, 9, 8, D, generator=torch.Generator().manual_seed(3))
    out = node(cube=q)["scores"][0, ..., 0]
    expected = torch.cdist(q.reshape(-1, D), node.coreset).min(1).values.reshape(9, 8)
    assert torch.allclose(out, expected, atol=1e-5)
    # the coreset rows are raw (unstandardised) bank members, copied verbatim (exact equality:
    # ``torch.cdist`` would report ~sqrt(eps * |x|^2) for identical rows via its mm expansion)
    bank = torch.cat([b["cube"].reshape(-1, D) for b in _grids()])
    assert all((bank == row).all(dim=1).any() for row in node.coreset)


def test_standardize_switch_changes_the_metric_on_anisotropic_channels():
    node_z, node_raw = _fitted(), _fitted(standardize=False)
    cube = torch.rand(1, H, W, C, generator=torch.Generator().manual_seed(8)) * 3 + 1
    assert torch.equal(node_raw.sd, torch.ones(C))
    assert not torch.allclose(node_z(cube=cube)["scores"], node_raw(cube=cube)["scores"])


def test_reference_sets_output_resolution_and_topk_pool():
    node = _fitted()
    cube = torch.rand(2, H, W, C, generator=torch.Generator().manual_seed(12))
    ref = torch.zeros(2, 50, 41, 3)
    out = node(cube=cube, reference=ref)
    assert out["scores"].shape == (2, 50, 41, 1)
    q = node._features(cube, node.stride)
    gh, gw = q.shape[-2:]
    dist = node._nearest_distance(q.permute(0, 2, 3, 1).reshape(-1, C)).reshape(2, 1, gh, gw)
    expected = F.interpolate(dist, size=(50, 41), mode="bilinear", align_corners=False)
    assert torch.allclose(out["scores"], expected.permute(0, 2, 3, 1), atol=1e-6)
    k = max(1, int(node.topk_frac * 50 * 41))
    topk = torch.topk(out["scores"].reshape(2, -1), k, dim=1).values.mean(1)
    assert torch.allclose(out["anomaly_score"], topk)
    assert node(cube=cube)["scores"].shape == (2, H, W, 1)  # no reference: input size


def test_reference_port_is_optional():
    node = PatchCoreDetector(**FIT_KW)
    assert node.INPUT_SPECS["reference"].optional is True
    assert "reference" not in node.OUTPUT_SPECS


def test_state_dict_layout_is_identical_in_feature_mode():
    node = PatchCoreDetector(**GRID_KW)
    node.statistical_initialization(iter(_grids()))
    assert set(node.state_dict()) == {"mu", "sd", "coreset"}
    fresh = PatchCoreDetector(**GRID_KW)
    fresh.load_state_dict(node.state_dict())
    assert fresh._statistically_initialized is True
    q = torch.randn(1, 9, 8, D, generator=torch.Generator().manual_seed(5))
    assert torch.equal(fresh(cube=q)["scores"], node(cube=q)["scores"])


def test_feature_mode_rejects_channel_mismatch():
    node = PatchCoreDetector(**GRID_KW)
    with pytest.raises(ValueError, match="channels"):
        node.statistical_initialization(iter([{"cube": torch.rand(1, 4, 4, D + 1)}]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for autocast path")
def test_cuda_fp16_raw_features_skip_the_z_clamp():
    """Without standardisation the +-64 clamp is not applied: features far beyond it still match."""
    g = torch.Generator().manual_seed(4)
    grids = [{"cube": torch.randn(1, 12, 10, D, generator=g) * 40.0} for _ in range(3)]
    n32 = PatchCoreDetector(**GRID_KW)
    n16 = PatchCoreDetector(**GRID_KW, autocast_dtype="float16")
    n32.statistical_initialization(iter(grids))
    n16.statistical_initialization(iter(grids))
    n32, n16 = n32.cuda(), n16.cuda()
    q = (torch.randn(1, 12, 10, D, generator=g) * 40.0).cuda()  # |x| well beyond 64
    s32 = n32(cube=q)["scores"]
    s16 = n16(cube=q)["scores"]
    assert torch.isfinite(s16).all()
    assert torch.allclose(s16, s32, rtol=2e-2, atol=2e-2 * float(s32.abs().max()))


def test_tf32_is_a_no_op_on_cpu_and_restores_the_process_setting():
    ref, tf = _fitted(), _fitted(tf32=True)
    assert tf.hparams["tf32"] is True and ref.hparams["tf32"] is False
    before = torch.get_float32_matmul_precision()
    cube = _stream(1, seed=9)[0]["cube"]
    a, b = ref(cube=cube), tf(cube=cube)
    assert torch.get_float32_matmul_precision() == before
    assert torch.equal(a["scores"], b["scores"]) and torch.equal(
        a["anomaly_score"], b["anomaly_score"]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for TF32")
def test_tf32_on_cuda_stays_close_to_fp32():
    ref, tf = _fitted().cuda(), _fitted(tf32=True).cuda()
    cube = _stream(1, seed=9)[0]["cube"].cuda()
    a, b = ref(cube=cube), tf(cube=cube)
    assert torch.allclose(b["scores"], a["scores"], rtol=1e-2, atol=1e-2)


# ----- 7. turbovec search backend ---------------------------------------------------------------

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
    """With rescore=False, the reported distance comes from the norm-reconstruction formula, not
    the true Euclidean distance. This test pins it on a tiny bank where the shortlist always
    covers every row (k == n_coreset_rows), so the only approximation left is the formula's own
    cosine input versus the exact cosine — checked here with a tolerance that catches a wrong
    formula while allowing turbovec's own quantisation error."""
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
    call — the restore path a Node base without the hook would leave it in) still builds the
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


def test_exact_backend_never_builds_a_turbovec_index():
    node = _fitted()
    node.prepare_inference()
    cube = _stream(1, seed=1)[0]["cube"]
    node(cube=cube)
    assert getattr(node, "_turbovec_state", None) is None
