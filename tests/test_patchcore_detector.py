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
        "standardize",
        "seed",
        "eps",
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
    ],
)
def test_invalid_hparams_raise(bad):
    with pytest.raises(ValueError):
        PatchCoreDetector(**{**FIT_KW, **bad})


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
