"""Shared node helpers: the top-k image score (one k rule for detector and gate), the topk_frac
check, the seeded row cap and the Phase-1 guard."""

from __future__ import annotations

import pytest
import torch
from cuvis_ai_core.node.metric_utils import subsample_hw

from cuvis_ai_patchcore.node._common import check_topk_frac, random_cap, require_fitted, topk_mean
from cuvis_ai_patchcore.node.calibration import ScoreRangeNormalizer
from cuvis_ai_patchcore.node.gate import FrameScoreGate
from cuvis_ai_patchcore.node.patchcore import PatchCoreDetector

pytestmark = pytest.mark.unit


def test_topk_mean_golden_and_floor_rule():
    x = torch.tensor([[1.0, 5.0, 3.0, 2.0], [0.0, -1.0, 4.0, 4.0]]).reshape(2, 2, 2, 1)
    assert torch.equal(topk_mean(x, 0.5), torch.tensor([4.0, 4.0]))  # k = 2
    assert torch.equal(topk_mean(x, 0.7), torch.tensor([4.0, 4.0]))  # floor(2.8) = 2
    assert torch.equal(topk_mean(x, 0.01), torch.tensor([5.0, 4.0]))  # k >= 1: the maximum
    assert torch.equal(topk_mean(x, 1.0), x.reshape(2, -1).mean(dim=1))


def test_topk_mean_pools_every_non_batch_value_and_returns_float32():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(3, 5, 4, 2, generator=g, dtype=torch.float64)
    out = topk_mean(x, 0.1)  # k = floor(0.1 * 40) = 4 over H * W * C
    expected = torch.topk(x.reshape(3, -1).float(), 4, dim=1).values.mean(dim=1)
    assert out.dtype == torch.float32 and out.shape == (3,)
    assert torch.equal(out, expected)


def test_detector_anomaly_score_equals_gate_frame_score_on_its_map():
    # 1160 x 25: rounding topk_frac * H * W in two steps (1.16 * 25) would lose a pixel (k = 28
    # instead of 29); the shared rule rounds once, so a gate on the map reproduces the score.
    g = torch.Generator().manual_seed(1)
    node = PatchCoreDetector(
        input_channels=3, coreset_size=8, stride=1, bank_stride=1, max_bank_size=64
    )
    node.statistical_initialization(iter([{"cube": torch.rand(1, 6, 5, 3, generator=g)}]))
    out = node(cube=torch.rand(2, 6, 5, 3, generator=g), reference=torch.zeros(2, 1160, 25, 1))
    gate = FrameScoreGate(threshold=0.0, topk_frac=node.topk_frac)
    assert torch.equal(gate(scores=out["scores"])["frame_score"], out["anomaly_score"])
    k = torch.topk(out["scores"].reshape(2, -1), 29, dim=1).values.mean(dim=1)
    assert torch.equal(out["anomaly_score"], k)


@pytest.mark.parametrize("frac", [1e-6, 0.001, 0.5, 1.0, 1])
def test_check_topk_frac_accepts_the_unit_interval(frac):
    assert check_topk_frac(frac) == float(frac)
    assert isinstance(check_topk_frac(frac), float)


@pytest.mark.parametrize("frac", [0.0, -0.1, 1.0001, 2])
def test_check_topk_frac_rejects_values_outside_it(frac):
    with pytest.raises(ValueError, match="topk_frac must be in"):
        check_topk_frac(frac)


def test_random_cap_within_the_cap_returns_rows_and_leaves_the_generator():
    rows = torch.arange(20.0).reshape(10, 2)
    gen = torch.Generator().manual_seed(4)
    state = gen.get_state()
    assert random_cap(rows, 10, gen) is rows
    assert torch.equal(gen.get_state(), state)


def test_random_cap_draws_a_seeded_subset_without_repeats():
    rows = torch.arange(200.0).reshape(100, 2)
    capped = random_cap(rows, 30, torch.Generator().manual_seed(4))
    keep = torch.randperm(100, generator=torch.Generator().manual_seed(4))[:30]
    assert torch.equal(capped, rows[keep])
    assert capped[:, 0].unique().numel() == 30


def test_require_fitted_names_the_node_until_phase_1():
    node = ScoreRangeNormalizer()
    with pytest.raises(RuntimeError, match="ScoreRangeNormalizer requires statistical_init"):
        require_fitted(node)
    node.statistical_initialization(iter([{"scores": torch.rand(1, 8, 8, 1)}]))
    require_fitted(node)


@pytest.mark.parametrize("stride", [1, 2, 3, 4])
def test_core_subsample_matches_the_strided_slice(stride):
    # ScoreRangeNormalizer thins its Phase-1 values with core's subsample_hw; pin it to the slice
    x = torch.rand(2, 11, 9, 3, generator=torch.Generator().manual_seed(stride))
    assert torch.equal(subsample_hw(x, stride), x[:, ::stride, ::stride, :])
