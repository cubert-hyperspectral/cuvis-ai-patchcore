"""ScoreRangeNormalizer: golden percentiles vs numpy, unclamped-above / floored-below behaviour,
Phase-1 subsampling and cap, state-dict round-trip, port contract, hparam validation."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from cuvis_ai_schemas.enums import ExecutionStage

from cuvis_ai_patchcore.node.calibration import ScoreRangeNormalizer

pytestmark = pytest.mark.unit

H, W = 23, 17


def _stream(n: int = 3, c: int = 1, seed: int = 0) -> list[dict[str, torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    return [{"scores": torch.rand(1, H, W, c, generator=g) * 5.0 + 1.0} for _ in range(n)]


def _numpy_bounds(stream, s: int, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
    vals = np.concatenate(
        [b["scores"][:, ::s, ::s, :].reshape(-1, b["scores"].shape[-1]).numpy() for b in stream]
    )
    return np.percentile(vals, low, axis=0), np.percentile(vals, high, axis=0)


# ----- 1. golden --------------------------------------------------------------------------------


def test_fitted_bounds_match_numpy_percentiles_of_the_subsampled_stream():
    node = ScoreRangeNormalizer(low=1.0, high=99.0, fit_subsample=2)
    node.statistical_initialization(iter(_stream()))
    lo, hi = _numpy_bounds(_stream(), 2, 1.0, 99.0)
    assert np.allclose(node.lo.numpy(), lo, atol=1e-5) and np.allclose(
        node.hi.numpy(), hi, atol=1e-5
    )
    x = _stream(1, seed=9)[0]["scores"]
    out = node(scores=x)["normalized"]
    expected = np.clip((x.numpy() - lo) / (hi - lo), 0, None)
    assert np.allclose(out.numpy(), expected, atol=1e-5)


def test_per_channel_bounds():
    node = ScoreRangeNormalizer(n_channels=2, low=5.0, high=95.0, fit_subsample=1)
    stream = _stream(2, c=2, seed=3)
    stream[0]["scores"][..., 1] *= 10.0  # second channel on another scale
    stream[1]["scores"][..., 1] *= 10.0
    node.statistical_initialization(iter(stream))
    lo, hi = _numpy_bounds(stream, 1, 5.0, 95.0)
    assert np.allclose(node.lo.numpy(), lo, atol=1e-4) and np.allclose(
        node.hi.numpy(), hi, atol=1e-4
    )


def test_unclamped_above_and_floored_below():
    node = ScoreRangeNormalizer(fit_subsample=1)
    node.statistical_initialization(iter(_stream()))
    hi = float(node.hi[0])
    lo = float(node.lo[0])
    x = torch.tensor([[[[hi * 10.0], [lo - 1.0], [hi]]]])  # far above / below / at the top
    out = node(scores=x)["normalized"][0, 0, :, 0]
    assert out[0] > 9.0  # an anomaly far above the normal range keeps its magnitude
    assert out[1] == 0.0  # below the low percentile is "normal" -> floored
    assert torch.isclose(out[2], torch.tensor(1.0), atol=1e-5)
    raw = ScoreRangeNormalizer(fit_subsample=1, floor=False)
    raw.statistical_initialization(iter(_stream()))
    assert raw(scores=x)["normalized"][0, 0, 1, 0] < 0.0


def test_cap_is_seeded_and_bounds_stay_close():
    small = ScoreRangeNormalizer(fit_subsample=1, max_fit_values=200, seed=1)
    small.statistical_initialization(iter(_stream(4)))
    again = ScoreRangeNormalizer(fit_subsample=1, max_fit_values=200, seed=1)
    again.statistical_initialization(iter(_stream(4)))
    assert torch.equal(small.lo, again.lo) and torch.equal(small.hi, again.hi)
    full = ScoreRangeNormalizer(fit_subsample=1)
    full.statistical_initialization(iter(_stream(4)))
    assert abs(float(small.hi[0] - full.hi[0])) < 0.5  # a 200-value sample of a [1, 6] range


# ----- 2. port contract / lifecycle -------------------------------------------------------------


def test_port_contract_and_metadata():
    node = ScoreRangeNormalizer()
    assert node.requires_initial_fit is True and node.TRAINABLE_BUFFERS == ()
    assert ExecutionStage.ALWAYS in node.execution_stages
    with pytest.raises(RuntimeError, match="statistical_initialization"):
        node(scores=torch.rand(1, H, W, 1))
    node.statistical_initialization(iter(_stream()))
    out = node(scores=torch.rand(2, H, W, 1))
    assert set(out) == set(node.OUTPUT_SPECS)
    assert out["normalized"].shape == (2, H, W, 1)
    assert out["normalized"].dtype == node.OUTPUT_SPECS["normalized"].dtype


def test_fit_rejects_channel_mismatch_and_empty_stream():
    node = ScoreRangeNormalizer(n_channels=1)
    with pytest.raises(ValueError, match="channels"):
        node.statistical_initialization(iter([{"scores": torch.rand(1, 4, 4, 2)}]))
    with pytest.raises(RuntimeError, match="no score maps"):
        node.statistical_initialization(iter([]))
    assert node._statistically_initialized is False


def test_state_dict_round_trip_marks_node_fitted():
    fitted = ScoreRangeNormalizer(fit_subsample=1)
    fitted.statistical_initialization(iter(_stream()))
    fresh = ScoreRangeNormalizer(fit_subsample=1)
    assert set(fitted.state_dict()) == {"lo", "hi"}
    fresh.load_state_dict(fitted.state_dict())
    assert fresh._statistically_initialized is True
    x = torch.rand(1, H, W, 1, generator=torch.Generator().manual_seed(5)) * 8
    assert torch.equal(fresh(scores=x)["normalized"], fitted(scores=x)["normalized"])


def test_hparams_json_round_trip():
    node = ScoreRangeNormalizer(
        n_channels=1, low=2, high=98, floor=False, fit_subsample=3, name="cal"
    )
    hp = node.hparams
    json.dumps(hp)
    assert (
        hp["low"] == 2.0
        and hp["high"] == 98.0
        and hp["floor"] is False
        and hp["fit_subsample"] == 3
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"n_channels": 0},
        {"low": -1.0},
        {"low": 50.0, "high": 50.0},
        {"high": 101.0},
        {"fit_subsample": 0},
        {"max_fit_values": 1},
    ],
)
def test_invalid_hparams_raise(bad):
    with pytest.raises(ValueError):
        ScoreRangeNormalizer(**bad)
