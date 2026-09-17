"""ScoreMapFusion: golden rules, port contract, fan-in wiring, hparam validation."""

from __future__ import annotations

import json

import pytest
import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai_patchcore.node.fusion import ScoreMapFusion

pytestmark = pytest.mark.unit

B, H, W = 2, 5, 7


def _maps(n: int, seed: int = 0) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [torch.rand(B, H, W, 1, generator=g) for _ in range(n)]


# ----- 1. golden rules ------------------------------------------------------------------------


def test_mean_min_max_match_torch_reductions():
    maps = _maps(3)
    stack = torch.stack(maps)
    assert torch.allclose(ScoreMapFusion(mode="mean")(scores=maps)["scores"], stack.mean(0))
    assert torch.equal(ScoreMapFusion(mode="min")(scores=maps)["scores"], stack.amin(0))
    assert torch.equal(ScoreMapFusion(mode="max")(scores=maps)["scores"], stack.amax(0))


def test_wmean_uses_normalised_weights():
    a, b = _maps(2)
    out = ScoreMapFusion(mode="wmean", weights=[3.0, 1.0])(scores=[a, b])["scores"]
    assert torch.allclose(out, 0.75 * a + 0.25 * b, atol=1e-6)


def test_equal_weights_wmean_equals_mean():
    maps = _maps(4)
    wm = ScoreMapFusion(mode="wmean", weights=[1, 1, 1, 1])(scores=maps)["scores"]
    mean = ScoreMapFusion(mode="mean")(scores=maps)["scores"]
    assert torch.allclose(wm, mean, atol=1e-6)


def test_single_map_passes_through_unchanged():
    (a,) = _maps(1)
    for mode in ("mean", "min", "max"):
        assert torch.equal(ScoreMapFusion(mode=mode)(scores=[a])["scores"], a)
    assert torch.equal(ScoreMapFusion(mode="mean")(scores=a)["scores"], a)  # bare tensor


def test_fusion_is_differentiable():
    a, b = (m.requires_grad_(True) for m in _maps(2))
    ScoreMapFusion(mode="mean")(scores=[a, b])["scores"].sum().backward()
    assert torch.allclose(a.grad, torch.full_like(a, 0.5))


# ----- 2. port contract -----------------------------------------------------------------------


def test_port_contract():
    node = ScoreMapFusion()
    out = node(scores=_maps(2))
    assert set(out) == set(node.OUTPUT_SPECS)
    assert out["scores"].shape == (B, H, W, 1)
    assert out["scores"].dtype == node.OUTPUT_SPECS["scores"].dtype
    assert node.INPUT_SPECS["scores"].variadic is True
    assert node.requires_initial_fit is False
    assert node.TRAINABLE_BUFFERS == ()
    assert ExecutionStage.ALWAYS in node.execution_stages


def test_shape_mismatch_raises():
    a = torch.rand(B, H, W, 1)
    b = torch.rand(B, H + 1, W, 1)
    with pytest.raises(ValueError, match="shape"):
        ScoreMapFusion()(scores=[a, b])


def test_wmean_weight_count_mismatch_raises():
    with pytest.raises(ValueError, match="weights"):
        ScoreMapFusion(mode="wmean", weights=[0.5, 0.5])(scores=_maps(3))


# ----- 3. fan-in wiring -----------------------------------------------------------------------


class _MapSource(Node):
    """Module-scope test source emitting a constant score map."""

    INPUT_SPECS: dict[str, PortSpec] = {}
    OUTPUT_SPECS = {"scores": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, 1))}

    def __init__(self, value: float = 0.0, **kwargs) -> None:
        super().__init__(value=value, **kwargs)
        self.value = float(value)

    def forward(self, **_) -> dict[str, torch.Tensor]:
        return {"scores": torch.full((1, H, W, 1), self.value)}


def test_variadic_port_collects_every_inbound_map():
    pipe = CuvisPipeline("fusion_fan_in")
    a, b, c = _MapSource(0.2, name="a"), _MapSource(0.4, name="b"), _MapSource(0.9, name="c")
    fuse = ScoreMapFusion(mode="mean", name="fuse")
    for src in (a, b, c):
        pipe.connect(src.outputs.scores, fuse.inputs.scores)
    out = pipe.forward(batch={}, stage=ExecutionStage.INFERENCE)[("fuse", "scores")]
    assert torch.allclose(out, torch.full((1, H, W, 1), 0.5), atol=1e-6)


# ----- 4. validation / serialization ----------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"mode": "gmean"},
        {"mode": "wmean"},  # missing weights
        {"mode": "wmean", "weights": [1.0, -1.0]},
        {"mode": "wmean", "weights": [0.0, 0.0]},
        {"mode": "mean", "weights": [0.5, 0.5]},  # weights only for wmean
    ],
)
def test_invalid_hparams_raise(bad):
    with pytest.raises(ValueError):
        ScoreMapFusion(**bad)


def test_hparams_round_trip_json():
    node = ScoreMapFusion(mode="wmean", weights=[2, 1], name="fuse")
    hp = node.hparams
    assert hp["mode"] == "wmean" and hp["weights"] == [2.0, 1.0]
    json.dumps(hp)
    assert ScoreMapFusion(mode="mean").hparams["weights"] is None
