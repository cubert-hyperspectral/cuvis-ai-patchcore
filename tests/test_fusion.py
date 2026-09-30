"""ScoreMapFusion and DecisionFusion: golden rules, port contract, fan-in, hparam validation."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai_patchcore.node.fusion import DecisionFusion, MaskComposite, ScoreMapFusion

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


def test_first_takes_the_first_nonzero_map_per_frame():
    a, b = _maps(2)
    a[0] = 0.0  # frame 0: the first map is blank (a gate that did not open)
    out = ScoreMapFusion(mode="first")(scores=[a, b])["scores"]
    assert torch.equal(out[0], b[0])  # falls through to the second map
    assert torch.equal(out[1], a[1])  # the first map wins wherever it is live
    blank = [torch.zeros(B, H, W, 1), torch.zeros(B, H, W, 1)]
    assert torch.equal(ScoreMapFusion(mode="first")(scores=blank)["scores"], blank[0])


def test_single_map_passes_through_unchanged():
    (a,) = _maps(1)
    for mode in ("mean", "min", "max", "first"):
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

    _category = NodeCategory.SOURCE
    _tags = frozenset({NodeTag.TORCH})
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


def test_first_follows_connection_order():
    pipe = CuvisPipeline("fusion_priority")
    a, b, c = _MapSource(0.0, name="a"), _MapSource(0.4, name="b"), _MapSource(0.9, name="c")
    fuse = ScoreMapFusion(mode="first", name="fuse")
    for src in (a, b, c):  # a is blank, so b (connected before c) is shown
        pipe.connect(src.outputs.scores, fuse.inputs.scores)
    out = pipe.forward(batch={}, stage=ExecutionStage.INFERENCE)[("fuse", "scores")]
    assert torch.equal(out, torch.full((1, H, W, 1), 0.4))


# ----- 4. validation / serialization ----------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"mode": "gmean"},
        {"mode": "wmean"},  # missing weights
        {"mode": "wmean", "weights": [1.0, -1.0]},
        {"mode": "wmean", "weights": [0.0, 0.0]},
        {"mode": "mean", "weights": [0.5, 0.5]},  # weights only for wmean
        {"mode": "first", "weights": [1.0, 1.0]},
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


# ----- 5. DecisionFusion ----------------------------------------------------------------------


def _masks() -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.zeros(B, H, W, 1, dtype=torch.bool)
    a[0, 0, 0, 0] = True  # frame 0 only
    b = torch.zeros(B, H, W, 1, dtype=torch.bool)
    b[:, 1, 1, 0] = True  # both frames
    return a, b


def test_decision_any_all_are_pixelwise_or_and():
    a, b = _masks()
    assert torch.equal(DecisionFusion(mode="any")(decisions=[a, b])["decisions"], a | b)
    assert torch.equal(DecisionFusion(mode="all")(decisions=[a, b])["decisions"], a & b)


def test_decision_first_takes_the_first_mask_with_a_set_pixel_per_frame():
    a, b = _masks()
    out = DecisionFusion(mode="first")(decisions=[a, b])["decisions"]
    assert torch.equal(out[0], a[0])  # frame 0: a has a pixel
    assert torch.equal(out[1], b[1])  # frame 1: a is empty, so b
    none = torch.zeros_like(a)
    assert not DecisionFusion(mode="first")(decisions=[none, none])["decisions"].any()


def test_decision_single_mask_passes_through_unchanged():
    a, _ = _masks()
    for mode in ("any", "all", "first"):
        assert torch.equal(DecisionFusion(mode=mode)(decisions=a)["decisions"], a)


def test_decision_port_contract():
    a, b = _masks()
    out = DecisionFusion()(decisions=[a, b])
    assert set(out) == set(DecisionFusion.OUTPUT_SPECS)
    assert out["decisions"].shape == (B, H, W, 1) and out["decisions"].dtype == torch.bool


def test_decision_shape_mismatch_raises():
    a, _ = _masks()
    with pytest.raises(ValueError):
        DecisionFusion()(decisions=[a, torch.zeros(B, H, W + 1, 1, dtype=torch.bool)])


class _MaskSource(Node):
    """Module-scope test source emitting a constant mask (one pixel set, or none)."""

    _category = NodeCategory.SOURCE
    _tags = frozenset({NodeTag.TORCH})
    INPUT_SPECS: dict[str, PortSpec] = {}
    OUTPUT_SPECS = {"decisions": PortSpec(dtype=torch.bool, shape=(-1, -1, -1, 1))}

    def __init__(self, pixel: int = -1, **kwargs) -> None:
        super().__init__(pixel=pixel, **kwargs)
        self.pixel = int(pixel)

    def forward(self, **_) -> dict[str, torch.Tensor]:
        m = torch.zeros(1, H * W, dtype=torch.bool)
        if self.pixel >= 0:
            m[0, self.pixel] = True
        return {"decisions": m.view(1, H, W, 1)}


def test_decision_first_follows_connection_order():
    pipe = CuvisPipeline("decision_priority")
    a, b, c = _MaskSource(-1, name="a"), _MaskSource(3, name="b"), _MaskSource(9, name="c")
    fuse = DecisionFusion(mode="first", name="fuse")
    for src in (a, b, c):  # a is empty, so b (connected before c) is taken
        pipe.connect(src.outputs.decisions, fuse.inputs.decisions)
    out = pipe.forward(batch={}, stage=ExecutionStage.INFERENCE)[("fuse", "decisions")]
    assert out.flatten().nonzero().flatten().tolist() == [3]


@pytest.mark.parametrize("bad", [{"mode": "union"}, {"mode": "mean"}])
def test_decision_invalid_mode_raises(bad):
    with pytest.raises(ValueError):
        DecisionFusion(**bad)


def test_decision_hparams_round_trip_json():
    node = DecisionFusion(mode="first", name="dfuse")
    assert node.hparams["mode"] == "first"
    json.dumps(node.hparams)
    assert DecisionFusion().hparams["mode"] == "any"


# ----- 6. MaskComposite -----------------------------------------------------------------------


def test_composite_labels_and_levels_largest_wins_on_overlap():
    shell = torch.zeros(B, H, W, 1, dtype=torch.bool)
    shell[:, 0:2, 0:3, 0] = True
    fo = torch.zeros(B, H, W, 1, dtype=torch.bool)
    fo[0, 1:3, 2:4, 0] = True  # overlaps the shell at (1, 2) in frame 0 only
    out = MaskComposite(labels=[1, 2], levels=[0.5, 1.0])(decisions=[shell, fo])
    exp_mask = torch.zeros(B, H, W, dtype=torch.int32)
    exp_mask[:, 0:2, 0:3] = 1
    exp_mask[0, 1:3, 2:4] = 2
    exp_level = torch.zeros(B, H, W, 1)
    exp_level[:, 0:2, 0:3, 0] = 0.5
    exp_level[0, 1:3, 2:4, 0] = 1.0
    assert torch.equal(out["mask"], exp_mask)
    assert torch.equal(out["scores"], exp_level)
    # the order of the lists, not of the masks' size, decides: a larger label on the first mask wins
    flipped = MaskComposite(labels=[2, 1], levels=[1.0, 0.5])(decisions=[shell, fo])
    assert int(flipped["mask"][0, 1, 2]) == 2 and float(flipped["scores"][0, 1, 2, 0]) == 1.0


def test_composite_defaults_are_shell_1_fo_2():
    node = MaskComposite()
    assert node.labels == [1, 2] and node.levels == [0.5, 1.0]
    empty = torch.zeros(B, H, W, 1, dtype=torch.bool)
    out = node(decisions=[empty, empty])
    assert not out["mask"].any() and not out["scores"].any()


def test_composite_counts_a_pixel_where_any_channel_is_set():
    m = torch.zeros(B, H, W, 2, dtype=torch.bool)
    m[0, 2, 3, 1] = True
    out = MaskComposite(labels=[4], levels=[0.25])(decisions=m)
    assert out["mask"].flatten().nonzero().flatten().tolist() == [2 * W + 3]
    assert float(out["scores"][0, 2, 3, 0]) == 0.25


def test_composite_port_contract():
    a, b = _masks()
    out = MaskComposite()(decisions=[a, b])
    assert set(out) == set(MaskComposite.OUTPUT_SPECS)
    assert out["mask"].shape == (B, H, W) and out["mask"].dtype == torch.int32
    assert out["scores"].shape == (B, H, W, 1) and out["scores"].dtype == torch.float32


def test_composite_mask_count_and_shape_mismatch_raise():
    a, b = _masks()
    with pytest.raises(ValueError):
        MaskComposite()(decisions=[a])  # two labels configured, one mask connected
    with pytest.raises(ValueError):
        MaskComposite()(decisions=[a, torch.zeros(B, H, W + 1, 1, dtype=torch.bool)])


def test_composite_labels_follow_connection_order():
    pipe = CuvisPipeline("composite_order")
    a, b = _MaskSource(3, name="a"), _MaskSource(9, name="b")
    comp = MaskComposite(labels=[1, 2], levels=[0.5, 1.0], name="comp")
    for src in (a, b):
        pipe.connect(src.outputs.decisions, comp.inputs.decisions)
    out = pipe.forward(batch={}, stage=ExecutionStage.INFERENCE)
    mask = out[("comp", "mask")].flatten()
    assert int(mask[3]) == 1 and int(mask[9]) == 2 and int(mask.count_nonzero()) == 2


@pytest.mark.parametrize(
    "bad",
    [
        {"labels": [], "levels": []},
        {"labels": [1, 2], "levels": [1.0]},
        {"labels": [0, 2]},
        {"labels": [1, True]},
        {"labels": [1.5, 2]},
        {"levels": [-0.1, 1.0]},
        {"levels": [0.5, float("nan")]},
        {"levels": [0.5, float("inf")]},
    ],
)
def test_composite_invalid_hparams_raise(bad):
    with pytest.raises(ValueError):
        MaskComposite(**bad)


def test_composite_hparams_round_trip_json():
    node = MaskComposite(labels=[1, 2], levels=[0.4, 1.0], name="comp")
    assert node.hparams["labels"] == [1, 2] and node.hparams["levels"] == [0.4, 1.0]
    json.dumps(node.hparams)


# ----- 7. ScoreMapFusion softmin ------------------------------------------------------------


def _numpy_softmin(maps: list[torch.Tensor], beta: float, weights=None) -> np.ndarray:
    x = np.stack([m.numpy().astype(np.float64) for m in maps])
    w = np.ones(len(maps)) if weights is None else np.asarray(weights, np.float64)
    w = (w / w.sum()).reshape(-1, 1, 1, 1, 1)
    return -np.log((w * np.exp(-beta * x)).sum(0)) / beta


@pytest.mark.parametrize("weights", [None, [2.0, 1.0, 1.0]])
def test_softmin_matches_the_closed_form(weights):
    maps = _maps(3, seed=4)
    out = ScoreMapFusion(mode="softmin", beta=10.0, weights=weights)(scores=maps)["scores"]
    assert np.allclose(out.numpy(), _numpy_softmin(maps, 10.0, weights), atol=1e-5)


def test_softmin_lies_between_the_minimum_and_the_mean():
    maps = _maps(3, seed=5)
    stack = torch.stack(maps)
    hard = ScoreMapFusion(mode="softmin", beta=1e4)(scores=maps)["scores"]
    assert torch.all(hard >= stack.amin(0) - 1e-6)
    assert torch.all(hard <= stack.amin(0) + np.log(3) / 1e4 + 1e-5)  # min + log(N) / beta
    # the small-beta limit in float64: in float32 the rounding error grows like eps / beta
    maps64 = [m.double() for m in maps]
    soft = ScoreMapFusion(mode="softmin", beta=1e-6)(scores=maps64)["scores"]
    assert torch.allclose(soft, torch.stack(maps64).mean(0), atol=1e-6)


def test_softmin_is_stable_for_large_scores():
    a, b = torch.full((1, 2, 2, 1), 1e4), torch.full((1, 2, 2, 1), 2e4)
    out = ScoreMapFusion(mode="softmin", beta=50.0)(scores=[a, b])["scores"]
    assert torch.isfinite(out).all() and torch.allclose(out, a + np.log(2) / 50.0, atol=1e-2)


def test_softmin_is_differentiable():
    maps = [m.requires_grad_() for m in _maps(2, seed=6)]
    ScoreMapFusion(mode="softmin", beta=3.0)(scores=maps)["scores"].sum().backward()
    assert all(m.grad is not None and torch.isfinite(m.grad).all() for m in maps)


def test_softmin_weight_count_mismatch_raises():
    node = ScoreMapFusion(mode="softmin", beta=2.0, weights=[1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        node(scores=_maps(2))


@pytest.mark.parametrize(
    "bad",
    [
        {"mode": "softmin"},  # missing beta
        {"mode": "softmin", "beta": 0.0},
        {"mode": "softmin", "beta": -1.0},
        {"mode": "softmin", "beta": float("inf")},
        {"mode": "softmin", "beta": 1.0, "weights": [1.0, -1.0]},
        {"mode": "mean", "beta": 1.0},  # beta only for softmin
        {"mode": "min", "beta": 1.0},
    ],
)
def test_softmin_invalid_hparams_raise(bad):
    with pytest.raises(ValueError):
        ScoreMapFusion(**bad)


def test_softmin_hparams_round_trip_and_beta_defaults_to_none():
    hp = ScoreMapFusion(mode="softmin", beta=10, weights=[1, 2], name="fuse").hparams
    assert hp["mode"] == "softmin" and hp["beta"] == 10.0 and hp["weights"] == [1.0, 2.0]
    json.dumps(hp)
    for mode in ("mean", "min", "max", "first"):
        assert ScoreMapFusion(mode=mode).hparams["beta"] is None
