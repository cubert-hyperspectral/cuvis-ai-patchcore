"""Reload smoke: a fitted PatchCoreDetector inside a CuvisPipeline survives save_to_file ->
load_pipeline (yaml + .pt) and reproduces its scores — the guard against the class-location and
buffer-serialization traps described in the cuvis-ai-node skill."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from cuvis_ai_core.node.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory, NodeTag
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai_patchcore.node.calibration import ScoreRangeNormalizer
from cuvis_ai_patchcore.node.fusion import ScoreMapFusion
from cuvis_ai_patchcore.node.gate import FrameScoreGate
from cuvis_ai_patchcore.node.patchcore import PatchCoreDetector

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[1]
C, H, W = 4, 20, 16


class _ConstantCubeSource(Node):
    """Module-scope test source: emits a deterministic random cube (seeded per instance)."""

    _category = NodeCategory.SOURCE
    _tags = frozenset({NodeTag.TORCH})
    INPUT_SPECS: dict[str, PortSpec] = {}
    OUTPUT_SPECS = {"cube": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1))}

    def __init__(self, seed: int = 0, **kwargs) -> None:
        super().__init__(seed=seed, **kwargs)
        self.seed = int(seed)

    def forward(self, **_) -> dict[str, torch.Tensor]:
        g = torch.Generator().manual_seed(self.seed)
        return {"cube": torch.rand(1, H, W, C, generator=g) * 2.0 + 1.0}


def _fit(node: PatchCoreDetector) -> None:
    g = torch.Generator().manual_seed(123)
    node.statistical_initialization(
        iter([{"cube": torch.rand(1, H, W, C, generator=g) * 2 + 1} for _ in range(3)])
    )


def test_save_and_reload_reproduces_scores(tmp_path):
    src = _ConstantCubeSource(seed=5, name="src")
    node = PatchCoreDetector(
        input_channels=C, coreset_size=32, stride=2, bank_stride=2, max_bank_size=200, name="pc"
    )
    _fit(node)
    pipe = CuvisPipeline("patchcore_reload_smoke")
    pipe.connect(src.outputs.cube, node.inputs.cube)

    ctx = Context(stage=ExecutionStage.INFERENCE)
    before = pipe.forward(batch={}, context=ctx)[("pc", "scores")]

    yaml_path = tmp_path / "pc.yaml"
    pipe.save_to_file(str(yaml_path))
    pt_path = yaml_path.with_suffix(".pt")
    assert pt_path.exists(), "save_to_file must write the .pt next to the yaml"

    # Python-built pipelines carry no plugins: field; declare it as suggest-plugins-fix would.
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    cfg["plugins"] = ["patchcore"]
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    registry = NodeRegistry()
    registry.register_plugin(str(REPO / "plugins.yaml"))
    restored = CuvisPipeline.load_pipeline(
        str(yaml_path), weights_path=str(pt_path), device="cpu", node_registry=registry
    )
    restored_pc = next(n for n in restored.nodes if not isinstance(n, str) and n.name == "pc")
    assert isinstance(restored_pc, PatchCoreDetector)
    assert restored_pc._statistically_initialized is True
    assert restored_pc.hparams["coreset_size"] == 32 and restored_pc.hparams["stride"] == 2

    after = restored.forward(batch={}, context=ctx)[("pc", "scores")]
    assert after.shape == (1, H, W, 1)
    assert torch.allclose(after, before, atol=1e-6)


D = 6
GH, GW = 6, 5  # coarse "patch grid" of the synthetic feature source


class _ConstantGridSource(Node):
    """Module-scope test source: emits a deterministic random feature grid (seeded per instance)."""

    _category = NodeCategory.SOURCE
    _tags = frozenset({NodeTag.TORCH})
    INPUT_SPECS: dict[str, PortSpec] = {}
    OUTPUT_SPECS = {"features": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1))}

    def __init__(self, seed: int = 0, **kwargs) -> None:
        super().__init__(seed=seed, **kwargs)
        self.seed = int(seed)

    def forward(self, **_) -> dict[str, torch.Tensor]:
        g = torch.Generator().manual_seed(self.seed)
        return {"features": torch.randn(1, GH, GW, D, generator=g) * 3.0}


def test_two_bank_fusion_pipeline_reloads(tmp_path):
    """Spectral bank + feature bank (reference-upsampled) fused by the variadic node round-trips."""
    src = _ConstantCubeSource(seed=5, name="src")
    grid = _ConstantGridSource(seed=9, name="grid")
    pc = PatchCoreDetector(
        input_channels=C, coreset_size=32, stride=2, bank_stride=2, max_bank_size=200, name="pc"
    )
    pcf = PatchCoreDetector(
        input_channels=D,
        coreset_size=20,
        stride=1,
        bank_stride=1,
        pool_size=1,
        max_bank_size=200,
        standardize=False,
        name="pcf",
    )
    fuse = ScoreMapFusion(mode="wmean", weights=[0.5, 0.5], name="fuse")
    _fit(pc)
    g = torch.Generator().manual_seed(77)
    pcf.statistical_initialization(
        iter([{"cube": torch.randn(1, GH, GW, D, generator=g) * 3.0} for _ in range(3)])
    )
    pipe = CuvisPipeline("two_bank_fusion_smoke")
    pipe.connect(src.outputs.cube, pc.inputs.cube)
    pipe.connect(grid.outputs.features, pcf.inputs.cube)
    pipe.connect(src.outputs.cube, pcf.inputs.reference)
    pipe.connect(pc.outputs.scores, fuse.inputs.scores)
    pipe.connect(pcf.outputs.scores, fuse.inputs.scores)

    ctx = Context(stage=ExecutionStage.INFERENCE)
    out = pipe.forward(batch={}, context=ctx)
    before = out[("fuse", "scores")]
    assert out[("pcf", "scores")].shape == (1, H, W, 1)  # upsampled to the cube via reference
    assert before.shape == (1, H, W, 1)
    assert torch.allclose(before, 0.5 * (out[("pc", "scores")] + out[("pcf", "scores")]), atol=1e-6)

    yaml_path = tmp_path / "fusion.yaml"
    pipe.save_to_file(str(yaml_path))
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    cfg["plugins"] = ["patchcore"]
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    registry = NodeRegistry()
    registry.register_plugin(str(REPO / "plugins.yaml"))
    restored = CuvisPipeline.load_pipeline(
        str(yaml_path),
        weights_path=str(yaml_path.with_suffix(".pt")),
        device="cpu",
        node_registry=registry,
    )
    restored_pcf = next(n for n in restored.nodes if not isinstance(n, str) and n.name == "pcf")
    assert restored_pcf.hparams["standardize"] is False
    after = restored.forward(batch={}, context=ctx)[("fuse", "scores")]
    assert torch.allclose(after, before, atol=1e-6)


def test_calibrated_two_bank_fusion_pipeline_reloads(tmp_path):
    """Bank -> ScoreRangeNormalizer -> fusion: the calibrators' bounds survive the round-trip."""
    src = _ConstantCubeSource(seed=5, name="src")
    grid = _ConstantGridSource(seed=9, name="grid")
    pc = PatchCoreDetector(
        input_channels=C, coreset_size=32, stride=2, bank_stride=2, max_bank_size=200, name="pc"
    )
    pcf = PatchCoreDetector(
        input_channels=D,
        coreset_size=20,
        stride=1,
        bank_stride=1,
        pool_size=1,
        max_bank_size=200,
        standardize=False,
        name="pcf",
    )
    cal_pc = ScoreRangeNormalizer(fit_subsample=1, name="cal_pc")
    cal_pcf = ScoreRangeNormalizer(fit_subsample=1, name="cal_pcf")
    fuse = ScoreMapFusion(mode="mean", name="fuse")
    _fit(pc)
    g = torch.Generator().manual_seed(77)
    pcf.statistical_initialization(
        iter([{"cube": torch.randn(1, GH, GW, D, generator=g) * 3.0} for _ in range(3)])
    )
    # calibrate each bank on its own scores of "normal" inputs
    ref = torch.rand(1, H, W, C, generator=torch.Generator().manual_seed(1)) * 2 + 1
    cal_pc.statistical_initialization(iter([{"scores": pc(cube=ref)["scores"]}]))
    cal_pcf.statistical_initialization(
        iter(
            [
                {
                    "scores": pcf(cube=torch.randn(1, GH, GW, D, generator=g) * 3.0, reference=ref)[
                        "scores"
                    ]
                }
            ]
        )
    )
    pipe = CuvisPipeline("calibrated_two_bank_fusion_smoke")
    pipe.connect(src.outputs.cube, pc.inputs.cube)
    pipe.connect(grid.outputs.features, pcf.inputs.cube)
    pipe.connect(src.outputs.cube, pcf.inputs.reference)
    pipe.connect(pc.outputs.scores, cal_pc.inputs.scores)
    pipe.connect(pcf.outputs.scores, cal_pcf.inputs.scores)
    pipe.connect(cal_pc.outputs.normalized, fuse.inputs.scores)
    pipe.connect(cal_pcf.outputs.normalized, fuse.inputs.scores)

    ctx = Context(stage=ExecutionStage.INFERENCE)
    out = pipe.forward(batch={}, context=ctx)
    before = out[("fuse", "scores")]
    assert torch.isfinite(before).all() and before.min() >= 0  # floored, unclamped above

    yaml_path = tmp_path / "calibrated.yaml"
    pipe.save_to_file(str(yaml_path))
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    cfg["plugins"] = ["patchcore"]
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    registry = NodeRegistry()
    registry.register_plugin(str(REPO / "plugins.yaml"))
    restored = CuvisPipeline.load_pipeline(
        str(yaml_path),
        weights_path=str(yaml_path.with_suffix(".pt")),
        device="cpu",
        node_registry=registry,
    )
    cal = next(n for n in restored.nodes if not isinstance(n, str) and n.name == "cal_pc")
    assert isinstance(cal, ScoreRangeNormalizer) and cal._statistically_initialized is True
    assert torch.equal(cal.lo, cal_pc.lo) and torch.equal(cal.hi, cal_pc.hi)
    after = restored.forward(batch={}, context=ctx)[("fuse", "scores")]
    assert torch.allclose(after, before, atol=1e-6)


def test_gated_priority_display_pipeline_reloads(tmp_path):
    """Gates -> ScoreMapFusion(mode="first"): the gate hparams, the alarm_scores wiring and the
    variadic connection order (which `first` depends on) survive save -> load."""
    src = _ConstantCubeSource(seed=5, name="src")
    pc = PatchCoreDetector(
        input_channels=C, coreset_size=32, stride=2, bank_stride=2, max_bank_size=200, name="pc"
    )
    cal = ScoreRangeNormalizer(fit_subsample=1, name="cal")
    _fit(pc)
    ref = torch.rand(1, H, W, C, generator=torch.Generator().manual_seed(1)) * 2 + 1
    cal.statistical_initialization(iter([{"scores": pc(cube=ref)["scores"]}]))
    # gate_blank never opens; gate_a (alarm on the raw bank map) and gate_b (mask) both open
    blank = FrameScoreGate(threshold=1e9, name="gate_blank")
    gate_a = FrameScoreGate(threshold=0.0, topk_frac=0.01, name="gate_a")
    gate_b = FrameScoreGate(threshold=0.0, mode="mask", mask_threshold=0.5, name="gate_b")
    fuse = ScoreMapFusion(mode="first", name="fuse")
    pipe = CuvisPipeline("gated_priority_display_smoke")
    pipe.connect(src.outputs.cube, pc.inputs.cube)
    pipe.connect(pc.outputs.scores, cal.inputs.scores)
    pipe.connect(cal.outputs.normalized, blank.inputs.scores)
    pipe.connect(cal.outputs.normalized, gate_a.inputs.scores)
    pipe.connect(pc.outputs.scores, gate_a.inputs.alarm_scores)
    pipe.connect(cal.outputs.normalized, gate_b.inputs.scores)
    for gate in (blank, gate_a, gate_b):
        pipe.connect(gate.outputs.scores, fuse.inputs.scores)

    ctx = Context(stage=ExecutionStage.INFERENCE)
    before = pipe.forward(batch={}, context=ctx)
    assert before[("gate_blank", "passed")].tolist() == [0]
    assert before[("gate_a", "passed")].tolist() == [1] and before[
        ("gate_b", "passed")
    ].tolist() == [1]
    assert torch.equal(before[("fuse", "scores")], before[("gate_a", "scores")])
    assert not torch.equal(before[("gate_a", "scores")], before[("gate_b", "scores")])

    yaml_path = tmp_path / "gated.yaml"
    pipe.save_to_file(str(yaml_path))
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    cfg["plugins"] = ["patchcore"]
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    registry = NodeRegistry()
    registry.register_plugin(str(REPO / "plugins.yaml"))
    restored = CuvisPipeline.load_pipeline(
        str(yaml_path),
        weights_path=str(yaml_path.with_suffix(".pt")),
        device="cpu",
        node_registry=registry,
    )
    nodes = {n.name: n for n in restored.nodes if not isinstance(n, str)}
    assert nodes["fuse"].hparams["mode"] == "first"
    assert nodes["gate_blank"].hparams["threshold"] == 1e9
    assert nodes["gate_b"].hparams["mode"] == "mask"
    assert nodes["gate_b"].hparams["mask_threshold"] == 0.5
    assert nodes["gate_a"].hparams["topk_frac"] == 0.01
    after = restored.forward(batch={}, context=ctx)
    for key in (("gate_a", "frame_score"), ("gate_b", "scores"), ("fuse", "scores")):
        assert torch.allclose(after[key], before[key], atol=1e-6), key
    assert torch.equal(after[("fuse", "scores")], after[("gate_a", "scores")])
