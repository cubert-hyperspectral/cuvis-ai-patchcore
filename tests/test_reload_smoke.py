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
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai_patchcore.node.patchcore import PatchCoreDetector

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[1]
C, H, W = 4, 20, 16


class _ConstantCubeSource(Node):
    """Module-scope test source: emits a deterministic random cube (seeded per instance)."""

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
