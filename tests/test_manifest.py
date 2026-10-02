"""Manifest loading: the checked-in plugins.yaml resolves through NodeRegistry (the skill's
required verification step) and every capability imports as a Node subclass."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml
from cuvis_ai_core.node.node import Node
from cuvis_ai_core.utils.node_registry import NodeRegistry

from cuvis_ai_patchcore.node.calibration import ScoreRangeNormalizer
from cuvis_ai_patchcore.node.fusion import DecisionFusion, MaskComposite, ScoreMapFusion
from cuvis_ai_patchcore.node.gate import FrameScoreGate
from cuvis_ai_patchcore.node.morphology import MaskBlobGate, MaskMinArea
from cuvis_ai_patchcore.node.objectness import MaskBlobFilter, SpectralObjectMask
from cuvis_ai_patchcore.node.patchcore import PatchCoreDetector
from cuvis_ai_patchcore.node.spatial import ScoreMapSmoothing
from cuvis_ai_patchcore.node.temporal import MaskPersistence

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "plugins.yaml"


def test_manifest_registers_plugin_and_resolves_nodes():
    registry = NodeRegistry()
    registry.register_plugin(str(MANIFEST))
    assert registry.list_plugins() == ["patchcore"]
    assert registry.get("PatchCoreDetector") is PatchCoreDetector
    assert registry.get("ScoreMapFusion") is ScoreMapFusion
    assert registry.get("ScoreRangeNormalizer") is ScoreRangeNormalizer
    assert registry.get("FrameScoreGate") is FrameScoreGate
    assert registry.get("DecisionFusion") is DecisionFusion
    assert registry.get("MaskComposite") is MaskComposite
    assert registry.get("MaskPersistence") is MaskPersistence
    assert registry.get("MaskMinArea") is MaskMinArea
    assert registry.get("ScoreMapSmoothing") is ScoreMapSmoothing
    assert registry.get("MaskBlobGate") is MaskBlobGate
    assert registry.get("SpectralObjectMask") is SpectralObjectMask
    assert registry.get("MaskBlobFilter") is MaskBlobFilter


def test_manifest_capabilities_are_importable_nodes():
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["name"] == "patchcore"
    assert manifest["package_name"] == "cuvis-ai-patchcore"
    assert manifest["capabilities"], "capabilities must not be empty"
    for entry in manifest["capabilities"]:
        module_name, _, cls_name = entry["class_name"].rpartition(".")
        cls = getattr(importlib.import_module(module_name), cls_name)
        assert issubclass(cls, Node), entry["class_name"]
