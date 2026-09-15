"""cuvis_ai_patchcore — HSI-PatchCore anomaly detection for cuvis-ai.

Plugin nodes are discovered through the ``plugins.yaml`` manifest (``capabilities:``), not by
import side effects. The re-export below is a convenience for
``from cuvis_ai_patchcore import PatchCoreDetector``.
"""

from cuvis_ai_patchcore.node.patchcore import PatchCoreDetector

__all__ = ["PatchCoreDetector"]
