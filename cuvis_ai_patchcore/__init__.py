"""cuvis_ai_patchcore — HSI-PatchCore anomaly detection for cuvis-ai.

Plugin nodes are discovered through the ``plugins.yaml`` manifest (``capabilities:``), not by
import side effects; importing the package registers the plugin's weight declarations
(the published walnut pipelines, ``weights.py``) with cuvis-ai-core's ``ModelWeights``. The
re-export below is a convenience for ``from cuvis_ai_patchcore import PatchCoreDetector``.
"""

from cuvis_ai_core.data.model_weights import ModelWeights

from cuvis_ai_patchcore.node.patchcore import PatchCoreDetector
from cuvis_ai_patchcore.weights import PLUGIN_NAME, WEIGHTS

ModelWeights.register(PLUGIN_NAME, WEIGHTS)

__all__ = ["PLUGIN_NAME", "WEIGHTS", "PatchCoreDetector"]
