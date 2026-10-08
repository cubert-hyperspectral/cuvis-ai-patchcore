"""The weight declarations: every row constructs, is registered at import, and is consistent."""

import re

import pytest
from cuvis_ai_core.data.model_weights import USED_FOR_LABELS, ModelWeights

import cuvis_ai_patchcore  # noqa: F401  (registers the rows)
from cuvis_ai_patchcore.weights import _WALNUT_REPO, _WALNUT_REVISION, PLUGIN_NAME, WEIGHTS

pytestmark = pytest.mark.unit

SHELL_WEIGHTS = {"weights/rgb_v2_ema.pth", "weights/cir_v2_ema.pth"}


def test_rows_are_trained_pipelines_with_their_files():
    assert len(WEIGHTS) == 6
    for row in WEIGHTS:
        assert row.kind == "trained_pipeline"
        assert row.filename == f"{row.name}/{row.name}.pt"
        aux = {a.path for a in row.aux_files}
        assert f"{row.name}/{row.name}.yaml" in aux
        assert SHELL_WEIGHTS <= aux
        assert len(row.summary) <= 60
        assert set(row.used_for) <= set(USED_FOR_LABELS)
        assert re.fullmatch(r"[0-9a-f]{40}", row.revision)
        assert re.fullmatch(r"[0-9a-f]{64}", row.sha256)


def test_names_are_unique_and_registered():
    names = [row.name for row in WEIGHTS]
    assert len(set(names)) == len(names)
    registered = {row.name for row in ModelWeights.rows() if row.plugin == PLUGIN_NAME}
    assert set(names) <= registered


def test_rows_share_the_published_repo_and_revision():
    assert {row.repo_id for row in WEIGHTS} == {_WALNUT_REPO}
    assert {row.revision for row in WEIGHTS} == {_WALNUT_REVISION}
