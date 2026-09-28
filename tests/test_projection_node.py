"""``PatchCoreDetector(coreset_projection=...)``: validation, sampler plumbing, state layout."""

from __future__ import annotations

import pytest
import torch

from cuvis_ai_patchcore.node.patchcore import PatchCoreDetector

pytestmark = pytest.mark.unit

D = 5


def _grids(n: int = 3, seed: int = 7) -> list[dict[str, torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    return [{"cube": torch.rand(1, 12, 12, D, generator=g)} for _ in range(n)]


def _fit(**overrides) -> PatchCoreDetector:
    kw = {
        "input_channels": D,
        "coreset_size": 16,
        "stride": 1,
        "bank_stride": 1,
        "pool_size": 1,
        "standardize": False,
        "seed": 0,
    }
    kw.update(overrides)
    node = PatchCoreDetector(**kw)
    node.statistical_initialization(iter(_grids()))
    return node


def test_defaults_to_exact_selection():
    node = PatchCoreDetector(input_channels=D, coreset_size=8)
    assert node.coreset_projection is None
    assert node.projection_eps == pytest.approx(0.9)


@pytest.mark.parametrize(
    "bad",
    [{"coreset_projection": "pca"}, {"projection_eps": 1.0}, {"projection_eps": 0.0}],
)
def test_rejects_invalid_projection_arguments(bad):
    with pytest.raises(ValueError):
        PatchCoreDetector(input_channels=D, coreset_size=8, **bad)


def test_projection_changes_the_coreset_but_not_the_state_layout():
    exact, proj = _fit(), _fit(coreset_projection="sparse_random")
    assert exact.coreset.shape == proj.coreset.shape == (16, D)
    assert set(exact.state_dict()) == set(proj.state_dict())
    assert not torch.allclose(exact.coreset, proj.coreset)


def test_projected_selection_rows_are_genuine_bank_rows():
    """The projection only steers *which* rows are kept; the bank stores the original features."""
    proj = _fit(coreset_projection="sparse_random")
    bank = torch.cat([g["cube"].reshape(-1, D) for g in _grids()])
    is_bank_row = (proj.coreset[:, None, :] == bank[None, :, :]).all(dim=-1).any(dim=-1)
    assert bool(is_bank_row.all())
    assert proj.coreset.unique(dim=0).shape[0] == 16


def test_projected_fit_is_reproducible_and_scores():
    a, b = _fit(coreset_projection="sparse_random"), _fit(coreset_projection="sparse_random")
    assert torch.equal(a.coreset, b.coreset)
    out = a(cube=torch.rand(1, 12, 12, D))
    assert out["scores"].shape == (1, 12, 12, 1)
    assert torch.isfinite(out["scores"]).all()


def test_projection_hparams_survive_a_save_load_round_trip():
    src = _fit(coreset_projection="sparse_random", projection_eps=0.5)
    dst = PatchCoreDetector(
        input_channels=D,
        coreset_size=16,
        stride=1,
        bank_stride=1,
        pool_size=1,
        standardize=False,
        coreset_projection="sparse_random",
        projection_eps=0.5,
    )
    dst.load_state_dict(src.state_dict())
    x = torch.rand(1, 12, 12, D)
    assert torch.allclose(src(cube=x)["scores"], dst(cube=x)["scores"])
    assert dst.projection_eps == pytest.approx(0.5)
