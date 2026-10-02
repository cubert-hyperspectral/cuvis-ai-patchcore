"""MaskMinArea: golden rule against scipy's 8-connected labelling, channels, frames, port
contract, hparams."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from scipy import ndimage

from cuvis_ai_patchcore.node.morphology import MaskMinArea

pytestmark = pytest.mark.unit

H, W = 20, 24


def _reference(m: np.ndarray, n: int) -> np.ndarray:
    """Blobs of a 2-D mask with >= n pixels, 8-connected, by scipy."""
    lab, nb = ndimage.label(m, structure=np.ones((3, 3)))
    if not nb:
        return np.zeros_like(m)
    area = ndimage.sum(np.ones_like(m), lab, index=np.arange(1, nb + 1))
    keep = np.zeros(nb + 1, bool)
    keep[1:] = area >= n
    return keep[lab]


def test_drops_the_small_blobs_and_keeps_the_large_ones():
    m = torch.zeros(1, H, W, 1, dtype=torch.bool)
    m[0, 1:3, 1:3, 0] = True  # 4 px
    m[0, 10:15, 10:16, 0] = True  # 30 px
    out = MaskMinArea(min_area=10)(decisions=m)["decisions"]
    exp = torch.zeros_like(m)
    exp[0, 10:15, 10:16, 0] = True
    assert torch.equal(out, exp)


def test_diagonal_neighbours_belong_to_one_blob():
    m = torch.zeros(1, H, W, 1, dtype=torch.bool)
    for k in range(6):  # a diagonal line of 6 px: one 8-connected blob
        m[0, 2 + k, 2 + k, 0] = True
    assert torch.equal(MaskMinArea(min_area=6)(decisions=m)["decisions"], m)
    assert not MaskMinArea(min_area=7)(decisions=m)["decisions"].any()


@pytest.mark.parametrize("n", [0, 1, 2, 5, 17, 60])
def test_matches_scipy_on_random_masks(n):
    g = torch.Generator().manual_seed(n)
    m = torch.rand(3, H, W, 2, generator=g) < 0.35
    out = MaskMinArea(min_area=n)(decisions=m)["decisions"].numpy()
    for b in range(3):
        for c in range(2):
            exp = m[b, :, :, c].numpy() if n <= 1 else _reference(m[b, :, :, c].numpy(), n)
            assert np.array_equal(out[b, :, :, c], exp), (b, c)


def test_frames_and_channels_are_filtered_on_their_own():
    m = torch.zeros(2, H, W, 2, dtype=torch.bool)
    m[0, 0:4, 0:4, 0] = True  # 16 px in frame 0, channel 0
    m[1, 0:2, 0:2, 1] = True  # 4 px in frame 1, channel 1
    out = MaskMinArea(min_area=10)(decisions=m)["decisions"]
    assert out[0, :, :, 0].sum() == 16 and not out[1].any() and not out[0, :, :, 1].any()


def test_does_not_mutate_its_input_and_handles_empty_masks():
    m = torch.zeros(1, H, W, 1, dtype=torch.bool)
    m[0, 1:3, 1:3, 0] = True
    before = m.clone()
    MaskMinArea(min_area=10)(decisions=m)
    assert torch.equal(m, before)
    empty = MaskMinArea()(decisions=torch.zeros(2, H, W, 1, dtype=torch.bool))["decisions"]
    assert empty.shape == (2, H, W, 1) and not empty.any()
    none = MaskMinArea()(decisions=torch.zeros(0, H, W, 1, dtype=torch.bool))["decisions"]
    assert none.shape == (0, H, W, 1)


def test_holds_no_serialized_state():
    assert len(MaskMinArea().state_dict()) == 0


def test_port_contract():
    m = torch.rand(2, H, W, 1, generator=torch.Generator().manual_seed(3)) < 0.5
    out = MaskMinArea()(decisions=m)
    spec = MaskMinArea.OUTPUT_SPECS["decisions"]
    assert set(out) == {"decisions"}
    assert out["decisions"].shape == m.shape and out["decisions"].dtype == spec.dtype


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_input_comes_back_on_cuda_and_equal():
    m = torch.rand(2, H, W, 1, generator=torch.Generator().manual_seed(5)) < 0.4
    cpu = MaskMinArea(min_area=8)(decisions=m)["decisions"]
    gpu = MaskMinArea(min_area=8)(decisions=m.cuda())["decisions"]
    assert gpu.is_cuda and torch.equal(gpu.cpu(), cpu)


@pytest.mark.parametrize("bad", [-1, 2.5, True, "250", None])
def test_invalid_min_area_raises(bad):
    with pytest.raises(ValueError):
        MaskMinArea(min_area=bad)


def test_hparams_round_trip_json():
    hp = MaskMinArea(min_area=100, name="speck_filter").hparams
    assert hp["min_area"] == 100
    json.dumps(hp)
    assert MaskMinArea().hparams["min_area"] == 250
