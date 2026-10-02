"""SpectralObjectMask: golden rule against the numpy angle map of the walnut study, brightness
invariance, the belt vs a different spectrum, port contract, hparams."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
import torch

from cuvis_ai_patchcore.node.objectness import SpectralObjectMask

pytestmark = pytest.mark.unit

H, W, C = 40, 48, 12


def _belt_cube(seed: int = 0, b: int = 1) -> torch.Tensor:
    """A belt spectrum with a little noise everywhere, a few pixels of another material."""
    g = torch.Generator().manual_seed(seed)
    belt = torch.linspace(0.2, 0.6, C)
    cube = belt * (1 + 0.01 * torch.randn(b, H, W, C, generator=g))
    cube[:, 8:16, 8:16, :] = torch.linspace(0.6, 0.2, C)  # an object: the opposite slope
    return cube.to(torch.float32)


def _reference(cube: np.ndarray, stride: int, deg: float) -> tuple[np.ndarray, np.ndarray]:
    """The study's make_angle4 + object_mask: angle on cube[::s, ::s], nearest to H x W."""
    c4 = cube[::stride, ::stride].astype(np.float32)
    x = c4.reshape(-1, c4.shape[-1])
    ref = np.median(x, axis=0)
    cos = (x @ ref) / (np.linalg.norm(x, axis=1) * np.linalg.norm(ref) + 1e-6)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1))).reshape(c4.shape[:2])
    full = (
        cv2.resize(
            (ang > deg).astype(np.uint8),
            (cube.shape[1], cube.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        > 0
    )
    return ang, full


@pytest.mark.parametrize("stride", [1, 2, 4])
def test_matches_the_numpy_angle_map(stride):
    cube = _belt_cube(seed=stride)
    out = SpectralObjectMask(min_angle_deg=6.0, stride=stride)(cube=cube)
    ang, full = _reference(cube[0].numpy(), stride, 6.0)
    # float32 arccos is ill-conditioned near 0 deg (the belt): a few thousandths of a degree
    assert np.allclose(out["angle"][0, ..., 0].numpy(), ang, atol=1e-2)
    assert np.array_equal(out["decisions"][0, ..., 0].numpy(), full)


def test_the_object_is_marked_and_the_belt_is_not():
    out = SpectralObjectMask(min_angle_deg=6.0, stride=1)(cube=_belt_cube())["decisions"][0, ..., 0]
    assert out[8:16, 8:16].all()
    rest = out.clone()
    rest[8:16, 8:16] = False
    assert not rest.any()


def test_brightness_does_not_make_an_object():
    cube = _belt_cube()
    cube[0, 25:35, 25:35, :] *= 0.3  # a shadow: the belt's spectrum, darker
    out = SpectralObjectMask(min_angle_deg=6.0, stride=1)(cube=cube)["decisions"][0, ..., 0]
    assert not out[25:35, 25:35].any()


def test_frames_of_a_batch_have_their_own_median():
    cube = _belt_cube(b=2)
    cube[1] = cube[1].flip(-1)  # frame 1: everything mirrored in the band axis
    out = SpectralObjectMask(stride=1)(cube=cube)["decisions"]
    assert out[0, 8:16, 8:16].all() and out[1, 8:16, 8:16].all()
    assert out[0].sum() == out[1].sum() == 64


def test_port_contract():
    cube = _belt_cube(b=2)
    node = SpectralObjectMask(stride=4)
    out = node(cube=cube)
    assert set(out) == {"decisions", "angle"}
    assert out["decisions"].shape == (2, H, W, 1)
    assert out["decisions"].dtype == SpectralObjectMask.OUTPUT_SPECS["decisions"].dtype
    assert out["angle"].shape == (2, H // 4, W // 4, 1) and out["angle"].dtype == torch.float32
    assert len(node.state_dict()) == 0


@pytest.mark.parametrize(
    "bad",
    [
        {"min_angle_deg": -1},
        {"min_angle_deg": 200},
        {"min_angle_deg": True},
        {"stride": 0},
        {"stride": 1.5},
        {"stride": True},
    ],
)
def test_invalid_hparams_raise(bad):
    with pytest.raises(ValueError):
        SpectralObjectMask(**bad)


def test_hparams_round_trip_json():
    hp = SpectralObjectMask(min_angle_deg=5, stride=2, name="objects").hparams
    assert hp["min_angle_deg"] == 5.0 and hp["stride"] == 2
    json.dumps(hp)
    d = SpectralObjectMask().hparams
    assert d["min_angle_deg"] == 6.0 and d["stride"] == 4
