"""MaskPersistence: golden rule against a brute-force window search, frame-to-frame state, batch
order, reset, port contract, hparam validation."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from cuvis_ai_patchcore.node.temporal import MaskPersistence

pytestmark = pytest.mark.unit

H, W = 12, 15


def _frame(*pixels: tuple[int, int], c: int = 1, ch: int = 0) -> torch.Tensor:
    """A [1, H, W, c] mask with the given (row, col) pixels set in channel ``ch``."""
    m = torch.zeros(1, H, W, c, dtype=torch.bool)
    for y, x in pixels:
        m[0, y, x, ch] = True
    return m


def _random_sequence(n: int, c: int = 1, density: float = 0.08, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, H, W, c, generator=g) < density


def _brute_force(prev: np.ndarray, cur: np.ndarray, r: int) -> np.ndarray:
    """``cur`` & (a pixel of ``prev``, same channel, within the square window of radius ``r``)."""
    out = np.zeros_like(cur)
    h, w, c = cur.shape
    for y in range(h):
        for x in range(w):
            for k in range(c):
                win = prev[max(0, y - r) : y + r + 1, max(0, x - r) : x + r + 1, k]
                out[y, x, k] = cur[y, x, k] and win.any()
    return out


def _expected(seq: torch.Tensor, r: int) -> torch.Tensor:
    """The rule over a whole sequence [N, H, W, C]: nothing on frame 0, then frame by frame."""
    s = seq.numpy()
    out = [np.zeros_like(s[0])] + [_brute_force(s[j - 1], s[j], r) for j in range(1, len(s))]
    return torch.from_numpy(np.stack(out))


# ----- 1. golden rule -------------------------------------------------------------------------


def test_keeps_the_pixels_within_the_square_window_of_the_previous_mask():
    node = MaskPersistence(radius_px=2)
    node(decisions=_frame((5, 5)))
    cur = _frame((5, 7), (5, 8), (7, 7), (8, 5), (3, 3), (2, 5))
    out = node(decisions=cur)["decisions"]
    # 2 px away in a row, a column or both (the window is a square, not a disc) are kept; 3 px not
    assert torch.equal(out, _frame((5, 7), (7, 7), (3, 3)))


@pytest.mark.parametrize("radius", [0, 1, 3, 7, 20])
@pytest.mark.parametrize("channels", [1, 2])
def test_matches_a_brute_force_window_search(radius, channels):
    seq = _random_sequence(6, c=channels, seed=radius + 10 * channels)
    node = MaskPersistence(radius_px=radius)
    out = torch.cat([node(decisions=seq[j : j + 1])["decisions"] for j in range(len(seq))])
    assert torch.equal(out, _expected(seq, radius))


def test_the_first_frame_shows_nothing():
    out = MaskPersistence()(decisions=torch.ones(1, H, W, 1, dtype=torch.bool))["decisions"]
    assert not out.any()


def test_a_moving_object_shows_from_its_second_frame_and_a_flicker_never():
    node = MaskPersistence(radius_px=4)
    for j in range(5):
        blob = [(4 + dy, 1 + 3 * j + dx) for dy in (0, 1) for dx in (0, 1)]  # 3 px per frame
        flicker = [(10, 1)] if j == 2 else []
        out = node(decisions=_frame(*blob, *flicker))["decisions"]
        assert torch.equal(out, _frame(*blob) if j else _frame())


def test_radius_zero_keeps_the_pixels_set_in_both_frames():
    a, b = _random_sequence(2, density=0.4, seed=3)
    node = MaskPersistence(radius_px=0)
    node(decisions=a[None])
    assert torch.equal(node(decisions=b[None])["decisions"], (a & b)[None])


def test_channels_are_filtered_on_their_own():
    node = MaskPersistence(radius_px=3)
    node(decisions=_frame((5, 5), c=2, ch=0))
    cur = _frame((5, 5), c=2, ch=0) | _frame((5, 5), c=2, ch=1)
    assert torch.equal(node(decisions=cur)["decisions"], _frame((5, 5), c=2, ch=0))


def test_the_image_border_adds_nothing():
    node = MaskPersistence(radius_px=2)
    node(decisions=torch.zeros(1, H, W, 1, dtype=torch.bool))
    assert not node(decisions=torch.ones(1, H, W, 1, dtype=torch.bool))["decisions"].any()
    # nor does the window wrap around to the opposite edge
    node(decisions=_frame((0, 0)))
    assert not node(decisions=_frame((0, W - 1), (H - 1, 0)))["decisions"].any()


# ----- 2. state across frames and calls -------------------------------------------------------


def test_a_batch_is_taken_in_order_and_the_state_carries_across_calls():
    seq = _random_sequence(6, seed=7)
    whole = MaskPersistence(radius_px=2)(decisions=seq)["decisions"]
    one = MaskPersistence(radius_px=2)
    singles = torch.cat([one(decisions=seq[j : j + 1])["decisions"] for j in range(6)])
    split = MaskPersistence(radius_px=2)
    parts = torch.cat(
        [split(decisions=seq[:2])["decisions"], split(decisions=seq[2:])["decisions"]]
    )
    assert torch.equal(whole, _expected(seq, 2))
    assert torch.equal(singles, whole) and torch.equal(parts, whole)


def test_reset_forgets_the_previous_frame():
    f = _frame((5, 5), (6, 6))
    node = MaskPersistence()
    node(decisions=f)
    node.reset()
    assert not node(decisions=f)["decisions"].any()
    assert torch.equal(node(decisions=f)["decisions"], f)


@pytest.mark.parametrize(
    "other",
    [torch.ones(1, H + 1, W, 1, dtype=torch.bool), torch.ones(1, H, W, 2, dtype=torch.bool)],
)
def test_a_change_of_mask_shape_starts_over(other):
    node = MaskPersistence()
    node(decisions=torch.ones(1, H, W, 1, dtype=torch.bool))
    assert not node(decisions=other)["decisions"].any()
    assert torch.equal(node(decisions=other)["decisions"], other)


def test_an_empty_batch_leaves_the_state_alone():
    f = _frame((5, 5))
    node = MaskPersistence()
    node(decisions=f)
    empty = node(decisions=torch.zeros(0, H, W, 1, dtype=torch.bool))["decisions"]
    assert empty.shape == (0, H, W, 1)
    assert torch.equal(node(decisions=f)["decisions"], f)


def test_does_not_mutate_or_keep_a_reference_to_its_input():
    seq = _random_sequence(2, density=0.3, seed=11)
    a, b = seq[0:1].clone(), seq[1:2].clone()
    node = MaskPersistence(radius_px=1)
    node(decisions=a)
    assert torch.equal(a, seq[0:1])
    a.zero_()  # an upstream reusing its buffer must not change the remembered frame
    assert torch.equal(node(decisions=b)["decisions"], _expected(seq, 1)[1:2])


def test_holds_no_serialized_state():
    node = MaskPersistence()
    node(decisions=_frame((5, 5)))
    assert len(node.state_dict()) == 0


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_matches_cpu_and_a_device_change_starts_over():
    seq = _random_sequence(5, c=2, seed=13)
    cpu = MaskPersistence(radius_px=3)(decisions=seq)["decisions"]
    node = MaskPersistence(radius_px=3)
    gpu = node(decisions=seq.cuda())["decisions"]
    assert gpu.is_cuda and torch.equal(gpu.cpu(), cpu)
    assert not node(decisions=seq[:1])["decisions"].any()  # back on the CPU: first frame again


# ----- 3. port contract and hparams -----------------------------------------------------------


def test_port_contract():
    seq = _random_sequence(3, c=2)
    out = MaskPersistence()(decisions=seq)
    spec = MaskPersistence.OUTPUT_SPECS["decisions"]
    assert set(out) == {"decisions"}
    assert out["decisions"].shape == seq.shape and out["decisions"].dtype == spec.dtype


@pytest.mark.parametrize("bad", [-1, 1.5, True, "40", None])
def test_invalid_radius_raises(bad):
    with pytest.raises(ValueError):
        MaskPersistence(radius_px=bad)


def test_hparams_round_trip_json():
    hp = MaskPersistence(radius_px=12, name="persist").hparams
    assert hp["radius_px"] == 12
    json.dumps(hp)
    assert MaskPersistence().hparams["radius_px"] == 40
