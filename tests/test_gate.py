"""FrameScoreGate: gating logic, frame score, mask mode, port contract, hparam validation."""

from __future__ import annotations

import pytest
import torch

from cuvis_ai_patchcore.node.gate import FrameScoreGate

pytestmark = pytest.mark.unit


def _scores() -> torch.Tensor:
    # [2, 4, 4, 1]: frame 0 all 0.1 (clean); frame 1 has 4 hot pixels = 5.0, rest 0.1.
    x = torch.full((2, 4, 4, 1), 0.1, dtype=torch.float32)
    x[1, 0, :, 0] = 5.0  # 4 pixels of 16 -> matches topk_frac=0.25 (k=4)
    return x


def test_gate_heatmap_passthrough_and_blank():
    node = FrameScoreGate(threshold=1.0, topk_frac=0.25, mode="heatmap")
    out = node(scores=_scores())
    # frame 0: top-4 mean = 0.1 <= 1.0 -> blanked; frame 1: top-4 mean = 5.0 > 1.0 -> passed through
    assert torch.equal(out["scores"][0], torch.zeros(4, 4, 1))
    assert torch.equal(out["scores"][1], _scores()[1])
    assert torch.allclose(out["frame_score"], torch.tensor([0.1, 5.0]), atol=1e-6)
    assert torch.equal(out["passed"], torch.tensor([0, 1], dtype=torch.int32))


def test_gate_mask_mode_binarises_passing_frame():
    node = FrameScoreGate(threshold=1.0, topk_frac=0.25, mode="mask")
    out = node(scores=_scores())
    assert torch.equal(out["scores"][0], torch.zeros(4, 4, 1))  # clean frame blank
    expected = (_scores()[1] > 1.0).float()  # the 4 hot pixels -> 1.0, rest 0.0
    assert torch.equal(out["scores"][1], expected)
    assert float(out["scores"][1].sum()) == 4.0


def test_alarm_scores_drives_gate_of_a_different_display_map():
    # Display map is low-amplitude on both frames (looks clean); the alarm map is hot on frame 1
    # only. The gate must pass the DISPLAY map through on frame 1 (alarm fired) and blank frame 0.
    display = torch.full((2, 4, 4, 1), 0.3, dtype=torch.float32)
    display[1, 2, :, 0] = 0.4  # low-amplitude localisation, never above `threshold` on its own
    alarm = torch.full((2, 4, 4, 1), 0.1, dtype=torch.float32)
    alarm[1, 0, :, 0] = 5.0  # frame 1 alarm top-4 mean = 5.0
    node = FrameScoreGate(threshold=1.0, topk_frac=0.25, mode="heatmap")
    out = node(scores=display, alarm_scores=alarm)
    # frame 0: alarm 0.1 <= 1.0 -> blanked; frame 1: alarm fired -> display passed through as-is
    assert torch.equal(out["scores"][0], torch.zeros(4, 4, 1))
    assert torch.equal(out["scores"][1], display[1])
    # the frame score comes from the alarm map
    assert torch.allclose(out["frame_score"], torch.tensor([0.1, 5.0]), atol=1e-6)
    assert torch.equal(out["passed"], torch.tensor([0, 1], dtype=torch.int32))


def test_alarm_scores_batch_mismatch_raises():
    node = FrameScoreGate(threshold=1.0)
    with pytest.raises(RuntimeError):
        node(scores=torch.rand(2, 4, 4, 1), alarm_scores=torch.rand(3, 4, 4, 1))


def test_mask_threshold_binarises_display_on_its_own_scale():
    # Alarm on a hot map; the display map is on another scale, binarised at mask_threshold
    # (not at `threshold`).
    display = torch.full((1, 4, 4, 1), 0.2, dtype=torch.float32)
    display[0, 1, :, 0] = 0.8  # 4 pixels above 0.5
    alarm = torch.full((1, 4, 4, 1), 5.0, dtype=torch.float32)  # alarm always fires
    node = FrameScoreGate(threshold=1.0, topk_frac=0.25, mode="mask", mask_threshold=0.5)
    out = node(scores=display, alarm_scores=alarm)
    assert float(out["scores"].sum()) == 4.0  # only the 0.8 pixels cross 0.5
    assert node.hparams["mask_threshold"] == 0.5


def test_port_contract():
    node = FrameScoreGate(threshold=0.5)
    out = node(scores=torch.rand(3, 8, 8, 1, dtype=torch.float32))
    assert out["scores"].shape == (3, 8, 8, 1) and out["scores"].dtype == torch.float32
    assert out["frame_score"].shape == (3,) and out["frame_score"].dtype == torch.float32
    assert out["passed"].shape == (3,) and out["passed"].dtype == torch.int32


@pytest.mark.parametrize(
    "kw",
    [
        {"threshold": "x"},
        {"threshold": 1.0, "topk_frac": 0.0},
        {"threshold": 1.0, "mode": "binary"},
    ],
)
def test_invalid_hparams_raise(kw):
    with pytest.raises(ValueError):
        FrameScoreGate(**kw)


def test_hparams_round_trip():
    node = FrameScoreGate(threshold=2.5, topk_frac=0.002, mode="mask")
    assert node.hparams["threshold"] == 2.5
    assert node.hparams["topk_frac"] == 0.002
    assert node.hparams["mode"] == "mask"


def test_smooth_k_gates_on_the_rolling_median_one_frame_at_a_time():
    # Streaming: one frame per forward. An isolated spike (frame 2) is suppressed by the median of
    # the last 3 scores; a sustained anomaly (frames 4-5) passes once it holds the majority.
    node = FrameScoreGate(threshold=1.0, topk_frac=1.0, smooth_k=3)
    passed, raw = [], []
    for v in (0.1, 0.1, 5.0, 0.1, 5.0, 5.0):
        out = node(scores=torch.full((1, 2, 2, 1), v))
        passed.append(int(out["passed"]))
        raw.append(float(out["frame_score"]))
    assert passed == [0, 0, 0, 0, 1, 1]
    assert raw == pytest.approx([0.1, 0.1, 5.0, 0.1, 5.0, 5.0])  # frame_score stays unsmoothed


def test_log_scores_logs_every_frame_decision():
    from loguru import logger

    lines: list[str] = []
    sink = logger.add(lines.append, level="INFO", format="{message}")
    try:
        node = FrameScoreGate(threshold=1.0, topk_frac=0.25, log_scores=True, name="gate")
        node(scores=_scores())
    finally:
        logger.remove(sink)
    assert len(lines) == 2
    assert "[gate] frame_score=0.1000 threshold=1.0000 passed=0" in lines[0]
    assert "[gate] frame_score=5.0000 threshold=1.0000 passed=1" in lines[1]


@pytest.mark.parametrize("kw", [{"smooth_k": 0}, {"smooth_k": 2.5}, {"mask_threshold": "x"}])
def test_invalid_smoothing_and_mask_hparams_raise(kw):
    with pytest.raises(ValueError):
        FrameScoreGate(threshold=1.0, **kw)
