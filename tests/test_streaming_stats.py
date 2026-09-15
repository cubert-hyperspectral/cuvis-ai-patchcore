"""StreamingMeanVar: batch-merged statistics equal the closed-form mean / unbiased variance."""

from __future__ import annotations

import pytest
import torch

from cuvis_ai_patchcore.streaming_stats import StreamingMeanVar

pytestmark = pytest.mark.unit


def test_matches_closed_form_over_uneven_batches():
    torch.manual_seed(0)
    x = torch.randn(1000, 6) * torch.tensor([1.0, 2.0, 0.5, 10.0, 0.01, 3.0]) + 5.0
    acc = StreamingMeanVar(6)
    for chunk in torch.split(x, [1, 7, 300, 692]):
        acc.update(chunk)
    assert acc.count == 1000
    assert torch.allclose(acc.mean, x.mean(dim=0), atol=1e-5)
    assert torch.allclose(acc.var, x.var(dim=0, unbiased=True), rtol=1e-5, atol=1e-6)


def test_empty_batch_is_ignored_and_var_zero_below_two_samples():
    acc = StreamingMeanVar(3)
    acc.update(torch.empty(0, 3))
    assert acc.count == 0
    acc.update(torch.tensor([[1.0, 2.0, 3.0]]))
    assert acc.count == 1
    assert torch.equal(acc.var, torch.zeros(3))
    assert torch.allclose(acc.mean, torch.tensor([1.0, 2.0, 3.0]))


def test_rejects_wrong_width():
    acc = StreamingMeanVar(3)
    with pytest.raises(ValueError):
        acc.update(torch.rand(4, 2))
    with pytest.raises(ValueError):
        StreamingMeanVar(0)
