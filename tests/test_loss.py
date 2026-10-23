import math

import pytest
import torch
from torch.nn import functional as F

from src.loss import masked_bce_with_logits


def test_known_loss_normalizes_by_valid_entries():
    logits = torch.tensor([[0.0, 50.0], [math.log(3), -50.0]], dtype=torch.float64)
    labels = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    mask = torch.tensor([[True, False], [True, False]])
    actual = masked_bce_with_logits(logits, labels, mask)
    expected = (math.log(2) - math.log(0.75)) / 2
    assert actual.item() == pytest.approx(expected)


def test_all_valid_matches_standard_bce():
    logits = torch.tensor([[2.0, -1.0, 0.0], [-2.0, 3.0, 1.0]])
    labels = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    mask = torch.ones_like(labels, dtype=torch.bool)
    torch.testing.assert_close(
        masked_bce_with_logits(logits, labels, mask),
        F.binary_cross_entropy_with_logits(logits, labels),
    )


def test_masked_values_cannot_change_loss_or_gradients():
    labels = torch.tensor([[1.0, 0.0, 1.0]])
    mask = torch.tensor([[True, False, True]])
    baseline = None
    for ignored in (0.0, 1e30, float("nan"), float("inf")):
        logits = torch.tensor([[0.0, ignored, 0.0]], requires_grad=True)
        loss = masked_bce_with_logits(logits, labels, mask)
        loss.backward()
        assert torch.isfinite(loss)
        torch.testing.assert_close(logits.grad, torch.tensor([[-0.25, 0.0, -0.25]]))
        if baseline is None:
            baseline = loss.detach()
        torch.testing.assert_close(loss, baseline)


@pytest.mark.parametrize("shape", [(2, 10), (0, 10)])
def test_all_masked_or_empty_returns_differentiable_zero(shape):
    logits = torch.full(shape, float("nan"), requires_grad=True)
    loss = masked_bce_with_logits(
        logits, torch.zeros(shape), torch.zeros(shape, dtype=torch.bool)
    )
    assert loss.item() == 0
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros(shape))


def test_extreme_valid_logits_are_stable():
    logits = torch.tensor([[1000.0, -1000.0]], requires_grad=True)
    loss = masked_bce_with_logits(
        logits, torch.tensor([[0.0, 1.0]]), torch.tensor([[True, True]])
    )
    assert loss.item() == pytest.approx(1000)
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[0.5, -0.5]]))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_loss_uses_float32(dtype):
    logits = torch.zeros((1, 10), dtype=dtype, requires_grad=True)
    loss = masked_bce_with_logits(
        logits, torch.ones((1, 10)), torch.ones((1, 10), dtype=torch.bool)
    )
    assert loss.dtype == torch.float32
    assert loss.item() == pytest.approx(math.log(2))
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_optimizer_step_reduces_loss_and_leaves_masked_parameter_unchanged():
    logits = torch.nn.Parameter(torch.zeros(1, 3))
    optimizer = torch.optim.SGD([logits], lr=0.5)
    labels = torch.tensor([[1.0, 0.0, 1.0]])
    mask = torch.tensor([[True, True, False]])
    before = masked_bce_with_logits(logits, labels, mask)
    before.backward()
    optimizer.step()
    assert masked_bce_with_logits(logits, labels, mask) < before
    assert logits[0, 2].item() == 0


@pytest.mark.parametrize("case", ["shape", "rank", "mask", "labels", "logits"])
def test_invalid_inputs_raise(case):
    logits = torch.zeros(2, 10)
    labels = torch.zeros(2, 10)
    mask = torch.ones(2, 10, dtype=torch.bool)
    if case == "shape":
        labels = labels[:1]
    elif case == "rank":
        logits, labels, mask = logits[0], labels[0], mask[0]
    elif case == "mask":
        mask = mask.float()
    elif case == "labels":
        labels[0, 0] = 0.5
    else:
        logits = logits.long()
    with pytest.raises(ValueError):
        masked_bce_with_logits(logits, labels, mask)
