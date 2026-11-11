"""Losses for independent CUAD category predictions with partial-clause masks."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def _select_valid(logits: Tensor, labels: Tensor, label_mask: Tensor) -> tuple[Tensor, Tensor]:
    """Check inputs and return (logits, targets) for valid entries, as flat tensors.

    Select entries before calculating a loss so even extreme/nonfinite masked
    logits cannot contaminate it. Half-precision inputs are calculated in float32
    for stable use with mixed-precision training.
    """
    if logits.ndim != 2 or labels.shape != logits.shape or label_mask.shape != logits.shape:
        raise ValueError("logits, labels, and label_mask must have matching 2D shapes")
    if not logits.is_floating_point():
        raise ValueError("logits must be floating point")
    if label_mask.dtype != torch.bool:
        raise ValueError("label_mask must be bool (True means include)")
    if labels.device != logits.device or label_mask.device != logits.device:
        raise ValueError("All loss inputs must be on the same device")
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("labels must be binary")

    selected = logits[label_mask]
    if selected.dtype in (torch.float16, torch.bfloat16):
        selected = selected.float()
    return selected, labels[label_mask].to(dtype=selected.dtype)


def masked_bce_with_logits(
    logits: Tensor, labels: Tensor, label_mask: Tensor, pos_weight: Tensor | None = None
) -> Tensor:
    """Mean binary cross entropy over valid (chunk, category) pairs only.

    All inputs have shape (batch, categories). ``logits`` are raw model outputs;
    labels are binary and ``label_mask`` is bool, with True meaning include.
    The denominator is the number of valid entries, not the full tensor size.
    An entirely masked or empty batch returns a differentiable zero.

    ``pos_weight`` optionally holds one multiplier per category for positive
    entries. Without it this is plain BCE.
    """
    selected, targets = _select_valid(logits, labels, label_mask)
    if selected.numel() == 0:
        return selected.sum()
    weights = None
    if pos_weight is not None:
        if pos_weight.shape != logits.shape[1:]:
            raise ValueError("pos_weight must have one value per category")
        weights = pos_weight.to(selected).expand_as(logits)[label_mask]
    return F.binary_cross_entropy_with_logits(
        selected, targets, pos_weight=weights, reduction="mean"
    )


def masked_focal_with_logits(
    logits: Tensor,
    labels: Tensor,
    label_mask: Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> Tensor:
    """Mean binary focal loss (Lin et al. 2017) over valid entries only.

    Each entry's BCE is multiplied by ``(1 - p_t) ** gamma``, so entries the model
    already gets right count for little, and by ``alpha`` for positives or
    ``1 - alpha`` for negatives. Masking and the denominator match
    masked_bce_with_logits. ``gamma=0, alpha=0.5`` is half of plain BCE.
    """
    if gamma < 0 or not 0 <= alpha <= 1:
        raise ValueError("gamma must be nonnegative and alpha must be in [0, 1]")
    selected, targets = _select_valid(logits, labels, label_mask)
    if selected.numel() == 0:
        return selected.sum()
    bce = F.binary_cross_entropy_with_logits(selected, targets, reduction="none")
    p_correct = torch.exp(-bce)
    balance = alpha * targets + (1 - alpha) * (1 - targets)
    return (balance * (1 - p_correct) ** gamma * bce).mean()
