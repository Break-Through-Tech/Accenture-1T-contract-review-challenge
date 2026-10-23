"""Loss for independent CUAD category predictions with partial-clause masks."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def masked_bce_with_logits(
    logits: Tensor, labels: Tensor, label_mask: Tensor
) -> Tensor:
    """Mean binary cross entropy over valid (chunk, category) pairs only.

    All inputs have shape (batch, categories). ``logits`` are raw model outputs;
    labels are binary and ``label_mask`` is bool, with True meaning include.
    The denominator is the number of valid entries, not the full tensor size.
    An entirely masked or empty batch returns a differentiable zero.

    Select entries before calculating BCE so even extreme/nonfinite masked
    logits cannot contaminate the loss. Half-precision inputs are calculated
    in float32 for stable use with mixed-precision training.
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
    if selected.numel() == 0:
        return selected.sum()
    targets = labels[label_mask].to(dtype=selected.dtype)
    return F.binary_cross_entropy_with_logits(selected, targets, reduction="mean")
