"""Differentiable training objectives."""
from __future__ import annotations

import torch
import torch.nn.functional as F

def masked_pair_smooth_l1(prediction: torch.Tensor, target: torch.Tensor,
                          validity: torch.Tensor, beta: float = 0.05) -> torch.Tensor:
    """Exact pair-wise validity-masked SmoothL1 average from the H004 spec."""
    if prediction.shape != target.shape or prediction.ndim != 5 or prediction.shape[2] != 3:
        raise ValueError("prediction/target must match [B,K,3,H,W]")
    if validity.shape != prediction.shape[:2] + prediction.shape[-2:]:
        raise ValueError("validity must be [B,K,H,W]")
    error = F.smooth_l1_loss(prediction, target, beta=beta, reduction="none").sum(dim=2)
    mask = validity.to(dtype=error.dtype)
    valid_count = mask.sum(dim=(-2, -1))
    eligible = valid_count > 0
    if not bool(eligible.any()):
        raise ValueError("batch contains no pair with a valid XYZ target")
    per_pair = (error * mask).sum(dim=(-2, -1)) / valid_count.clamp_min(1.0)
    return per_pair[eligible].mean()
