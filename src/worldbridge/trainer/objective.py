"""Differentiable training objectives."""
from __future__ import annotations

import torch
import torch.nn.functional as F

def loss_scale_to_reference(
    reference_loss: torch.Tensor,
    auxiliary_loss: torch.Tensor,
    *,
    epsilon: float = 1e-6,
    max_scale: float = 1000.0,
) -> torch.Tensor:
    """Detached batch-local scale that matches an auxiliary loss to a reference."""
    if reference_loss.numel() != 1 or auxiliary_loss.numel() != 1:
        raise ValueError("reference and auxiliary losses must be scalars")
    if epsilon <= 0.0 or max_scale <= 0.0:
        raise ValueError("epsilon and max_scale must be positive")
    reference = reference_loss.detach()
    auxiliary = auxiliary_loss.detach()
    scale = reference / auxiliary.clamp_min(float(epsilon))
    scale = scale.clamp(min=0.0, max=float(max_scale))
    return torch.where(auxiliary > float(epsilon), scale, torch.zeros_like(scale))


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


def boundary_weighted_pair_smooth_l1(prediction: torch.Tensor, target: torch.Tensor,
                                    validity: torch.Tensor, boundary: torch.Tensor,
                                    multiplier: float = 2.0, beta: float = 0.05) -> torch.Tensor:
    """Same XYZ target/beta and equal eligible-pair reduction; only pixel weights differ.

    Boundary is GT source-only [B,H,W], shared across K. Normalize by the sum
    of valid weights within EACH pair. Do not drop occluded-but-valid points.
    """
    if prediction.shape != target.shape or prediction.ndim != 5 or prediction.shape[2] != 3:
        raise ValueError('prediction/target must match [B,K,3,H,W]')
    if validity.shape != prediction.shape[:2] + prediction.shape[-2:]:
        raise ValueError('validity must be [B,K,H,W]')
    if boundary.shape != (prediction.shape[0], *prediction.shape[-2:]) or boundary.dtype != torch.bool:
        raise ValueError('GT boundary must be bool [B,H,W]')
    if multiplier not in (1.0, 2.0):
        raise ValueError('audited boundary multiplier must be 1 or 2')
    if multiplier == 1.0:
        return masked_pair_smooth_l1(prediction, target, validity, beta)
    error = F.smooth_l1_loss(prediction, target, beta=beta, reduction='none').sum(dim=2)
    weights = 1.0 + (float(multiplier) - 1.0) * boundary.detach().to(error.dtype)[:, None]
    mask = validity.to(error.dtype) * weights
    weight_sum = mask.sum(dim=(-2, -1))
    eligible = weight_sum > 0
    if not bool(eligible.any()):
        raise ValueError('batch contains no pair with a valid XYZ target')
    per_pair = (error * mask).sum(dim=(-2, -1)) / weight_sum.clamp_min(1.0)
    return per_pair[eligible].mean()
