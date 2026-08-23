"""Evaluation-only metrics and paired uncertainty estimates."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def align_sim3_to_ground_truth(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Fit one proper Sim(3) from a predicted trajectory to dataset GT.

    A single scale/rotation/translation is fitted jointly over every requested
    target and valid source-grid point. Using one transform preserves temporal
    consistency; fitting one transform per target would not.
    """
    if prediction.ndim != 4 or prediction.shape[1] != 3:
        raise ValueError(f"prediction must be [K,3,H,W], got {tuple(prediction.shape)}")
    if target.shape != prediction.shape:
        raise ValueError(
            f"target shape {tuple(target.shape)} does not match prediction {tuple(prediction.shape)}"
        )
    if valid.shape != (prediction.shape[0], prediction.shape[2], prediction.shape[3]):
        raise ValueError(f"valid must be [K,H,W], got {tuple(valid.shape)}")

    finite = torch.isfinite(prediction).all(dim=1) & torch.isfinite(target).all(dim=1)
    mask = valid.bool() & finite
    count = int(mask.sum())
    if count < 3:
        raise RuntimeError(f"Sim(3) alignment requires at least 3 valid points, got {count}")
    x = prediction.permute(0, 2, 3, 1)[mask].double()
    y = target.permute(0, 2, 3, 1)[mask].double()
    mean_x = x.mean(dim=0)
    mean_y = y.mean(dim=0)
    centered_x = x - mean_x
    centered_y = y - mean_y
    variance_x = centered_x.square().sum() / count
    if not torch.isfinite(variance_x) or float(variance_x) <= 1e-12:
        raise RuntimeError("Sim(3) alignment is degenerate: prediction variance is zero")
    covariance = centered_y.T @ centered_x / count
    u, singular, vh = torch.linalg.svd(covariance)
    correction = torch.ones(3, dtype=torch.float64, device=x.device)
    if torch.det(u @ vh) < 0:
        correction[-1] = -1.0
    rotation = u @ torch.diag(correction) @ vh
    scale = (singular * correction).sum() / variance_x
    if not torch.isfinite(scale) or float(scale) <= 0:
        raise RuntimeError(f"Sim(3) alignment produced invalid scale {float(scale)}")
    translation = mean_y - scale * (rotation @ mean_x)

    aligned = scale.to(prediction.dtype) * torch.einsum(
        "ij,kjhw->kihw", rotation.to(prediction.dtype), prediction
    ) + translation.to(prediction.dtype).reshape(1, 3, 1, 1)
    before = torch.linalg.vector_norm(x - y, dim=1)
    aligned_valid = scale * (x @ rotation.T) + translation
    after = torch.linalg.vector_norm(aligned_valid - y, dim=1)
    metadata: dict[str, object] = {
        "enabled": True,
        "method": "joint_proper_umeyama_sim3_prediction_to_dataset_gt",
        "points": count,
        "scale": float(scale),
        "rotation": rotation.cpu().tolist(),
        "translation": translation.cpu().tolist(),
        "rmse_before_m": float(torch.sqrt(before.square().mean())),
        "rmse_after_m": float(torch.sqrt(after.square().mean())),
    }
    return aligned, metadata


def paired_bootstrap_ci(values: list[float], seed: int,
                        draws: int = 10_000) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        raise ValueError("paired bootstrap requires at least two samples")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, array.size, size=(int(draws), array.size))
    means = array[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]

def masked_metrics(prediction: torch.Tensor, target: torch.Tensor,
                   valid: torch.Tensor, scale: torch.Tensor,
                   beta: float) -> tuple[float, float, int]:
    mask = valid[:, None]
    valid_points = int(valid.sum())
    if valid_points < 1:
        raise RuntimeError("evaluation sample has no valid supervised point")
    loss = F.smooth_l1_loss(prediction, target, beta=beta, reduction="none")
    loss_sum = float((loss * mask).sum())
    metric_error = (prediction - target) * scale
    epe_sum = float((torch.linalg.vector_norm(metric_error, dim=1) * valid).sum())
    return loss_sum, epe_sum, valid_points
