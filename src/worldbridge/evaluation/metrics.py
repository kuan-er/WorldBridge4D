"""Evaluation-only metrics and paired uncertainty estimates."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

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
