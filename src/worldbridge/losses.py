"""Validity-masked coordinate losses; visibility is only an analysis/input signal."""
from __future__ import annotations
import torch


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, beta: float = 0.05) -> torch.Tensor:
    mask = valid.to(pred.dtype)
    error = torch.nn.functional.smooth_l1_loss(pred, target, beta=beta, reduction="none")
    while mask.ndim < error.ndim:
        mask = mask.unsqueeze(-1)
    denom = mask.sum().clamp_min(1.0)
    return (error * mask).sum() / (denom * error.shape[-1])


def balanced_query_loss(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, groups: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """Average group losses equally so diagonal queries cannot dominate."""
    losses = []
    stats = {}
    for name, code in (("reconstruction", 0), ("first_frame_tracking", 1), ("arbitrary_source_tracking", 2)):
        sel = groups == code
        if sel.any():
            value = masked_smooth_l1(pred[sel], target[sel], valid[sel])
            losses.append(value)
            stats[name] = float(value.detach())
    if not losses:
        raise ValueError("no query groups")
    return torch.stack(losses).mean(), stats
