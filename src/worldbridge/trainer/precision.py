"""Explicit master storage, separate from FSDP/autocast compute precision."""
from __future__ import annotations

import torch


def prepare_fsdp_master_parameters(model: torch.nn.Module, precision: str) -> None:
    """Call BEFORE model checkpoint loading and optimizer group construction.

    Convert the complete model to keep each FSDP flat parameter dtype uniform,
    including frozen parameters. FSDP still casts compute parameters to BF16.
    Preparing before load also preserves FP32 values on subsequent exact resume.
    """
    if precision == "model":
        return
    if precision != "fp32":
        raise ValueError("fsdp_master_precision must be model or fp32")
    model.float()


def assert_fp32_optimizer_storage(optimizer: torch.optim.Optimizer) -> None:
    for group in optimizer.param_groups:
        if any(p.dtype != torch.float32 for p in group["params"]):
            raise RuntimeError("FP32 master parameters were lost during FSDP/resume")
    for state in optimizer.state.values():
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if key in state and state[key].dtype != torch.float32:
                raise RuntimeError(f"Adam {key} must use FP32 master storage")
