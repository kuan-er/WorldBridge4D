"""Distributed initialization, wrapping, and collective gradient clipping."""
from __future__ import annotations

from contextlib import contextmanager
import datetime as dt
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullOptimStateDictConfig, FullStateDictConfig, FullyShardedDataParallel as FSDP,
    StateDictType,
)

def initialize_distributed() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    if not torch.cuda.is_available():
        raise RuntimeError("256px distributed training requires CUDA")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group(
        "nccl", init_method="env://", timeout=dt.timedelta(hours=24), device_id=device,
    )
    return rank, world, local, device

def clip_optimizer_grad_norm_(
    optimizer: torch.optim.Optimizer, max_norm: float, device: torch.device,
) -> torch.Tensor:
    """Collectively clip optimizer-owned sharded gradients on every rank.

    With ``use_orig_params=True`` a small trainable suffix can reside entirely
    on one rank while another rank has no local gradients. FSDP's convenience
    clipper returns early on the empty rank, which leaves the owner blocked in
    its collective. This helper always participates in one all-reduce.
    """
    max_norm = float(max_norm)
    if max_norm <= 0:
        raise ValueError("gradient clip norm must be positive")
    local_squared = torch.zeros((), device=device, dtype=torch.float64)
    gradients = []
    seen: set[int] = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            gradient = parameter.grad
            if gradient is None:
                continue
            gradients.append(gradient)
            local_squared += gradient.detach().double().square().sum()
    dist.all_reduce(local_squared, op=dist.ReduceOp.SUM)
    total_norm = local_squared.sqrt()
    coefficient = torch.clamp(
        torch.as_tensor(max_norm, device=device, dtype=torch.float64)
        / (total_norm + 1e-6),
        max=1.0,
    )
    for gradient in gradients:
        gradient.mul_(coefficient.to(dtype=gradient.dtype))
    return total_norm.float()

def wan_block_auto_wrap_policy(
    module: torch.nn.Module, recurse: bool, nonwrapped_numel: int,
) -> bool:
    """Shard Wan blocks, but never a parent whose custom forward is bypassed.

    Structured hidden extraction calls ``dit.patch_embedding`` and individual
    blocks directly instead of calling ``dit.forward``. Wrapping the parent
    Wan transformer would leave its flat parameters sharded during those direct
    calls. Each transformer block is invoked normally, so it is the safe FSDP
    unit for the LoRA route.
    """
    if recurse:
        return True
    return module.__class__.__name__ == "WanTransformerBlock"

def fsdp_state_context(model: FSDP):
    return FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    )
