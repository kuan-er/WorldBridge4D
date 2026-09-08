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

def initialize_distributed(timeout_seconds: float = 86400) -> tuple[int, int, int, torch.device]:
    if not 0 < float(timeout_seconds) < float("inf"):
        raise ValueError("distributed timeout must be finite and positive")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    if not torch.cuda.is_available():
        raise RuntimeError("256px distributed training requires CUDA")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group(
        "nccl", init_method="env://", timeout=dt.timedelta(seconds=timeout_seconds), device_id=device,
    )
    return rank, world, local, device

def fsdp_state_context(model: FSDP):
    return FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    )
