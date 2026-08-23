"""FSDP checkpoint publication and strict exact restore."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ..utils.io import atomic_json
from .checkpoint import capture_rng_state, restore_rng_state
from .distributed import fsdp_state_context

def save_checkpoint(path: Path, model: FSDP, optimizer: torch.optim.Optimizer,
                    config: dict[str, Any], training_state: dict[str, Any],
                    mean: np.ndarray, scale: np.ndarray, rank: int, world: int) -> None:
    local_rng = capture_rng_state()
    rng_states: list[Any] | None = [None] * world if rank == 0 else None
    dist.gather_object(local_rng, rng_states, dst=0)
    with fsdp_state_context(model):
        model_state = model.state_dict()
        optimizer_state = FSDP.optim_state_dict(model, optimizer)
    if rank == 0:
        payload = {
            "format": 3, "model": model_state, "optimizer": optimizer_state,
            "config": config, "coordinate_mean": mean.tolist(), "coordinate_scale": scale.tolist(),
            "training_state": {**training_state, "rng_states": rng_states, "world_size": world},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
    dist.barrier()


def update_latest_checkpoint(checkpoint: Path, latest: Path) -> None:
    """Atomically point ``latest`` at an immutable same-filesystem checkpoint."""
    latest.parent.mkdir(parents=True, exist_ok=True)
    temporary = latest.with_name(f".{latest.name}.{os.getpid()}.link.tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(checkpoint, temporary)
        temporary.replace(latest)
    finally:
        temporary.unlink(missing_ok=True)


def launch_durable_checkpoint_replica(source: Path, destination: Path,
                                      step: int) -> bool:
    """Start a detached, best-effort SSD-to-HDD replica without blocking training."""
    log_path = source.parent / "durable_replication.log"
    command = [
        sys.executable, "-m", "worldbridge.trainer.commands.replicate_checkpoint",
        "--source", str(source), "--destination", str(destination),
        "--step", str(int(step)),
    ]
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab", buffering=0) as stream:
            subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, close_fds=True, start_new_session=True,
            )
        print(json.dumps({
            "event": "durable_checkpoint_queued", "step": int(step),
            "source": str(source), "destination": str(destination),
            "log": str(log_path),
        }), flush=True)
        return True
    except Exception as error:
        print(json.dumps({
            "event": "durable_checkpoint_queue_failed", "step": int(step),
            "source": str(source), "destination": str(destination),
            "error": repr(error),
        }), file=sys.stderr, flush=True)
        return False


def prune_periodic_checkpoints(output: Path, keep_last: int) -> list[str]:
    """Bound disk use while retaining latest.pt and the newest named milestones."""
    paths = sorted(output.glob("checkpoint-*.pt"))
    remove = paths[:-max(int(keep_last), 0)] if keep_last else paths
    removed = []
    for path in remove:
        path.unlink(missing_ok=True)
        removed.append(path.name)
    return removed


def load_unwrapped_model_checkpoint(
    path: Path, model: torch.nn.Module, rank: int, world: int,
) -> tuple[dict[str, Any] | None, dict[str, Any], list[Any]]:
    """Load rank 0 before FSDP so ``sync_module_states`` broadcasts exact weights.

    Loading a rank-0-only FULL_STATE_DICT after FSDP construction does not
    broadcast it: nonzero ranks that receive an empty state dict retain their
    construction weights. The subsequent all-reduce then trains a hybrid model.
    """
    payload = torch.load(
        path, map_location="cpu", mmap=True, weights_only=True,
    ) if rank == 0 else None
    metadata = [(
        int(payload["training_state"].get("world_size", world)),
        payload["training_state"], payload["training_state"].get("rng_states", []),
    ) if rank == 0 else None]
    dist.broadcast_object_list(metadata, src=0)
    saved_world, training_state, states = metadata[0]
    if saved_world != world:
        raise ValueError(f"exact resume world-size mismatch: checkpoint={saved_world}, current={world}")
    if len(states) != world:
        raise ValueError("checkpoint lacks one RNG state per rank")
    if rank == 0:
        model.load_state_dict(payload["model"], strict=True)
    return payload, training_state, states


def load_optimizer_checkpoint(payload: dict[str, Any] | None, model: FSDP,
                              optimizer: torch.optim.Optimizer, states: list[Any],
                              rank: int) -> None:
    full_optimizer_state = payload["optimizer"] if rank == 0 else None
    optimizer_state = FSDP.scatter_full_optim_state_dict(
        full_optimizer_state, model, optim=optimizer,
    )
    optimizer.load_state_dict(optimizer_state)
    restore_rng_state(states[rank])
