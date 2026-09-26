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
    allowed_missing_prefixes: tuple[str, ...] = (),
    allowed_unexpected_prefixes: tuple[str, ...] = (),
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
        if not allowed_missing_prefixes and not allowed_unexpected_prefixes:
            model.load_state_dict(payload["model"], strict=True)
        else:
            missing, unexpected = model.load_state_dict(payload["model"], strict=False)
            invalid_missing = [
                name for name in missing
                if not name.startswith(allowed_missing_prefixes)
            ]
            invalid_unexpected = [
                name for name in unexpected
                if not name.startswith(allowed_unexpected_prefixes)
            ]
            if (invalid_unexpected or invalid_missing
                    or (not missing and not unexpected)):
                raise RuntimeError(
                    "structural checkpoint migration mismatch: "
                    f"missing={missing[:8]}, invalid_missing={invalid_missing[:8]}, "
                    f"unexpected={unexpected[:8]}, invalid_unexpected={invalid_unexpected[:8]}"
                )
            print(json.dumps({
                "event": "structural_model_extension_loaded",
                "fresh_parameters": len(missing),
                "dropped_parameters": len(unexpected),
                "allowed_prefixes": list(allowed_missing_prefixes),
                "dropped_prefixes": list(allowed_unexpected_prefixes),
            }), flush=True)
    return payload, training_state, states


def load_filtered_optimizer_checkpoint(
    payload: dict[str, Any] | None,
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    states: list[Any],
    current_group_names: dict[str, list[str]],
    rank: int,
    allowed_fresh_prefixes: tuple[str, ...],
    require_all_source_state: bool = False,
    allowed_unexpected_prefixes: tuple[str, ...] = (),
) -> None:
    """Restore retained moments while allowing only audited new parameters."""
    full_optimizer_state = None
    restored = 0
    fresh_names: list[str] = []
    if rank == 0:
        if payload is None:
            raise RuntimeError("rank zero lacks the structural fine-tune checkpoint")
        current_groups = []
        for group in optimizer.param_groups:
            group_name = str(group["name"])
            values = {key: value for key, value in group.items() if key != "params"}
            values["params"] = list(current_group_names[group_name])
            current_groups.append(values)
        names = [str(name) for group in current_groups for name in group["params"]]
        if len(names) != len(set(names)):
            raise RuntimeError("fine-tune optimizer contains duplicate parameters")
        source_state = payload["optimizer"]["state"]
        dropped = [
            name for name in source_state
            if name not in names and not name.startswith(allowed_unexpected_prefixes)
        ]
        if dropped:
            raise RuntimeError(
                f'structural extension would discard existing Adam state: {dropped[:8]}'
            )
        if require_all_source_state and set(source_state) - set(names) \
                and not allowed_unexpected_prefixes:
            raise RuntimeError('structural extension would discard existing Adam state')
        fresh_names = [name for name in names if name not in source_state]
        invalid_fresh = [
            name for name in fresh_names
            if not name.startswith(allowed_fresh_prefixes)
        ]
        if invalid_fresh:
            raise RuntimeError(
                f"checkpoint lacks unaudited optimizer state: {invalid_fresh[:8]}"
            )
        retained_state = {
            name: source_state[name] for name in names if name in source_state
        }
        restored = len(retained_state)
        full_optimizer_state = {
            "state": retained_state,
            "param_groups": current_groups,
        }
    optimizer_state = FSDP.scatter_full_optim_state_dict(
        full_optimizer_state, model, optim=optimizer,
    )
    optimizer.load_state_dict(optimizer_state)
    restore_rng_state(states[rank])
    if rank == 0:
        print(json.dumps({
            "event": "filtered_optimizer_state_loaded",
            "restored_trainable_parameters": restored,
            "fresh_trainable_parameters": len(fresh_names),
            "fresh_prefixes": list(allowed_fresh_prefixes),
            "optimizer_groups": [group["name"] for group in optimizer.param_groups],
            "rng_states": len(states),
        }), flush=True)


def load_optimizer_checkpoint(payload: dict[str, Any] | None, model: FSDP,
                              optimizer: torch.optim.Optimizer, states: list[Any],
                              rank: int) -> None:
    full_optimizer_state = payload["optimizer"] if rank == 0 else None
    optimizer_state = FSDP.scatter_full_optim_state_dict(
        full_optimizer_state, model, optim=optimizer,
    )
    optimizer.load_state_dict(optimizer_state)
    restore_rng_state(states[rank])
