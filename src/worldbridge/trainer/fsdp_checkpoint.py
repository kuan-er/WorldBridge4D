"""FSDP checkpoint publication, migration, and exact restore."""
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


def launch_post_resume_checksum(marker: Path, resume: Path, step: int) -> None:
    """Hash an already strictly restored immutable checkpoint in the background."""
    marker = marker.resolve()
    value = json.loads(marker.read_text())
    marked_checkpoint = Path(value["checkpoint"]).resolve()
    if marked_checkpoint != resume.resolve():
        raise ValueError(
            f"post-resume checksum marker checkpoint mismatch: {marked_checkpoint} != {resume.resolve()}"
        )
    log_path = marker.parent / "post_resume_checksum.log"
    command = [
        "ionice", "-c", "3", "nice", "-n", "19", sys.executable,
        "-m", "worldbridge.trainer.commands.handoff",
        "--finalize-checksum-marker", str(marker),
    ]
    with log_path.open("ab", buffering=0) as stream:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            close_fds=True, start_new_session=True,
        )
    print(json.dumps({
        "event": "post_resume_checksum_started", "step": int(step),
        "pid": process.pid, "marker": str(marker), "log": str(log_path),
    }), flush=True)


def prune_periodic_checkpoints(output: Path, keep_last: int) -> list[str]:
    """Bound disk use while retaining latest.pt and the newest named milestones."""
    paths = sorted(output.glob("checkpoint-*.pt"))
    remove = paths[:-max(int(keep_last), 0)] if keep_last else paths
    removed = []
    for path in remove:
        path.unlink(missing_ok=True)
        removed.append(path.name)
    return removed


def load_model_state_for_resume(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
    allowed_missing_prefixes: tuple[str, ...] = (),
) -> list[str]:
    """Load an exact state, or one audited zero-init structural extension."""
    if not allowed_missing_prefixes:
        model.load_state_dict(state, strict=True)
        return []
    missing, unexpected = model.load_state_dict(state, strict=False)
    invalid_missing = [
        name for name in missing
        if not name.startswith(allowed_missing_prefixes)
    ]
    if unexpected or invalid_missing or not missing:
        raise RuntimeError(
            "structural checkpoint migration mismatch; "
            f"missing={missing[:8]}, invalid_missing={invalid_missing[:8]}, "
            f"unexpected={unexpected[:8]}"
        )
    return list(missing)


def load_unwrapped_model_checkpoint(
    path: Path, model: torch.nn.Module, rank: int, world: int,
    allowed_missing_prefixes: tuple[str, ...] = (),
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
        missing = load_model_state_for_resume(
            model, payload["model"], allowed_missing_prefixes,
        )
        if missing:
            print(json.dumps({
                "event": "structural_model_extension_loaded",
                "fresh_parameters": len(missing),
                "allowed_prefixes": list(allowed_missing_prefixes),
            }), flush=True)
    return payload, training_state, states


def load_initial_model_weights(
    path: Path,
    model: torch.nn.Module,
    rank: int,
    expected_global_step: int = 0,
    expected_clips_seen: dict[str, int] | None = None,
    expected_schedule: dict[str, Any] | None = None,
    restore_optimizer: bool = False,
) -> dict[str, Any] | None:
    """Initialize the RGB-pyramid route from an old strict model-only state.

    With ``restore_optimizer``, existing parameters retain their saved AdamW
    moments and step while the new RGB parameters receive empty, lazy optimizer
    state.  This is a structure-aware continuation, not a strict exact resume.
    """
    if rank != 0:
        return None
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    source_state = payload.get("training_state", {})
    if expected_global_step:
        source_step = int(source_state.get("global_step", -1))
        if source_step != int(expected_global_step):
            raise RuntimeError(
                f"initial model step {source_step} != expected {expected_global_step}"
            )
        source_clips = {
            key: int(value)
            for key, value in source_state.get("clips_seen", {}).items()
        }
        expected_clips = {
            key: int(value) for key, value in (expected_clips_seen or {}).items()
        }
        if source_clips != expected_clips:
            raise RuntimeError(
                f"initial model clip counters {source_clips} != expected {expected_clips}"
            )
        source_config = payload.get("config", {})
        schedule_mismatches = {
            key: (source_config.get(key), value)
            for key, value in (expected_schedule or {}).items()
            if source_config.get(key) != value
        }
        if schedule_mismatches:
            raise RuntimeError(
                f"initial model LR schedule mismatch: {schedule_mismatches}"
            )
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    allowed_prefixes = (
        "decoder.upsampler.source_rgb_encoder.",
        "decoder.upsampler.source_fusions.",
    )
    invalid_missing = [
        name for name in missing if not name.startswith(allowed_prefixes)
    ]
    if invalid_missing or unexpected:
        raise RuntimeError(
            "initial model checkpoint mismatch; "
            f"missing={invalid_missing[:8]}, unexpected={unexpected[:8]}"
        )
    if not missing:
        raise RuntimeError("initial model checkpoint already contains the RGB pyramid")
    print(json.dumps({
        "event": "initial_model_weights_loaded",
        "checkpoint": str(path.resolve()),
        "new_parameters": len(missing),
        "optimizer": (
            "checkpoint_plus_fresh_rgb" if restore_optimizer else "fresh"
        ),
        "rng": "restored" if restore_optimizer else "fresh",
        "global_step": int(expected_global_step),
        "clips_seen": expected_clips_seen or {},
    }), flush=True)
    return payload


def extend_optimizer_state_for_rgb(
    source: dict[str, Any], current_group_names: dict[str, list[str]],
) -> tuple[dict[str, Any], list[str]]:
    """Extend a name-keyed full AdamW state with only the new RGB parameters."""
    source_groups = {
        str(group.get("name")): group for group in source["param_groups"]
    }
    if set(source_groups) != set(current_group_names):
        raise RuntimeError(
            f"optimizer groups changed: {set(source_groups)} != {set(current_group_names)}"
        )
    merged_groups = []
    added_names: list[str] = []
    allowed_prefixes = (
        "decoder.upsampler.source_rgb_encoder.",
        "decoder.upsampler.source_fusions.",
    )
    for group_name, names in current_group_names.items():
        old_group = source_groups[group_name]
        old_names = list(old_group["params"])
        old_set = set(old_names)
        current_set = set(names)
        removed = sorted(old_set - current_set)
        added = [name for name in names if name not in old_set]
        invalid_added = [
            name for name in added if not name.startswith(allowed_prefixes)
        ]
        if removed or invalid_added:
            raise RuntimeError(
                f"optimizer parameter mismatch in {group_name}: "
                f"removed={removed[:8]}, invalid_added={invalid_added[:8]}"
            )
        if added and group_name != "dense_decoder":
            raise RuntimeError(
                f"new RGB optimizer parameters unexpectedly entered {group_name}"
            )
        merged_groups.append({**old_group, "params": list(names)})
        added_names.extend(added)
    if not added_names:
        raise RuntimeError("optimizer checkpoint already contains RGB parameters")
    return {
        "state": source["state"],
        "param_groups": merged_groups,
    }, added_names


def load_extended_optimizer_checkpoint(
    payload: dict[str, Any] | None,
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    states: list[Any],
    current_group_names: dict[str, list[str]],
    rank: int,
) -> None:
    full_optimizer_state = None
    added_names: list[str] = []
    if rank == 0:
        if payload is None:
            raise RuntimeError("rank zero lacks initial optimizer checkpoint")
        full_optimizer_state, added_names = extend_optimizer_state_for_rgb(
            payload["optimizer"], current_group_names,
        )
    optimizer_state = FSDP.scatter_full_optim_state_dict(
        full_optimizer_state, model, optim=optimizer,
    )
    optimizer.load_state_dict(optimizer_state)
    restore_rng_state(states[rank])
    if rank == 0:
        print(json.dumps({
            "event": "initial_optimizer_state_loaded",
            "restored_parameters": len(payload["optimizer"]["state"]),
            "fresh_rgb_parameters": len(added_names),
            "optimizer_step": int(payload["training_state"]["global_step"]),
            "rng_states": len(states),
        }), flush=True)


def filter_optimizer_state_for_trainable(
    source: dict[str, Any], current_groups: list[dict[str, Any]],
    *, allow_fresh_non_rgb: bool = False,
    allowed_fresh_rgb_prefixes: tuple[str, ...] = (),
) -> tuple[dict[str, Any], list[str]]:
    """Restore all available moments while strictly preserving RGB history.

    RGB-only warm-up requires every current tensor to have a source state. The
    following joint phase may add Wan and decoder tensors with fresh AdamW state,
    but all RGB tensors must still restore their accumulated moments.
    """
    prefixes = (
        "decoder.upsampler.source_rgb_encoder.",
        "decoder.upsampler.source_fusions.",
    )
    names = [str(name) for group in current_groups for name in group["params"]]
    if len(names) != len(set(names)):
        raise RuntimeError("fine-tune optimizer contains duplicate parameters")
    current = set(names)
    source_names = set(source["state"])
    invalid = [name for name in names if not name.startswith(prefixes)]
    if invalid and not allow_fresh_non_rgb:
        raise RuntimeError(
            f"RGB-only optimizer unexpectedly includes non-RGB parameters: {invalid[:8]}"
        )
    rgb_names = {name for name in current if name.startswith(prefixes)}
    missing_rgb = sorted(rgb_names - source_names)
    invalid_missing_rgb = [
        name for name in missing_rgb
        if not name.startswith(allowed_fresh_rgb_prefixes)
    ]
    if invalid_missing_rgb:
        raise RuntimeError(
            f"source checkpoint lacks RGB AdamW state: {invalid_missing_rgb[:8]}"
        )
    fresh = [name for name in names if name not in source_names]
    if fresh and not allow_fresh_non_rgb:
        raise RuntimeError(f"source checkpoint lacks RGB AdamW state: {fresh[:8]}")
    fresh_rgb = [name for name in fresh if name.startswith(prefixes)]
    invalid_fresh_rgb = [
        name for name in fresh_rgb
        if not name.startswith(allowed_fresh_rgb_prefixes)
    ]
    if invalid_fresh_rgb:
        raise RuntimeError(
            f"joint phase would reset RGB AdamW state: {invalid_fresh_rgb[:8]}"
        )
    restored = [name for name in names if name in source_names]
    return ({
        "state": {name: source["state"][name] for name in restored},
        "param_groups": current_groups,
    }, fresh)


def load_finetune_optimizer_checkpoint(
    payload: dict[str, Any] | None,
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    states: list[Any],
    current_group_names: dict[str, list[str]],
    rank: int,
    *,
    allow_fresh_non_rgb: bool = False,
    allowed_fresh_rgb_prefixes: tuple[str, ...] = (),
) -> None:
    full_optimizer_state = None
    restored = 0
    fresh = 0
    if rank == 0:
        if payload is None:
            raise RuntimeError("rank zero lacks the fine-tune source checkpoint")
        current_groups = []
        for group in optimizer.param_groups:
            group_name = str(group["name"])
            values = {key: value for key, value in group.items() if key != "params"}
            values["params"] = list(current_group_names[group_name])
            current_groups.append(values)
        full_optimizer_state, fresh_names = filter_optimizer_state_for_trainable(
            payload["optimizer"], current_groups,
            allow_fresh_non_rgb=allow_fresh_non_rgb,
            allowed_fresh_rgb_prefixes=allowed_fresh_rgb_prefixes,
        )
        restored = len(full_optimizer_state["state"])
        fresh = len(fresh_names)
    optimizer_state = FSDP.scatter_full_optim_state_dict(
        full_optimizer_state, model, optim=optimizer,
    )
    optimizer.load_state_dict(optimizer_state)
    restore_rng_state(states[rank])
    if rank == 0:
        print(json.dumps({
            "event": "finetune_optimizer_state_loaded",
            "restored_trainable_parameters": restored,
            "fresh_trainable_parameters": fresh,
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
