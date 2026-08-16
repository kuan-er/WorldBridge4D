#!/usr/bin/env python3
"""Recover the tiny resume-planning sidecar from a verified full checkpoint.

This is for interruption after ``latest.pt`` was atomically published but
before normal-exit ``train_status.json``. It validates the full checkpoint once
and refuses to replace a conflicting sidecar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while value := stream.read(chunk):
            digest.update(value)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f".{os.getpid()}.tmp.json")
    try:
        with temporary.open("w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def recover_status(checkpoint: Path, output: Path) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    output = output.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint is missing: {checkpoint}")
    payload = torch.load(
        checkpoint, map_location="cpu", mmap=True, weights_only=True,
    )
    if int(payload.get("format", -1)) != 3:
        raise ValueError(f"unsupported checkpoint format: {payload.get('format')}")
    state = payload.get("training_state", {})
    step = int(state.get("global_step", -1))
    world = int(state.get("world_size", -1))
    rng_states = state.get("rng_states", [])
    clips = {key: int(value) for key, value in state.get("clips_seen", {}).items()}
    expected_names = {"kubric", "pointodyssey", "dynamic_replica"}
    if step < 1 or world < 1 or len(rng_states) != world or set(clips) != expected_names:
        raise ValueError(
            "checkpoint resume metadata is incomplete: "
            f"step={step}, world={world}, rng_states={len(rng_states)}, clips={clips}"
        )
    config = payload.get("config", {})
    target = int(config.get("max_steps", config.get("schedule_horizon_steps", -1)))
    if target <= step:
        raise ValueError(f"invalid target_steps={target} for checkpoint step={step}")
    result = {
        "completed_steps": step,
        "target_steps": target,
        "world_size": world,
        "clips_seen": clips,
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256(checkpoint),
        "recovered_from_checkpoint": True,
    }
    if output.exists():
        existing = json.loads(output.read_text())
        comparable = ("completed_steps", "target_steps", "world_size", "clips_seen")
        if any(existing.get(key) != result[key] for key in comparable):
            raise RuntimeError(f"refusing conflicting status sidecar: {output}")
        return existing
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    output = Path(args.output) if args.output else checkpoint.parent / "train_status.json"
    result = recover_status(checkpoint, output)
    print(json.dumps({"event": "train_status_recovered", **result}), flush=True)
    print("TRAIN_STATUS_RECOVERY_OK", flush=True)


if __name__ == "__main__":
    main()
