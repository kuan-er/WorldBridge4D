#!/usr/bin/env python3
"""Freeze and validate a K19 GPU1/4 checkpoint for the optimized 150k handoff."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
from typing import Any

import torch
import yaml


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha256(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_checkpoint(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        source_stat, destination_stat = source.stat(), destination.stat()
        if (source_stat.st_dev, source_stat.st_ino) != (
            destination_stat.st_dev, destination_stat.st_ino
        ):
            raise FileExistsError(f"immutable handoff checkpoint already differs: {destination}")
        return
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        try:
            os.link(source, temporary)
        except OSError:
            with source.open("rb") as reader, temporary.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=16 * 1024 * 1024)
                writer.flush()
                os.fsync(writer.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def build_extended_config(
    original: dict[str, Any],
    completed_step: int,
    target_steps: int,
    resume_status_path: Path,
) -> dict[str, Any]:
    config = dict(original)
    if int(config.get("targets_per_source", -1)) != 19:
        raise ValueError("handoff requires the current K19 trajectory")
    if (
        int(config.get("microbatch_per_gpu", -1)),
        int(config.get("gradient_accumulation", -1)),
    ) != (2, 2):
        raise ValueError("handoff requires the current B2/A2 trajectory")
    if int(config.get("schedule_horizon_steps", -1)) != 100000:
        raise ValueError("handoff expects the original 100k cosine horizon")
    if not int(config.get("warmup_steps", 0)) < completed_step < 100000:
        raise ValueError("handoff step must lie between warmup and the original horizon")
    if target_steps <= 100000 or target_steps <= completed_step:
        raise ValueError("target must extend both the original horizon and checkpoint step")

    config.update({
        "max_steps": int(target_steps),
        "schedule_extension_start_step": int(completed_step),
        "schedule_extension_horizon_steps": int(target_steps),
        "geometry_prefetch_depth": 2,
        "geometry_prefetch_workers": 4,
        "diagnostic_ensure_dataset_coverage": True,
        "resume_status_path": str(resume_status_path.resolve()),
    })
    kubric = dict(config["datasets"]["kubric"])
    kubric.update({
        "geometry_sample_cache_size": 16,
        "geometry_mmap_max_open_shards": 90,
    })
    config["datasets"] = dict(config["datasets"])
    config["datasets"]["kubric"] = kubric
    checkpoints = {
        int(step) for step in config.get("checkpoint_steps", [])
        if completed_step < int(step) <= target_steps
    }
    checkpoints.add(target_steps)
    config["checkpoint_steps"] = sorted(checkpoints)
    tracking = dict(config.get("tracking", {}))
    tracking["group"] = "worldbridge4d-256-two-gpu-b2-k19-a2-optimized-150k"
    tags = list(tracking.get("tags", []))
    for tag in (
        "optimized-prefetch-4x2", "dataset-diagnostic-coverage",
        f"resume-step-{completed_step}", f"target-{target_steps}",
        "continuous-cosine-extension",
    ):
        if tag not in tags:
            tags.append(tag)
    tracking["tags"] = tags
    config["tracking"] = tracking
    return config


def prepare(
    checkpoint: Path,
    status_path: Path,
    handoff_dir: Path,
    checkpoint_dir: Path,
    target_steps: int,
    defer_checksum: bool = False,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    status_path = status_path.resolve()
    if not checkpoint.is_file() or not status_path.is_file():
        raise FileNotFoundError("checkpoint and train_status.json must both exist")
    status = json.loads(status_path.read_text())
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    if int(payload.get("format", -1)) != 3:
        raise ValueError("unsupported checkpoint format")
    if not payload.get("model") or not payload.get("optimizer"):
        raise ValueError("checkpoint is missing full model or optimizer state")
    training = payload.get("training_state", {})
    step = int(training.get("global_step", -1))
    if step != int(status.get("completed_steps", -2)):
        raise ValueError(f"checkpoint/status step mismatch: {step} != {status.get('completed_steps')}")
    if int(training.get("world_size", -1)) != 2 or int(status.get("world_size", -1)) != 2:
        raise ValueError("handoff requires an exact two-rank checkpoint")
    if len(training.get("rng_states", [])) != 2:
        raise ValueError("checkpoint must contain one RNG state per rank")
    if training.get("clips_seen") != status.get("clips_seen"):
        raise ValueError("checkpoint/status dataset counters differ")

    immutable = handoff_dir.resolve() / f"checkpoint-{step:07d}.pt"
    freeze_checkpoint(checkpoint, immutable)
    digest = None if defer_checksum else sha256(immutable)
    immutable_status = checkpoint_dir.resolve() / "train_status.json"
    atomic_text(immutable_status, json.dumps({
        "completed_steps": step,
        "world_size": 2,
        "clips_seen": training["clips_seen"],
    }, indent=2) + "\n")
    config = build_extended_config(
        payload["config"], step, int(target_steps), immutable_status,
    )
    config_path = handoff_dir.resolve() / f"runtime_config_k19_optimized_150k_step{step}.yaml"
    atomic_text(config_path, yaml.safe_dump(config, sort_keys=False))
    marker = {
        "format": "worldbridge4d.gpu14_optimized_150k_handoff.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "checkpoint": str(immutable),
        "checkpoint_bytes": immutable.stat().st_size,
        "checkpoint_sha256": digest,
        "checksum_state": "deferred" if defer_checksum else "complete",
        "completed_step": step,
        "world_size": 2,
        "targets_per_source": 19,
        "microbatch_per_gpu": 2,
        "gradient_accumulation": 2,
        "target_steps": int(target_steps),
        "config": str(config_path),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "resume_status": str(immutable_status),
        "physical_gpus": [1, 4],
    }
    marker_path = handoff_dir.resolve() / "HANDOFF_READY.json"
    atomic_text(marker_path, json.dumps(marker, indent=2) + "\n")
    return {**marker, "marker": str(marker_path)}


def verify_marker(marker_path: Path, verify_checksum: bool = True) -> dict[str, Any]:
    marker = json.loads(marker_path.read_text())
    if marker.get("format") != "worldbridge4d.gpu14_optimized_150k_handoff.v1":
        raise ValueError("invalid handoff marker format")
    if marker.get("physical_gpus") != [1, 4] or int(marker.get("world_size", -1)) != 2:
        raise ValueError("handoff marker is not pinned to GPUs 1/4 with two ranks")
    checkpoint = Path(marker["checkpoint"])
    config = Path(marker["config"])
    status = Path(marker["resume_status"])
    if not checkpoint.is_file() or not config.is_file() or not status.is_file():
        raise FileNotFoundError("handoff artifact disappeared")
    if checkpoint.stat().st_size != int(marker["checkpoint_bytes"]):
        raise ValueError("handoff checkpoint size changed")
    if verify_checksum:
        if marker.get("checksum_state") != "complete" or not marker.get("checkpoint_sha256"):
            raise ValueError("handoff checkpoint checksum is not complete")
        if sha256(checkpoint) != marker["checkpoint_sha256"]:
            raise ValueError("handoff checkpoint checksum changed")
    return marker


def finalize_checksum_marker(marker_path: Path) -> dict[str, Any]:
    """Hash the immutable resume checkpoint after strict trainer restore."""
    marker_path = marker_path.resolve()
    lock_path = marker_path.with_suffix(marker_path.suffix + ".checksum.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = verify_marker(marker_path, verify_checksum=False)
        if marker.get("checksum_state") == "complete" and marker.get("checkpoint_sha256"):
            return marker
        checkpoint = Path(marker["checkpoint"])
        before = checkpoint.stat()
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        try:
            digest = sha256(checkpoint)
            after = checkpoint.stat()
            if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                raise RuntimeError("immutable checkpoint identity changed during checksum")
            marker["checkpoint_sha256"] = digest
            marker["checksum_state"] = "complete"
            marker["checksum_completed_at_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_text(marker_path, json.dumps(marker, indent=2) + "\n")
            return marker
        except Exception as error:
            marker["checksum_state"] = "failed"
            marker["checksum_error"] = repr(error)
            marker["checksum_failed_at_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_text(marker_path, json.dumps(marker, indent=2) + "\n")
            raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--status", type=Path)
    parser.add_argument("--handoff-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--target-steps", type=int, default=150000)
    parser.add_argument("--verify-marker", type=Path)
    parser.add_argument("--finalize-checksum-marker", type=Path)
    parser.add_argument("--skip-checksum", action="store_true")
    parser.add_argument("--defer-checksum", action="store_true")
    args = parser.parse_args()
    if args.verify_marker is not None:
        print(json.dumps(verify_marker(args.verify_marker, not args.skip_checksum), indent=2))
        return
    if args.finalize_checksum_marker is not None:
        print(json.dumps(finalize_checksum_marker(args.finalize_checksum_marker), indent=2))
        return
    required = (args.checkpoint, args.status, args.handoff_dir, args.checkpoint_dir)
    if any(value is None for value in required):
        parser.error("prepare mode requires checkpoint, status, handoff-dir, and checkpoint-dir")
    print(json.dumps(prepare(
        args.checkpoint, args.status, args.handoff_dir,
        args.checkpoint_dir, args.target_steps, args.defer_checksum,
    ), indent=2))


if __name__ == "__main__":
    main()
