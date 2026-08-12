#!/usr/bin/env python3
"""Train Dense4D on the validated Dynamic Replica cache.

The default launcher uses one process on physical GPU 4.  This is deliberately
implemented with torch.distributed even for world size one: a later handoff can
stop at an optimizer boundary, save ``checkpoint.pt``, and restart the exact
model/AdamW/global-step state with more ranks.  Canonical Wan latent shards and
clip order do not depend on world size.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
import os
from pathlib import Path
import random
import signal
import sys
import threading
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from worldbridge.dense4d import masked_pair_smooth_l1
from worldbridge.dense4d_runtime import (
    apply_linear_warmup, build_real_model, capture_rng_state, parameter_groups,
    precision_dtype, restore_rng_state, save_checkpoint,
)
from worldbridge.dynamic_replica import DynamicReplicaDataset, T

_STOP_SIGNAL = threading.Event()


def _signal_handler(_signum: int, _frame: Any) -> None:
    # Signal handlers must not perform CUDA or filesystem work.  The training
    # loop turns this into a distributed, optimizer-boundary stop request.
    _STOP_SIGNAL.set()


def init_distributed() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world < 1:
        raise RuntimeError(f"invalid WORLD_SIZE={world}")
    if not torch.cuda.is_available():
        raise RuntimeError("Dynamic Replica formal training requires CUDA")
    torch.cuda.set_device(local)
    dist.init_process_group(
        "nccl", init_method="env://", timeout=dt.timedelta(hours=24),
        device_id=torch.device(f"cuda:{local}"),
    )
    return rank, world, local


def validate_cache(config: dict[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    root = Path(config["cache_root"]).resolve()
    required = [
        root / "CACHE_COMPLETE.json", root / "manifest.json",
        root / "splits" / "train.jsonl",
        Path(config["coordinate_stats"]), Path(config["empty_text_condition"]),
        Path(config["wan_root"]) / "Wan2.1_VAE.pth",
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("validated Dynamic Replica cache is not ready; missing " + ", ".join(missing))
    marker = json.loads((root / "CACHE_COMPLETE.json").read_text())
    if marker.get("formal_training_allowed") is not True:
        raise RuntimeError(f"cache marker does not allow formal training: {marker}")
    rows = [json.loads(line) for line in (root / "splits" / "train.jsonl").read_text().splitlines() if line]
    if not rows:
        raise RuntimeError("Dynamic Replica train split is empty")
    indices = [int(row["index"]) for row in rows]
    if len(set(indices)) != len(indices) or indices != sorted(indices):
        raise RuntimeError("train rows must retain deterministic global manifest order")
    if any(len(row.get("frames", [])) != T for row in rows):
        raise RuntimeError("train split contains a clip that is not exactly 21 frames")
    state = json.loads((root / "PREPROCESSING_STATE.json").read_text())
    if state.get("status") != "validated":
        raise RuntimeError(f"preprocessing state is not validated: {state.get('status')}")
    return root, rows


def load_train_latents(root: Path, rows: list[dict[str, Any]]) -> torch.Tensor:
    """Materialize canonical global-order shards once, then select train rows.

    This avoids opening a safetensors file for every optimizer step and keeps
    shard names/order independent of DDP world size and future handoffs.
    """
    import re
    from safetensors.torch import load_file

    paths: list[tuple[Path, int, int]] = []
    for path in sorted((root / "latents" / "wan2.1_1.3b_fp32").glob("shard_*.safetensors")):
        match = re.fullmatch(r"shard_(\d+)_(\d+)", path.stem)
        if match:
            paths.append((path, int(match.group(1)), int(match.group(2))))
    if not paths:
        raise FileNotFoundError(f"no Dynamic Replica latent shards under {root / 'latents'}")
    expected_first = 0
    all_latents: torch.Tensor | None = None
    for path, first, count in paths:
        if first != expected_first or count <= 0:
            raise RuntimeError(f"latent shard coverage gap at {expected_first}: {path.name}")
        value = load_file(str(path), device="cpu")["latents"]
        if tuple(value.shape) != (count, 16, 6, 16, 16):
            raise RuntimeError(f"latent shape mismatch in {path}: {tuple(value.shape)}")
        if all_latents is None:
            all_latents = torch.empty((first + count, *value.shape[1:]), dtype=torch.float32)
        elif all_latents.shape[1:] != value.shape[1:]:
            raise RuntimeError(f"latent channel shape changed in {path}")
        if all_latents.shape[0] < first + count:
            all_latents = torch.cat((all_latents, torch.empty((first + count - all_latents.shape[0], *value.shape[1:]), dtype=torch.float32)))
        all_latents[first:first + count].copy_(value)
        expected_first += count
        del value
    assert all_latents is not None
    global_indices = torch.tensor([int(row["index"]) for row in rows], dtype=torch.long)
    if int(global_indices.max()) >= len(all_latents):
        raise RuntimeError(f"latent cache ends at {len(all_latents)}, but train needs {int(global_indices.max()) + 1}")
    selected = all_latents[global_indices].contiguous()
    del all_latents
    return selected


def plan(rows: list[dict[str, Any]], global_step: int, clips_seen: int,
         batch: int, world: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return this rank's contiguous slice; ``clips_seen`` survives world changes."""
    rank = int(os.environ.get("RANK", "0"))
    total = batch * world
    begin = int(clips_seen) + rank * batch
    indices = np.asarray([(begin + n) % len(rows) for n in range(batch)], dtype=np.int64)
    # Draw sources for the global batch so all ranks agree without a broadcast.
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(global_step)]))
    sources = rng.integers(T, size=total, dtype=np.int64)[rank * batch:(rank + 1) * batch]
    source = np.broadcast_to(sources[:, None], (batch, T)).copy()
    target = np.broadcast_to(np.arange(T, dtype=np.int64)[None, :], (batch, T)).copy()
    return indices, source, target


class GeometryPrefetcher:
    """One Future per clip, with bounded look-ahead and shared dataset cache."""
    def __init__(self, dataset: DynamicReplicaDataset, mean: np.ndarray,
                 scale: np.ndarray, workers: int, depth: int):
        self.dataset = dataset
        self.mean = mean.astype(np.float32)
        self.scale = scale.astype(np.float32)
        self.pool = ThreadPoolExecutor(max_workers=max(1, int(workers)), thread_name_prefix="dynamic-geometry")
        self.futures: dict[int, list[Any]] = {}
        self.depth = max(1, int(depth))

    def submit(self, step: int, indices: np.ndarray, source: np.ndarray) -> None:
        def make(index: int, source_frame: int) -> tuple[np.ndarray, np.ndarray]:
            xyz, valid = self.dataset.source_all_targets(index, source_frame)
            normalized = (xyz - self.mean[None, :, None, None]) / self.scale[None, :, None, None]
            return normalized.astype(np.float32), valid
        self.futures[int(step)] = [
            self.pool.submit(make, int(index), int(src))
            for index, src in zip(indices, source[:, 0])
        ]

    def get(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        values = [future.result() for future in self.futures.pop(int(step))]
        return np.stack([item[0] for item in values]), np.stack([item[1] for item in values])

    def close(self) -> None:
        self.pool.shutdown(wait=True)


def init_wandb(config: dict[str, Any], output: Path, rank: int):
    tracking = config.get("tracking", {})
    if rank != 0 or not tracking.get("enabled", True) or os.environ.get("WANDB_MODE") == "disabled":
        return None
    import wandb
    output.mkdir(parents=True, exist_ok=True)
    run_id_file = output / "wandb_run_id"
    run_id = os.environ.get("WANDB_RUN_ID")
    if not run_id:
        if run_id_file.exists():
            run_id = run_id_file.read_text().strip()
        else:
            run_id = wandb.util.generate_id()
            run_id_file.write_text(run_id + "\n")
    mode = os.environ.get("WANDB_MODE", "online")
    if mode == "online" and not os.environ.get("WANDB_API_KEY") and not Path.home().joinpath(".netrc").exists():
        mode = "offline"
        print("WANDB_UPLOAD_PENDING: no credential; recording an offline run", flush=True)
    run = wandb.init(
        id=run_id, resume="allow", mode=mode,
        project=os.environ.get("WANDB_PROJECT", tracking.get("project", "worldbridge4d")),
        entity=os.environ.get("WANDB_ENTITY", tracking.get("entity")),
        name=os.environ.get("WANDB_NAME", "dynamic-replica-dense4d-500k"),
        group=os.environ.get("WANDB_GROUP", tracking.get("group", "dynamic-replica-dense4d-500k")),
        job_type="train", tags=tracking.get("tags", ["dynamic-replica", "dense4d"]), config=config,
    )
    run.define_metric("global_step")
    run.define_metric("train/*", step_metric="global_step")
    print(f"WANDB_RUN_URL: {run.url}", flush=True)
    return run


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--resume")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--disable-wandb", action="store_true")
    ap.add_argument("--stop-file", help="checkpoint and exit after the current optimizer update")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGUSR1, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    config = yaml.safe_load(Path(args.config).read_text())
    rank, world, local = init_distributed()
    device = torch.device(f"cuda:{local}")
    seed = int(config.get("seed", 2026))
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    root, rows = validate_cache(config)
    dataset = DynamicReplicaDataset(root, "train")
    if len(dataset) != len(rows):
        raise RuntimeError(f"dataset index length changed: {len(dataset)} != {len(rows)}")
    stats_path = Path(config["coordinate_stats"])
    with np.load(stats_path) as stats:
        mean = stats["mean"].astype(np.float32)
        scale = np.maximum(stats["scale"].astype(np.float32), 1e-6)
    clean_latents = load_train_latents(root, rows)
    if rank == 0:
        print(json.dumps({"event": "dynamic_replica_cache_ready", "clips": len(dataset),
                          "latent_shape": list(clean_latents.shape[1:]), "world_size": world}), flush=True)

    model = build_real_model(config, device)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True) if rank == 0 else None
    resume_payload = None
    resume_path = args.resume or (str(output / "checkpoint.pt") if (output / "checkpoint.pt").is_file() else None)
    if resume_path:
        resume_payload = torch.load(resume_path, map_location="cpu", mmap=True, weights_only=True)
        model.load_state_dict(resume_payload["model"], strict=True)
    ddp = DDP(model, device_ids=[local], output_device=local, broadcast_buffers=False,
              # Structured Wan readout intentionally leaves a fixed subset of
              # full-finetune parameters unused; this is required for DDP's
              # reduction state on the second and later optimizer steps.
              find_unused_parameters=True)
    optimizer = torch.optim.AdamW(parameter_groups(model, config),
                                   weight_decay=float(config.get("weight_decay", 1e-4)))
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])
    batch = int(config.get("batch_size_per_gpu", config.get("batch_size", 1)))
    steps = int(args.steps or config.get("steps", 500000))
    if batch != 8:
        raise ValueError(f"the registered Dynamic Replica capacity run requires batch_size_per_gpu=8, got {batch}")
    if resume_payload is not None:
        old_batch = int(resume_payload.get("config", {}).get("batch_size_per_gpu", batch))
        if old_batch != batch:
            # AdamW/global_step checkpoints remain valid when changing the
            # micro-batch; clips_seen preserves sample accounting.  Record the
            # explicit capacity change instead of silently claiming an exact
            # same-batch resume.
            if rank == 0:
                print(f"DYNAMIC_REPLICA_BATCH_CHANGE: checkpoint={old_batch} current={batch}; optimizer state resumed", flush=True)
    config["batch_size_per_gpu"] = batch
    config["steps"] = steps
    start_step = int(resume_payload["training_state"]["global_step"]) if resume_payload else 0
    clips_seen = int(resume_payload["training_state"].get("clips_seen", 0)) if resume_payload else 0
    if start_step >= steps:
        raise ValueError(f"checkpoint global_step={start_step} already reaches target steps={steps}")
    # Only rank zero owns the checkpoint RNG.  Other ranks are deterministically
    # re-seeded on a world-size handoff; data/source planning is stateless.
    if rank == 0 and resume_payload and "rng_state" in resume_payload.get("training_state", {}):
        restore_rng_state(resume_payload["training_state"]["rng_state"])
    del resume_payload

    run = None if args.disable_wandb else init_wandb(config, output, rank)
    prefetch = GeometryPrefetcher(dataset, mean, scale,
                                  config.get("geometry_prefetch_workers", 8),
                                  config.get("geometry_prefetch_depth", 2))
    checkpoint_every = int(config.get("checkpoint_every_steps", 1000))
    started = time.perf_counter()
    first_loss: float | None = None
    last_loss: float | None = None
    completed_steps = start_step
    stopped_for_handoff = False
    prefetch_depth = max(1, int(config.get("geometry_prefetch_depth", 2)))

    def should_stop() -> bool:
        value = torch.tensor(int(_STOP_SIGNAL.is_set() or (rank == 0 and args.stop_file and Path(args.stop_file).exists())),
                             device=device, dtype=torch.int32)
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        return bool(value.item())

    def checkpoint(tag: str) -> None:
        if rank == 0:
            save_checkpoint(
                output / "checkpoint.pt", model, config, mean, scale,
                extra={"steps": steps, "total_steps": completed_steps, "dataset": "DynamicReplica",
                       "world_size_at_save": world, "checkpoint_tag": tag},
                optimizer=optimizer,
                training_state={"global_step": completed_steps, "optimizer_updates": completed_steps,
                                "clips_seen": clips_seen, "rng_state": capture_rng_state()},
            )
            write_json(output / "checkpoint_status.json", {
                "dataset": "DynamicReplica", "global_step": completed_steps,
                "clips_seen": clips_seen, "world_size": world, "tag": tag,
                "saved_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            })
            print(f"DYNAMIC_REPLICA_CHECKPOINT: global_step={completed_steps} tag={tag}", flush=True)
        dist.barrier()

    try:
        for queued_step in range(start_step, min(start_step + prefetch_depth, steps)):
            qi, qs, _ = plan(rows, queued_step, clips_seen + (queued_step - start_step) * batch * world,
                             batch, world, seed)
            prefetch.submit(queued_step, qi, qs)
        for step in range(start_step, steps):
            indices, source, target = plan(rows, step, clips_seen, batch, world, seed)
            target_xyz_np, valid_np = prefetch.get(step)
            replacement = step + prefetch_depth
            if replacement < steps:
                ri, rs, _ = plan(rows, replacement,
                                 clips_seen + (replacement - start_step) * batch * world,
                                 batch, world, seed)
                prefetch.submit(replacement, ri, rs)
            source_t = torch.from_numpy(source).to(device, non_blocking=True)
            target_t = torch.from_numpy(target).to(device, non_blocking=True)
            target_xyz = torch.from_numpy(target_xyz_np).to(device, non_blocking=True)
            valid = torch.from_numpy(valid_np).to(device, non_blocking=True)
            latent = clean_latents[torch.from_numpy(indices)].to(device, dtype=precision_dtype(config.get("precision", "bf16")), non_blocking=True)
            dtype = precision_dtype(config.get("precision", "bf16"))
            apply_linear_warmup(optimizer, step + 1, int(config.get("warmup_steps", 256)))
            with torch.autocast(device_type="cuda", dtype=dtype):
                prediction, _, _ = ddp(latent, source_t, target_t)
                loss = masked_pair_smooth_l1(
                    prediction.float(), target_xyz.float(), valid,
                    beta=float(config.get("smooth_l1_beta", 0.05)),
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at global_step={step + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip", 1.0)))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            value = float(loss.detach())
            first_loss = value if first_loss is None else first_loss
            last_loss = value
            completed_steps = step + 1
            clips_seen += batch * world

            if rank == 0 and (step == start_step or completed_steps % int(config.get("diagnostic_every_steps", 20)) == 0 or completed_steps == steps):
                with torch.no_grad():
                    epe = torch.linalg.vector_norm(
                        (prediction.float() - target_xyz) * torch.as_tensor(scale, device=device).view(1, 1, 3, 1, 1), dim=2,
                    )
                    epe_value = float(epe[valid].mean()) if bool(valid.any()) else float("nan")
                payload = {
                    "global_step": completed_steps, "train/loss": value, "train/epe_m": epe_value,
                    "train/clips_seen": clips_seen,
                    "system/world_size": world,
                    "system/peak_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                    "timing/elapsed_seconds": time.perf_counter() - started,
                }
                print(json.dumps(payload), flush=True)
                if run is not None:
                    run.log(payload, step=completed_steps)

            need_periodic = bool(checkpoint_every and completed_steps % checkpoint_every == 0)
            need_stop = should_stop()
            if need_periodic or need_stop:
                checkpoint("handoff" if need_stop else "periodic")
            if need_stop:
                stopped_for_handoff = True
                break

        prefetch.close()
        if not stopped_for_handoff and completed_steps >= steps:
            checkpoint("final")
    finally:
        # get()/checkpoint failures should still release all geometry workers.
        prefetch.close()
        if run is not None:
            run.finish()

    dist.barrier()
    if rank == 0:
        result = {
            "dataset": "DynamicReplica", "world_size": world, "target_steps": steps,
            "completed_steps": completed_steps, "stopped_for_handoff": stopped_for_handoff,
            "batch_size_per_gpu": batch, "global_batch_size": batch * world,
            "clips_seen": clips_seen, "clips": len(dataset), "first_loss": first_loss,
            "final_loss": last_loss, "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "checkpoint": str(output / "checkpoint.pt"), "wandb_url": run.url if run else None,
            "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        write_json(output / "train_metrics.json", result)
        if completed_steps >= steps and not stopped_for_handoff:
            write_json(output / "TRAINING_COMPLETE.json", result)
        print(json.dumps(result, indent=2), flush=True)
        print("DYNAMIC_REPLICA_HANDOFF_READY" if stopped_for_handoff else "DYNAMIC_REPLICA_TRAIN_OK", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
