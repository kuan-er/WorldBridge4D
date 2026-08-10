#!/usr/bin/env python3
"""Three-GPU DDP Dense4D training on the native PointOdyssey release.

Launch with ``torchrun --standalone --nproc_per_node=3`` and expose physical
GPUs 2,4,6 through ``CUDA_VISIBLE_DEVICES=2,4,6``.  The global step is an
optimizer update; one update consumes ``batch_size_per_gpu * world_size``
clips.  Rank zero is the only W&B client.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import sys
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
    precision_dtype, save_checkpoint,
)
from worldbridge.pointodyssey import PointOdysseyDataset
from worldbridge.wan import WanVAEEncoder, WAN_LATENT_SHAPE


def rank_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world != 3:
        raise RuntimeError(f"PointOdyssey DDP requires exactly 3 processes, got WORLD_SIZE={world}")
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", init_method="env://", timeout=__import__("datetime").timedelta(hours=24))
    return rank, world, local


def plan(num_clips: int, global_step: int, batch: int, world: int, seed: int,
         frames: int = 21) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return this rank's deterministic slice of one global DDP batch."""
    total = batch * world
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(global_step)]))
    indices = rng.integers(num_clips, size=total, dtype=np.int64)
    sources = rng.integers(frames, size=total, dtype=np.int64)
    begin = batch * int(os.environ.get("RANK", "0"))
    indices = indices[begin:begin + batch]
    sources = sources[begin:begin + batch]
    source = np.broadcast_to(sources[:, None], (batch, frames)).copy()
    target = np.broadcast_to(np.arange(frames, dtype=np.int64)[None, :], (batch, frames)).copy()
    return indices, source, target


def ensure_index_and_stats(config: dict[str, Any], rank: int) -> None:
    """Require the CPU-produced immutable handoff rather than racing a build."""
    root = Path(config["cache_root"])
    required = [root / "manifest.json", root / "splits" / "train.jsonl",
                root / "stats" / "coordinate_stats_train_source.npz"]
    if rank == 0:
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError("PointOdyssey handoff is incomplete; missing " + ", ".join(missing))
    dist.barrier()
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("PointOdyssey handoff disappeared: " + ", ".join(missing))


def ensure_latent_shard(dataset: PointOdysseyDataset, config: dict[str, Any],
                        rank: int, world: int, device: torch.device) -> Path:
    """Encode one disjoint shard per rank, then make the shared cache visible."""
    from safetensors.torch import save_file

    root = Path(config["latent_root"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"train-rank{rank:02d}-of{world:02d}.safetensors"
    count = len(dataset)
    ranges = np.array_split(np.arange(count, dtype=np.int64), world)
    own = ranges[rank]
    if not path.exists():
        checkpoint = Path(config["wan_root"]) / "Wan2.1_VAE.pth"
        encoder = WanVAEEncoder(checkpoint, device=device, dtype=torch.float32)
        values: list[torch.Tensor] = []
        started = time.perf_counter()
        with torch.inference_mode():
            for number, index in enumerate(own):
                rgb = torch.from_numpy(dataset.rgb(int(index))).permute(0, 3, 1, 2)[None].to(device)
                latent = encoder(rgb)
                if tuple(latent.shape[1:]) != WAN_LATENT_SHAPE:
                    raise RuntimeError(f"unexpected latent shape at index {index}: {tuple(latent.shape)}")
                values.append(latent.float().cpu()[0])
                if rank == 0 and (number == 0 or (number + 1) % 100 == 0):
                    print(json.dumps({"event": "pointodyssey_latents", "done": number + 1,
                                      "total": len(own), "seconds": time.perf_counter() - started}), flush=True)
        tensor = torch.stack(values).contiguous()
        temporary = path.with_suffix(".tmp.safetensors")
        save_file({"latent": tensor}, str(temporary))
        temporary.replace(path)
        del encoder, values, tensor
        torch.cuda.empty_cache()
    dist.barrier()
    paths = [root / f"train-rank{r:02d}-of{world:02d}.safetensors" for r in range(world)]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("latent cache incomplete: " + ", ".join(missing))
    return path


def load_latents(config: dict[str, Any], count: int, world: int) -> torch.Tensor:
    from safetensors.torch import load_file
    shards = []
    for rank in range(world):
        path = Path(config["latent_root"]) / f"train-rank{rank:02d}-of{world:02d}.safetensors"
        payload = load_file(str(path), device="cpu")
        shards.append(payload["latent"])
    # The split order is deterministic and matches np.array_split ranges.
    return torch.cat(shards, dim=0).contiguous()


class GeometryPrefetcher:
    """Bounded CPU source-track rasterization for one DDP rank."""
    def __init__(self, dataset: PointOdysseyDataset, mean: np.ndarray, scale: np.ndarray,
                 workers: int, depth: int):
        self.dataset, self.mean, self.scale = dataset, mean.astype(np.float32), scale.astype(np.float32)
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pointodyssey-geometry")
        self.futures: dict[int, Any] = {}
        self.depth = int(depth)

    def submit(self, step: int, indices: np.ndarray, source: np.ndarray) -> None:
        def make(index: int, src: int):
            xyz, valid = self.dataset.source_all_targets(int(index), int(src))
            normalized = (xyz - self.mean[None, :, None, None]) / self.scale[None, :, None, None]
            return normalized.astype(np.float32), valid
        self.futures[int(step)] = self.pool.submit(
            lambda: ([make(int(i), int(s)) for i, s in zip(indices, source[:, 0])])
        )

    def get(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        values = self.futures.pop(int(step)).result()
        return np.stack([x[0] for x in values]), np.stack([x[1] for x in values])

    def close(self) -> None:
        self.pool.shutdown(wait=True)


def init_wandb(config: dict[str, Any], rank: int):
    tracking = config.get("tracking", {})
    if rank != 0 or not tracking.get("enabled", True) or os.environ.get("WANDB_MODE") == "disabled":
        return None
    import wandb
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", tracking.get("project", "worldbridge4d")),
        entity=os.environ.get("WANDB_ENTITY", tracking.get("entity")),
        name=os.environ.get("WANDB_NAME", "pointodyssey-dense4d-ddp-30k"),
        group=os.environ.get("WANDB_GROUP", tracking.get("group", "pointodyssey-dense4d-ddp")),
        job_type="train", tags=["pointodyssey", "dense4d", "ddp", "3gpu"], config=config,
        mode=os.environ.get("WANDB_MODE", "online"),
    )
    run.define_metric("global_step")
    run.define_metric("train/*", step_metric="global_step")
    print(f"WANDB_RUN_URL: {run.url}", flush=True)
    return run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    rank, world, local = rank_info()
    device = torch.device(f"cuda:{local}")
    seed = int(config.get("seed", 2029))
    random.seed(seed + rank); np.random.seed(seed + rank); torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    ensure_index_and_stats(config, rank)
    dataset = PointOdysseyDataset(config["cache_root"], "train")
    stats_file = np.load(Path(config["cache_root"]) / "stats" / "coordinate_stats_train_source.npz")
    mean, scale = stats_file["mean"].astype(np.float32), stats_file["scale"].astype(np.float32)
    shard = ensure_latent_shard(dataset, config, rank, world, device)
    if rank == 0:
        print(f"POINTODYSSEY_LATENT_CACHE_READY: {shard.parent}", flush=True)
    clean_latents = load_latents(config, len(dataset), world)
    model = build_real_model(config, device)
    ddp = DDP(model, device_ids=[local], output_device=local, broadcast_buffers=False)
    groups = parameter_groups(model, config)
    optimizer = torch.optim.AdamW(groups, weight_decay=float(config.get("weight_decay", 1e-4)))
    dtype = precision_dtype(config.get("precision", "bf16"))
    run = init_wandb(config, rank)
    batch = int(config.get("batch_size_per_gpu", config.get("batch_size", 1)))
    steps = int(config.get("steps", 30000))
    prefetch = GeometryPrefetcher(dataset, mean, scale,
                                  int(config.get("geometry_prefetch_workers", 8)),
                                  int(config.get("geometry_prefetch_depth", 2)))
    started = time.perf_counter(); first_loss = None; last_loss = None
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True) if rank == 0 else None
    checkpoint_every = int(config.get("checkpoint_every_steps", 5000))
    for step in range(steps):
        indices, source, target = plan(len(dataset), step, batch, world, seed)
        prefetch.submit(step, indices, source)
        # Maintain a bounded lookahead without changing the deterministic plan.
        if step + 1 < steps:
            ni, ns, _ = plan(len(dataset), step + 1, batch, world, seed)
            prefetch.submit(step + 1, ni, ns)
        target_xyz_np, valid_np = prefetch.get(step)
        source_t = torch.from_numpy(source).to(device, non_blocking=True)
        target_t = torch.from_numpy(target).to(device, non_blocking=True)
        target_xyz = torch.from_numpy(target_xyz_np).to(device, non_blocking=True)
        valid = torch.from_numpy(valid_np).to(device, non_blocking=True)
        latent = clean_latents[torch.from_numpy(indices)].to(device, dtype=dtype, non_blocking=True)
        apply_linear_warmup(optimizer, step + 1, int(config.get("warmup_steps", 256)))
        with torch.autocast(device_type="cuda", dtype=dtype):
            prediction, z4d, _ = ddp(latent, source_t, target_t)
            loss = masked_pair_smooth_l1(prediction.float(), target_xyz.float(), valid,
                                         beta=float(config.get("smooth_l1_beta", 0.05)))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip", 1.0)))
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        value = float(loss.detach()); first_loss = value if first_loss is None else first_loss; last_loss = value
        if rank == 0 and (step == 0 or (step + 1) % int(config.get("diagnostic_every_steps", 20)) == 0 or step + 1 == steps):
            with torch.no_grad():
                epe = torch.linalg.vector_norm((prediction.float() - target_xyz) * torch.as_tensor(scale, device=device).view(1, 1, 3, 1, 1), dim=2)
                epe = float(epe[valid].mean()) if bool(valid.any()) else float("nan")
            payload = {"global_step": step + 1, "train/loss": value, "train/epe_m": epe,
                       "train/clips_seen": (step + 1) * batch * world,
                       "system/peak_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                       "timing/elapsed_seconds": time.perf_counter() - started}
            print(json.dumps(payload), flush=True)
            if run is not None: run.log(payload, step=step + 1)
        if rank == 0 and checkpoint_every and (step + 1) % checkpoint_every == 0:
            save_checkpoint(out / "checkpoint.pt", model, config, mean, scale,
                            extra={"steps": steps, "total_steps": step + 1, "dataset": "PointOdyssey"},
                            optimizer=optimizer, training_state={"global_step": step + 1, "optimizer_updates": step + 1})
    prefetch.close(); dist.barrier()
    if rank == 0:
        save_checkpoint(out / "checkpoint.pt", model, config, mean, scale,
                        extra={"steps": steps, "total_steps": steps, "dataset": "PointOdyssey"},
                        optimizer=optimizer, training_state={"global_step": steps, "optimizer_updates": steps})
        result = {"dataset": "PointOdyssey", "world_size": world, "steps": steps,
                  "batch_size_per_gpu": batch, "global_batch_size": batch * world,
                  "clips": len(dataset), "first_loss": first_loss, "final_loss": last_loss,
                  "elapsed_seconds": time.perf_counter() - started,
                  "peak_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                  "checkpoint": str(out / "checkpoint.pt"), "wandb_url": run.url if run else None}
        (out / "train_metrics.json").write_text(json.dumps(result, indent=2))
        if run is not None:
            run.summary.update(result); run.finish()
        print(json.dumps(result, indent=2), flush=True); print("POINTODYSSEY_DDP_TRAIN_OK", flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
