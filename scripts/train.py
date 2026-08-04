#!/usr/bin/env python3
"""Train Compact or Full geometry-supervised deterministic 4D latent baseline."""
from __future__ import annotations
import argparse
import datetime
import itertools
import json
import math
import os
import pathlib
import subprocess
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from worldbridge.data import MOViFDataset
from worldbridge.losses import balanced_query_loss
from worldbridge.models import WorldLatentModel
from worldbridge.pipeline import encode_sample, query_tensors, sample_balanced_queries, train_coordinate_stats


def load_config(path):
    cfg = yaml.safe_load(pathlib.Path(path).read_text())
    cfg["config_path"] = str(path)
    return cfg


def make_model(cfg):
    m = cfg["model"]
    return WorldLatentModel(
        cfg["mode"], latent_channels=m["latent_channels"], latent_time=m["latent_time"],
        latent_height=m["latent_height"], latent_width=m["latent_width"],
        trajectory_channels=m["trajectory_channels"], trajectory_hidden=m["trajectory_hidden"],
        trajectory_time_dim=m.get("trajectory_time_dim", 8),
        reconstruction_channels=m.get("reconstruction_channels", 32),
        context_hidden=m["context_hidden"], decoder_hidden=m["decoder_hidden"],
    )


def init_distributed():
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA in this experiment")
        torch.cuda.set_device(local_rank)
        # Rank 0 computes train-only coordinate statistics before the first
        # collective. Full MOVi-F can exceed NCCL's default 10-minute timeout;
        # rank 1 must be allowed to wait for the broadcast.
        dist.init_process_group(
            backend="nccl", init_method="env://", timeout=datetime.timedelta(hours=12),
        )
    return world_size, rank, local_rank


def init_wandb(cfg, rank):
    tracking = cfg.get("tracking", {})
    if rank != 0 or not tracking.get("enabled", False) or os.getenv("WANDB_MODE", "online") == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("tracking.enabled=true but wandb is not installed") from exc
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", tracking.get("project", "worldbridge4d")),
        entity=os.getenv("WANDB_ENTITY", tracking.get("entity")),
        group=os.getenv("WANDB_GROUP", tracking.get("group", "worldbridge4d-full-vs-compact")),
        job_type="train",
        name=os.getenv("WANDB_NAME", f"{cfg['mode']}-full-train-{os.getenv('PRL_RUN_ID', 'local')}"),
        tags=list(tracking.get("tags", [])) + [cfg["mode"], "full-movi-f", "ddp"],
        config=cfg,
        reinit="return_previous",
    )
    print(f"WANDB_RUN_URL: {run.url}", flush=True)
    return run


class DistributedTrainStep(nn.Module):
    """Keep geometry encoding and decoder inside one DDP forward.

    This matters because Full uses bounded calls to the trajectory encoder and
    the training script queries ``model.decoder`` after encoding. Wrapping this
    complete step, rather than only the model's ordinary forward, guarantees
    that every parameter participates in the same DDP reduction.
    """

    def __init__(self, model, device, block, depth_tolerance, depth_relative_tolerance,
                 mode, num_queries):
        super().__init__()
        self.model = model
        self.device = device
        self.block = block
        self.depth_tolerance = depth_tolerance
        self.depth_relative_tolerance = depth_relative_tolerance
        self.mode = mode
        self.num_queries = num_queries

    def forward(self, sample, qseed):
        z, geom = encode_sample(
            self.model, sample, self.device, self.block,
            self.depth_tolerance, self.depth_relative_tolerance,
        )
        query = sample_balanced_queries(
            geom, self.mode, self.num_queries, np.random.default_rng(int(qseed)),
        )
        source, target, uv01, target_x, valid, groups = query_tensors(query, self.device)
        pred_norm = self.model.decoder(z, source, uv01, target, sample.num_frames)
        pred = self.model.denormalize_coordinates(pred_norm)[0]
        loss, parts = balanced_query_loss(pred, target_x, valid, groups)
        return loss, parts, tuple(z.shape), int(valid.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir")
    ap.add_argument("--steps", type=int, help="Global clip updates; DDP divides these across ranks")
    ap.add_argument("--resume")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.steps is not None:
        cfg["train"]["steps"] = args.steps

    world_size, rank, local_rank = init_distributed()
    distributed = world_size > 1
    seed = int(cfg.get("seed", 0))
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() and cfg["train"].get("device", "auto") != "cpu" else "cpu"
    )
    if device.type == "cuda":
        # With a single visible GPU, some CUDA 12 builds reject a logical
        # device object after CUDA_VISIBLE_DEVICES remapping; the current
        # device is unambiguous here.
        torch.cuda.set_device(local_rank)
        torch.cuda.reset_peak_memory_stats()

    run = init_wandb(cfg, rank)
    data = cfg["data"]
    ds = MOViFDataset(
        data["root"], "train", data["clip_length"], data.get("clip_start", 0),
        data.get("max_examples"), seed,
    )
    stats_examples = data.get("stats_examples")
    if rank == 0 and run is not None:
        run.log({"train/epoch": 0.0, "train/progress": 0.0, "train/phase_coordinate_stats": 1}, step=0)
    if rank == 0:
        stats_samples = ds if stats_examples is None else itertools.islice(ds, int(stats_examples))
        mean, scale = train_coordinate_stats(
            stats_samples, data["depth_tolerance"], data["depth_relative_tolerance"],
        )
    else:
        mean = np.zeros(3, np.float32)
        scale = np.ones(3, np.float32)
    if distributed:
        stats = torch.tensor(np.concatenate([mean, scale])).to(device)
        dist.broadcast(stats, src=0)
        mean, scale = stats[:3].cpu().numpy(), stats[3:].cpu().numpy()
        dist.barrier()
    if rank == 0 and run is not None:
        run.log({"train/epoch": 0.0, "train/progress": 0.0, "train/phase_coordinate_stats": 0}, step=0)

    model = make_model(cfg).to(device)
    model.set_coordinate_stats(torch.from_numpy(mean), torch.from_numpy(scale))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0)),
    )
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt["step"])

    amp = bool(cfg["train"].get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    accum = int(cfg["train"].get("gradient_accumulation", 1))
    num_queries = int(cfg["train"]["queries_per_step"])
    block = int(cfg["train"].get("trajectory_block_size", 16384))
    fixed_queries = bool(cfg["train"].get("fixed_queries", False))
    global_steps = int(cfg["train"]["steps"])
    local_steps = math.ceil(global_steps / world_size)

    # Deterministic DistributedSampler-style padding. Each clip is used once
    # for the requested full pass; only the final rank may receive one padded
    # duplicate when 5737 is not divisible by two.
    global_indices = [i % len(ds) for i in range(global_steps)]
    padded = global_indices + [0] * ((-len(global_indices)) % world_size)
    local_indices = padded[rank::world_size]
    if len(local_indices) != local_steps:
        raise AssertionError((len(local_indices), local_steps))

    step_model = DistributedTrainStep(
        model, device, block, data["depth_tolerance"], data["depth_relative_tolerance"],
        cfg["mode"], num_queries,
    )
    if distributed:
        step_model = DDP(step_model, device_ids=[local_rank], output_device=local_rank)

    first_loss = None
    last_loss = None
    optimizer.zero_grad(set_to_none=True)
    total_queries = 0
    started = time.perf_counter()
    step_model.train()
    for local_step in range(start_step, local_steps):
        global_position = local_step * world_size + rank
        sample = ds[local_indices[local_step]]
        qseed = seed + 1000 + (global_position if not fixed_queries else local_indices[local_step])
        with torch.amp.autocast("cuda", enabled=amp):
            loss, parts, latent_shape, valid_count = step_model(sample, qseed)
            scaled_loss = loss / accum
        scaler.scale(scaled_loss).backward()
        if (local_step + 1) % accum == 0 or local_step + 1 == local_steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"].get("gradient_clip", 1.0))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        value = float(loss.detach())
        first_loss = value if first_loss is None else first_loss
        last_loss = value
        total_queries += num_queries
        if rank == 0 and (local_step == start_step or (local_step + 1) % int(cfg["train"].get("log_every", 5)) == 0 or local_step + 1 == local_steps):
            record = {
                "step": local_step + 1, "global_position": global_position,
                "loss": value, "groups": parts, "latent_shape": list(latent_shape),
                "valid_queries": valid_count, "device": str(device), "world_size": world_size,
            }
            print(json.dumps(record), flush=True)
            if run is not None:
                processed = min((local_step + 1) * world_size, global_steps)
                run.log(
                    {"train/loss": value, "train/valid_queries": valid_count,
                     "train/epoch": processed / len(ds), "train/progress": processed / global_steps,
                     **{f"train/{k}": v for k, v in parts.items()}},
                    step=local_step + 1,
                )

    if distributed:
        dist.barrier()
    elapsed = time.perf_counter() - started
    ratio = float(last_loss / max(first_loss, 1e-12))
    out = pathlib.Path(cfg["output_dir"])
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        git_commit = os.getenv("PRL_GIT_COMMIT") or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip()
        summary = {
            "mode": cfg["mode"], "seed": seed, "steps": global_steps,
            "local_steps": local_steps, "world_size": world_size,
            "first_loss": first_loss, "final_loss": last_loss, "loss_ratio": ratio,
            "elapsed_seconds": elapsed,
            "train_queries_per_second": total_queries * world_size / elapsed,
            "peak_gpu_memory_mb": (torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0),
            "parameters": sum(p.numel() for p in model.parameters()),
            "coordinate_mean": mean.tolist(), "coordinate_scale": scale.tolist(),
            "device": str(device), "torch": torch.__version__, "git_commit": git_commit,
            "prl_run_id": os.getenv("PRL_RUN_ID"), "command": sys.argv, "config": cfg,
        }
        checkpoint = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "step": global_steps, "config": cfg, "summary": summary,
        }
        ckpt_path = out / "checkpoint.pt"
        torch.save(checkpoint, ckpt_path)
        restored = make_model(cfg).to(device)
        restored.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model"])
        summary["checkpoint_restore_verified"] = True
        if run is not None:
            summary["wandb_url"] = run.url
            run.summary.update({f"train/{k}": v for k, v in summary.items() if isinstance(v, (int, float, str, bool))})
            run.finish()
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        pathlib.Path("artifacts").mkdir(exist_ok=True)
        pathlib.Path("artifacts/checkpoint.complete").write_text(str(ckpt_path) + "\n")
        print(json.dumps(summary, sort_keys=True), flush=True)
        print(f"CHECKPOINT_CREATED: {ckpt_path}", flush=True)
        required = cfg["train"].get("require_loss_ratio")
        if required is not None and ratio > float(required):
            raise RuntimeError(f"overfit loss ratio {ratio:.4f} > required {required}")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
