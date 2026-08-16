#!/usr/bin/env python3
"""Multi-GPU offline VAE latent pre-encoding for the 256px three-dataset route.

Runs the same deterministic per-rank clip enumeration and balanced hash owner
assignment as train_three_dataset_256_fsdp.py's --lazy-vae-cache branch, but
stops after encoding (no FSDP/model construction, no training).  The per-clip
latent cache is shared and idempotent, so this can be run with any world size
and safely resumed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from validate_three_dataset_256_cache_roots import (
    assert_expected_latent_files, validate_cache_roots,
)
from worldbridge.training256 import DATASET_NAMES, load_training_datasets


def _load_train_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "train_three_dataset_256_fsdp",
        ROOT / "scripts" / "train_three_dataset_256_fsdp.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()

    train = _load_train_module()
    config = yaml.safe_load(Path(args.config).read_text())
    # Defense in depth: the launcher checks this before torchrun, and every
    # standalone worker checks again before writing authoritative latents.
    validate_cache_roots(config, create=True)
    rank, world, local, device = train.initialize_distributed()
    seed = int(config.get("seed", 20260812))
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

    if rank == 0:
        train.prepare_training_indexes(config)
    dist.barrier()

    datasets = load_training_datasets(config, allow_missing_latents=True)
    target_steps = int(args.steps if args.steps is not None else config["max_steps"])
    accumulation = int(config["gradient_accumulation"])
    microbatch = int(config["microbatch_per_gpu"])

    local_required = train.required_latent_indices(
        datasets, seed, 0, target_steps, rank, accumulation, microbatch,
    )
    gathered: list[Any] = [None] * world
    dist.all_gather_object(gathered, local_required)

    owned = {name: [] for name in DATASET_NAMES}
    for name in DATASET_NAMES:
        all_indices = sorted({index for item in gathered for index in item[name]})
        for index in all_indices:
            if train.lazy_latent_owner(name, index, world) == rank:
                owned[name].append(index)

    owned_total = sum(len(v) for v in owned.values())
    owned_totals: list[Any] = [None] * world
    dist.all_gather_object(owned_totals, owned_total)
    expected_total = sum(map(int, owned_totals))
    print(json.dumps({
        "event": "precompute_owned", "rank": rank, "world": world,
        "owned": owned_total,
        **{name: len(v) for name, v in owned.items()},
    }), flush=True)

    counts = train.warm_lazy_latents(config, datasets, owned, device, rank)
    dist.barrier()

    if rank == 0:
        roots = validate_cache_roots(config)
        dataset_counts = assert_expected_latent_files(roots, expected_total)
        print(json.dumps({
            "event": "precompute_latents_done", "world_size": world,
            "expected_total": expected_total, "dataset_counts": dataset_counts,
        }), flush=True)
        print("PRECOMPUTE_LATENTS_OK", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
