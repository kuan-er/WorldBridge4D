"""Deterministic sparse latent planning and concurrent Wan-VAE production."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
import torch

from ..data.constants import DATASET_NAMES
from ..data.factory import load_training_datasets
from ..data.sampling import deterministic_sample_plan
from ..models.wan import WAN_LATENT_SHAPE_256, WanVAEEncoder
from ..utils.io import atomic_json, sha256
from .schedulers import dataset_for_step

def required_latent_requests(datasets: dict[str, Any], seed: int, start_step: int,
                             target_steps: int, rank: int, accumulation: int,
                             microbatch_per_gpu: int = 1,
                             dataset_mix_counts=None
                             ) -> list[tuple[int, str, int]]:
    """Return rank-local requests in update order, including repeated clips."""
    requests = []
    slots_per_rank = int(accumulation) * int(microbatch_per_gpu)
    for step in range(int(start_step), int(target_steps)):
        name = dataset_for_step(step, seed, dataset_mix_counts)
        dataset = datasets[name]
        for slot in range(slots_per_rank):
            index, _, _ = deterministic_sample_plan(
                dataset, name, seed, step, slot, rank, slots_per_rank
            )
            requests.append((step, name, int(index)))
    return requests


def required_latent_indices(datasets: dict[str, Any], seed: int, start_step: int,
                            target_steps: int, rank: int, accumulation: int,
                            microbatch_per_gpu: int = 1,
                            dataset_mix_counts=None
                            ) -> dict[str, list[int]]:
    required: dict[str, set[int]] = {name: set() for name in DATASET_NAMES}
    for _step, name, index in required_latent_requests(
        datasets, seed, start_step, target_steps, rank, accumulation,
        microbatch_per_gpu, dataset_mix_counts,
    ):
        required[name].add(index)
    return {name: sorted(indices) for name, indices in required.items()}


def lazy_latent_owner(name: str, index: int, world: int) -> int:
    """Assign a dataset clip to one rank with a stable, balanced hash."""
    dataset_id = DATASET_NAMES.index(str(name))
    return (int(index) * 1_315_423_911 + dataset_id * 2_654_435_761) % int(world)


def vae_checkpoint_identity(config: dict[str, Any]) -> tuple[Path, str]:
    checkpoint = Path(config.get(
        "vae_checkpoint", Path(config["wan_root"]) / "Wan2.1_VAE.pth"
    )).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"WAN VAE checkpoint missing: {checkpoint}")
    return checkpoint, sha256(checkpoint)


def set_lazy_vae_identity(datasets: dict[str, Any], checksum: str) -> None:
    for dataset in datasets.values():
        dataset.set_lazy_vae_sha256(checksum)


def warm_lazy_latents(config: dict[str, Any], datasets: dict[str, Any],
                      required: dict[str, list[int]], device: torch.device,
                      rank: int) -> dict[str, int]:
    """Populate only this rank's planned cache misses, then release the VAE."""
    checkpoint, checksum = vae_checkpoint_identity(config)
    set_lazy_vae_identity(datasets, checksum)
    missing = [(name, index) for name in DATASET_NAMES for index in required[name]
               if not datasets[name].latent_cached(index)]
    counts = {"required": sum(len(x) for x in required.values()), "misses": len(missing),
              "written": 0, "reused_after_wait": 0}
    if not missing:
        print(json.dumps({"event": "lazy_vae_warmup", "rank": rank, **counts}), flush=True)
        return counts
    # FP32 is the canonical offline contract. The VAE is deleted before the
    # trainable 1.3B model/FSDP/optimizer are constructed, avoiding co-residency.
    encoder = WanVAEEncoder(
        checkpoint, device=device, dtype=torch.float32,
        expected_shape=WAN_LATENT_SHAPE_256,
    )
    for name, index in missing:
        dataset = datasets[name]
        if dataset.latent_cached(index):
            counts["reused_after_wait"] += 1
            continue
        rgb = torch.from_numpy(np.asarray(dataset.rgb(index))).permute(0, 3, 1, 2)[None]
        with torch.inference_mode():
            value = encoder(rgb).float().cpu().numpy()[0]
        if dataset.cache_latent(index, value, checksum):
            counts["written"] += 1
        else:
            counts["reused_after_wait"] += 1
        print(json.dumps({
            "event": "lazy_vae_clip", "rank": rank, "dataset": name,
            "index": index, "written": counts["written"], "misses": len(missing),
        }), flush=True)
    del encoder
    torch.cuda.empty_cache()
    print(json.dumps({"event": "lazy_vae_warmup", "rank": rank, **counts}), flush=True)
    return counts
