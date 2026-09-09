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


def pipeline_work_for_rank(gathered_requests: list[list[tuple[int, str, int]]],
                           rank: int, world: int) -> list[tuple[int, str, int]]:
    """Deduplicate globally and balance VAE work with a stable clip hash."""
    earliest: dict[tuple[str, int], int] = {}
    for requests in gathered_requests:
        for step, name, index in requests:
            key = (str(name), int(index))
            earliest[key] = min(int(step), earliest.get(key, int(step)))
    owned = []
    for (name, index), step in earliest.items():
        if lazy_latent_owner(name, index, world) == int(rank):
            owned.append((step, name, index))
    return sorted(owned, key=lambda value: (value[0], DATASET_NAMES.index(value[1]), value[2]))


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


class LazyVAEPipeline:
    """Background VAE producer sharing a GPU with training on its own stream."""

    def __init__(self, config: dict[str, Any], work: list[tuple[int, str, int]],
                 device: torch.device, rank: int, output: Path) -> None:
        self.config = config
        self.work = list(work)
        self.device = device
        self.rank = int(rank)
        self.output = output
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.error: BaseException | None = None
        self.position = 0
        self.counts = {
            "assigned": len(self.work), "cached": 0, "written": 0,
            "reused_after_wait": 0,
        }
        checkpoint, self.checksum = vae_checkpoint_identity(config)
        # A separate dataset instance prevents the producer's RGB reads from
        # racing the geometry reader's mutable per-dataset caches.
        self.datasets = load_training_datasets(config, allow_missing_latents=True)
        set_lazy_vae_identity(self.datasets, self.checksum)
        self.encoder = WanVAEEncoder(
            checkpoint, device=device, dtype=torch.float32,
            expected_shape=WAN_LATENT_SHAPE_256,
        )
        self.stream = torch.cuda.Stream(device=device)
        self.stream.wait_stream(torch.cuda.current_stream(device))

    def _produce(self, step: int, name: str, index: int) -> None:
        dataset = self.datasets[name]
        if dataset.latent_cached(index):
            self.counts["cached"] += 1
            return
        rgb = torch.from_numpy(np.asarray(dataset.rgb(index))).permute(0, 3, 1, 2)[None]
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            value = self.encoder(rgb).float().cpu().numpy()[0]
        if dataset.cache_latent(index, value, self.checksum):
            self.counts["written"] += 1
        else:
            self.counts["reused_after_wait"] += 1
        print(json.dumps({
            "event": "lazy_vae_pipeline_clip", "rank": self.rank,
            "needed_by_step": step, "dataset": name, "index": index,
            "position": self.position + 1, **self.counts,
        }), flush=True)

    def warm_through(self, exclusive_step: int) -> None:
        while self.position < len(self.work) and self.work[self.position][0] < int(exclusive_step):
            self._produce(*self.work[self.position])
            self.position += 1
        print(json.dumps({
            "event": "lazy_vae_pipeline_warm", "rank": self.rank,
            "exclusive_step": int(exclusive_step), "remaining": len(self.work) - self.position,
            **self.counts,
        }), flush=True)

    def _record_error(self, exc: BaseException) -> None:
        self.error = exc
        value = {"rank": self.rank, "type": type(exc).__name__, "message": str(exc)}
        try:
            atomic_json(self.output / f"pipeline_error_rank{self.rank}.json", value)
        finally:
            print(json.dumps({"event": "lazy_vae_pipeline_error", **value}), flush=True)

    def _run(self) -> None:
        try:
            while self.position < len(self.work) and not self.stop_event.is_set():
                self._produce(*self.work[self.position])
                self.position += 1
            if self.position == len(self.work):
                self.stream.synchronize()
                self.encoder = None
                print(json.dumps({
                    "event": "lazy_vae_pipeline_complete", "rank": self.rank,
                    **self.counts,
                }), flush=True)
        except BaseException as exc:
            self._record_error(exc)

    def start(self) -> None:
        self.thread = threading.Thread(
            target=self._run, name=f"lazy-vae-rank-{self.rank}", daemon=True,
        )
        self.thread.start()

    def wait_for(self, dataset: Any, name: str, indices: list[int],
                 timeout_seconds: float) -> None:
        deadline = time.monotonic() + float(timeout_seconds)
        missing = set(map(int, indices))
        while missing:
            missing = {index for index in missing if not dataset.latent_cached(index)}
            if not missing:
                return
            if self.error is not None:
                raise RuntimeError("local lazy VAE pipeline failed") from self.error
            errors = sorted(self.output.glob("pipeline_error_rank*.json"))
            if errors:
                raise RuntimeError(f"lazy VAE pipeline failed: {errors[0].read_text().strip()}")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for pipeline latent {name}/{sorted(missing)}"
                )
            time.sleep(0.05)

    def snapshot(self) -> dict[str, int]:
        return {**self.counts, "position": self.position, "remaining": len(self.work) - self.position}

    def close(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=600)
            if self.thread.is_alive():
                raise RuntimeError("lazy VAE pipeline did not stop within 600 seconds")
        self.encoder = None
