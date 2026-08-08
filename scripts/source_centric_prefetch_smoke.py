#!/usr/bin/env python3
"""Bounded equality smoke for the H004 source-centric geometry pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from worldbridge.data import MOViFDataset
from worldbridge.dense4d_data import CoordinateStats
from worldbridge.dense4d_prefetch import (
    SourceCentricPrefetcher,
    build_source_centric_batch,
    make_source_centric_plan,
    source_centric_loss_weights,
    weighted_masked_pair_smooth_l1,
)


def digest(plan) -> str:
    value = np.concatenate((plan.sample_indices.reshape(-1), plan.source.reshape(-1), plan.target.reshape(-1)))
    return hashlib.sha256(value.tobytes()).hexdigest()


def assert_equal(name: str, first: np.ndarray, second: np.ndarray) -> None:
    if not np.array_equal(first, second):
        difference = float(np.max(np.abs(first.astype(np.float64) - second.astype(np.float64))))
        raise AssertionError(f"{name} differs; max_abs_difference={difference}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--clips", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    dataset = MOViFDataset(
        config["data_root"], split="train", clip_length=21,
        clip_start=int(config.get("clip_start", 0)), max_examples=int(args.clips), seed=2029,
    )
    samples = [dataset[index] for index in range(len(dataset))]
    stats = CoordinateStats.from_npz(config["coordinate_stats"])
    plans = [make_source_centric_plan(len(samples), len(samples), 21, step) for step in range(args.steps)]
    sync_batches = []
    sync_start = time.perf_counter()
    for plan in plans:
        batch = build_source_centric_batch(samples, stats, plan)
        sync_batches.append(batch)
        assert batch.plan.source.shape == (len(samples), 21)
        assert batch.plan.target.shape == (len(samples), 21)
        assert batch.normalized_xyz.shape == (len(samples), 21, 3, 128, 128)
        assert batch.visible.shape == (len(samples), 21, 128, 128)
        assert batch.valid.shape == (len(samples), 21, 128, 128)
    sync_seconds = time.perf_counter() - sync_start

    async_batches = []
    wait_seconds = []
    async_start = time.perf_counter()
    with SourceCentricPrefetcher(samples, stats, workers=8, queue_depth=2) as prefetcher:
        for plan in plans:
            prefetcher.submit(plan)
        for _ in plans:
            batch, wait = prefetcher.next()
            async_batches.append(batch)
            wait_seconds.append(wait)
    async_seconds = time.perf_counter() - async_start

    for index, (sync_batch, async_batch) in enumerate(zip(sync_batches, async_batches)):
        if digest(sync_batch.plan) != digest(async_batch.plan):
            raise AssertionError(f"sampling plan differs at step {index}")
        for name in ("sample_indices", "source", "target"):
            assert_equal(f"step{index}.{name}", getattr(sync_batch.plan, name), getattr(async_batch.plan, name))
        for name in ("normalized_xyz", "metric_xyz", "visible", "valid"):
            assert_equal(f"step{index}.{name}", getattr(sync_batch, name), getattr(async_batch, name))
        torch.manual_seed(424242 + index)
        prediction = torch.randn_like(torch.from_numpy(sync_batch.normalized_xyz))
        weights = torch.from_numpy(source_centric_loss_weights(sync_batch.plan.source, sync_batch.plan.target))
        sync_loss = weighted_masked_pair_smooth_l1(
            prediction, torch.from_numpy(sync_batch.normalized_xyz),
            torch.from_numpy(sync_batch.valid), weights,
        )
        async_loss = weighted_masked_pair_smooth_l1(
            prediction, torch.from_numpy(async_batch.normalized_xyz),
            torch.from_numpy(async_batch.valid), weights,
        )
        torch.testing.assert_close(sync_loss, async_loss, rtol=0.0, atol=0.0)

    source_schedule = [plan.source[:, 0].tolist() for plan in plans]
    result = {
        "clips": len(samples), "steps": args.steps,
        "source_schedule": source_schedule,
        "plan_sha256": [digest(plan) for plan in plans],
        "sync_seconds": sync_seconds, "async_seconds": async_seconds,
        "prefetch_wait_seconds": wait_seconds,
        "shapes": {
            "source": list(sync_batches[0].plan.source.shape),
            "target": list(sync_batches[0].plan.target.shape),
            "xyz": list(sync_batches[0].normalized_xyz.shape),
            "visible": list(sync_batches[0].visible.shape),
            "valid": list(sync_batches[0].valid.shape),
        },
        "exact_target_equality": True,
        "exact_loss_equality": True,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    print("SOURCE_CENTRIC_PREFETCH_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
