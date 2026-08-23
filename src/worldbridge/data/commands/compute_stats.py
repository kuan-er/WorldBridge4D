#!/usr/bin/env python3
"""Train-only 35/30/35 coordinate moments with fixed points per clip."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from worldbridge.data import DATASET_NAMES, load_training_datasets

WEIGHTS = {"kubric": 0.35, "pointodyssey": 0.30, "dynamic_replica": 0.35}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--points-per-clip", type=int, default=4096)
    parser.add_argument("--max-clips-per-dataset", type=int)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    # Coordinate moments use geometry only; allow this prerequisite to run
    # before either compact shards or the lazy VAE tier exists.
    datasets = load_training_datasets(config, allow_missing_latents=True)
    summaries = {}
    for dataset_id, name in enumerate(DATASET_NAMES):
        dataset = datasets[name]
        clip_count = len(dataset)
        if args.max_clips_per_dataset is not None:
            clip_count = min(clip_count, args.max_clips_per_dataset)
        total = np.zeros(3, np.float64)
        total_squared = np.zeros(3, np.float64)
        point_count = 0
        for index in range(clip_count):
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, dataset_id, index]))
            source = int(rng.integers(21))
            xyz, valid = dataset.source_all_targets(index, source)
            diagonal = xyz[source].transpose(1, 2, 0)[valid[source]]
            diagonal = diagonal[np.isfinite(diagonal).all(axis=1)]
            if not len(diagonal):
                continue
            count = min(args.points_per_clip, len(diagonal))
            chosen = rng.choice(len(diagonal), size=count, replace=False)
            values = diagonal[chosen].astype(np.float64, copy=False)
            total += values.sum(axis=0)
            total_squared += np.square(values).sum(axis=0)
            point_count += count
            if (index + 1) % 100 == 0 or index + 1 == clip_count:
                print(json.dumps({"dataset": name, "clips": index + 1, "points": point_count}), flush=True)
        if point_count == 0:
            raise RuntimeError(f"no train-only diagonal points for {name}")
        mean = total / point_count
        second = total_squared / point_count
        summaries[name] = {"mean": mean, "second": second, "point_count": point_count, "clips": clip_count}
    mixture_mean = sum(WEIGHTS[name] * summaries[name]["mean"] for name in DATASET_NAMES)
    mixture_second = sum(WEIGHTS[name] * summaries[name]["second"] for name in DATASET_NAMES)
    mixture_scale = np.sqrt(np.maximum(mixture_second - np.square(mixture_mean), 1e-12))
    output = Path(args.output or config["mixture_coordinate_stats"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    np.savez(
        temporary, mean=mixture_mean.astype(np.float32), scale=mixture_scale.astype(np.float32),
        coordinate_frame=np.array("source"), stats_source=np.array("train_only_fixed_points_per_clip_mixture"),
        points_per_clip=np.int64(args.points_per_clip), seed=np.int64(args.seed),
        dataset_names=np.asarray(DATASET_NAMES), dataset_weights=np.asarray([WEIGHTS[name] for name in DATASET_NAMES]),
        dataset_point_counts=np.asarray([summaries[name]["point_count"] for name in DATASET_NAMES], np.int64),
        dataset_clip_counts=np.asarray([summaries[name]["clips"] for name in DATASET_NAMES], np.int64),
    )
    temporary.replace(output)
    record = {
        "output": str(output), "mean": mixture_mean.tolist(), "scale": mixture_scale.tolist(),
        "weights": WEIGHTS, "points_per_clip": args.points_per_clip,
        "datasets": {name: {"point_count": value["point_count"], "clips": value["clips"]}
                     for name, value in summaries.items()},
    }
    output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    print("THREE_DATASET_256_STATS_READY", flush=True)


if __name__ == "__main__":
    main()
