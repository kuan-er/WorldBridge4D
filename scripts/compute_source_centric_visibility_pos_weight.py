#!/usr/bin/env python3
"""Compute E5's fixed visibility BCE pos_weight from the exact train schedule."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from worldbridge.data import MOViFDataset
from worldbridge.dense4d_data import CoordinateStats
from worldbridge.dense4d_prefetch import SourceCentricPrefetcher, make_source_centric_plan
from train_dense4d import load_or_create_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    dataset = MOViFDataset(
        config["data_root"], split="train", clip_length=21,
        clip_start=int(config.get("clip_start", 0)), max_examples=None, seed=int(config["seed"]),
    )
    samples = load_or_create_samples(dataset, config)
    stats = CoordinateStats.from_npz(config["coordinate_stats"])
    positive = 0
    negative = 0
    start_time = time.perf_counter()
    steps = int(config["steps"])
    with SourceCentricPrefetcher(samples, stats, workers=8, queue_depth=2) as prefetcher:
        for step in range(min(2, steps)):
            prefetcher.submit(make_source_centric_plan(len(samples), int(config["batch_size"]), 21, step))
        for step in range(steps):
            batch, _ = prefetcher.next()
            source = batch.plan.source
            target = batch.plan.target
            off_diagonal = source != target
            mask = batch.valid & off_diagonal[:, :, None, None]
            positive += int((batch.visible & mask).sum())
            negative += int(((~batch.visible) & mask).sum())
            next_step = step + 2
            if next_step < steps:
                prefetcher.submit(make_source_centric_plan(len(samples), int(config["batch_size"]), 21, next_step))
            if (step + 1) % 100 == 0:
                print(json.dumps({"step": step + 1, "positive": positive, "negative": negative}), flush=True)
    if positive == 0:
        raise RuntimeError("fixed schedule contains no positive off-diagonal visibility pixels")
    pos_weight = negative / positive
    result = {
        "protocol": "h004_source_centric_ablation_screen",
        "schedule": {
            "steps": steps, "batch_size": int(config["batch_size"]), "clips": len(samples),
            "source_rule": "(clip_index + visit_index) % 21", "targets": list(range(21)),
            "seed": int(config["seed"]),
        },
        "positive_visible_valid_off_diagonal": positive,
        "negative_occluded_valid_off_diagonal": negative,
        "pos_weight": pos_weight,
        "elapsed_seconds": time.perf_counter() - start_time,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    print("SOURCE_CENTRIC_POS_WEIGHT_OK", flush=True)


if __name__ == "__main__":
    main()
