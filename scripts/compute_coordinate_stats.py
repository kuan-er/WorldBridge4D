#!/usr/bin/env python3
"""Compute train-only XYZ mean/scale with bounded parallel TFRecord workers."""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def chunk_moments(job):
    data_root, clip_length, clip_start, max_examples, start, end, seed = job
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    from worldbridge.data import MOViFDataset
    from worldbridge.geometry import GeometryBuilder

    dataset = MOViFDataset(data_root, "train", clip_length, clip_start, max_examples, seed)
    total = np.zeros(3, np.float64)
    total2 = np.zeros(3, np.float64)
    count = 0
    for index in range(start, end):
        pointmap, valid = GeometryBuilder(dataset[index]).pointmaps()
        values = pointmap[valid].astype(np.float64, copy=False)
        count += len(values)
        total += values.sum(axis=0, dtype=np.float64)
        total2 += np.square(values, dtype=np.float64).sum(axis=0)
    return start, end, count, total, total2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/dataset/MOVi-F")
    parser.add_argument("--output", required=True)
    parser.add_argument("--clip-length", type=int, default=21)
    parser.add_argument("--clip-start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    # Constructing the index uses TFDS metadata and does not decode records.
    from worldbridge.data import MOViFDataset
    dataset = MOViFDataset(args.data_root, "train", args.clip_length, args.clip_start, args.max_examples, args.seed)
    count = 0
    total = np.zeros(3, np.float64)
    total2 = np.zeros(3, np.float64)
    workers = max(1, min(int(args.workers), len(dataset)))
    edges = np.linspace(0, len(dataset), workers + 1, dtype=np.int64)
    jobs = [
        (args.data_root, args.clip_length, args.clip_start, args.max_examples,
         int(edges[i]), int(edges[i + 1]), args.seed)
        for i in range(workers)
    ]
    completed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for start, end, local_count, local_total, local_total2 in pool.map(chunk_moments, jobs):
            count += local_count; total += local_total; total2 += local_total2
            completed += end - start
            print(f"COORDINATE_STATS_PROGRESS {completed}/{len(dataset)}", flush=True)
    if count == 0:
        raise RuntimeError("no valid train coordinates")
    mean = total / count
    scale = np.sqrt(np.maximum(total2 / count - mean * mean, 1e-6))
    output = pathlib.Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, mean=mean.astype(np.float32), scale=scale.astype(np.float32),
                 examples=len(dataset), point_count=count, workers=workers,
                 clip_length=args.clip_length, clip_start=args.clip_start)
    temporary.replace(output)
    print({"output": str(output), "examples": len(dataset), "point_count": count,
           "workers": workers, "mean": mean.tolist(), "scale": scale.tolist()}, flush=True)
    print(f"COORDINATE_STATS_CREATED: {output}", flush=True)


if __name__ == "__main__":
    main()
