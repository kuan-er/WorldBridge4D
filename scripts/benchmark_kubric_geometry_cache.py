#!/usr/bin/env python3
"""Compare compact-NPZ and mmap Kubric sample loading with exactness checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldbridge.geometry import GeometryBuilder
from worldbridge.training256 import KubricGeometryMmapStore, MOViF256Dataset


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, np.float64), q))


def sample_equal(left, right) -> bool:
    fields = (
        "depth", "depth_valid", "segmentation", "camera_positions",
        "camera_quaternions", "instance_positions", "instance_quaternions",
        "instance_dynamic", "instance_visibility", "depth_range",
    )
    return all(np.array_equal(getattr(left, field), getattr(right, field)) for field in fields)


def benchmark(source: Path, mmap_root: Path, count: int, geometry_count: int,
              seed: int) -> dict[str, object]:
    store = KubricGeometryMmapStore(mmap_root)
    if count < 1 or count > store.count:
        raise ValueError(f"count must be in [1,{store.count}]")
    rng = np.random.default_rng(seed)
    indices = rng.choice(store.count, size=count, replace=False).tolist()
    old_times: list[float] = []
    new_times: list[float] = []
    geometry_old: list[float] = []
    geometry_new: list[float] = []
    geometry_checked = 0
    for position, index in enumerate(indices):
        path = source / f"geom_{index:06d}.npz"

        def load_old():
            started = time.perf_counter()
            value = MOViF256Dataset._load_compact_sample(path)
            return value, time.perf_counter() - started

        def load_new():
            started = time.perf_counter()
            value = MOViF256Dataset._load_compact_sample(path, store.read(index))
            return value, time.perf_counter() - started

        # Alternate order to avoid systematically giving either route a warm
        # compact-metadata page-cache advantage.
        if position % 2:
            new, new_elapsed = load_new(); old, old_elapsed = load_old()
        else:
            old, old_elapsed = load_old(); new, new_elapsed = load_new()
        if not sample_equal(old, new):
            raise AssertionError(f"sample mismatch at raw index {index}")
        old_times.append(old_elapsed)
        new_times.append(new_elapsed)
        if position < geometry_count:
            source_frame = int(rng.integers(0, 21))
            started = time.perf_counter()
            old_geometry = GeometryBuilder(old).trajectory_block(
                source_frame, coordinate_frame="source", compute_visibility=True,
            )
            geometry_old.append(time.perf_counter() - started)
            started = time.perf_counter()
            new_geometry = GeometryBuilder(new).trajectory_block(
                source_frame, coordinate_frame="source", compute_visibility=True,
            )
            geometry_new.append(time.perf_counter() - started)
            if any(not np.array_equal(a, b) for a, b in zip(old_geometry, new_geometry)):
                raise AssertionError(f"geometry mismatch at raw index {index}, source {source_frame}")
            geometry_checked += 1
    old_mean, new_mean = statistics.mean(old_times), statistics.mean(new_times)
    result: dict[str, object] = {
        "event": "kubric_mmap_benchmark",
        "samples": count,
        "geometry_samples": geometry_checked,
        "exact": True,
        "old_load_mean_seconds": old_mean,
        "new_load_mean_seconds": new_mean,
        "load_mean_speedup": old_mean / new_mean,
        "old_load_p50_seconds": percentile(old_times, 0.50),
        "new_load_p50_seconds": percentile(new_times, 0.50),
        "old_load_p95_seconds": percentile(old_times, 0.95),
        "new_load_p95_seconds": percentile(new_times, 0.95),
        "load_p95_speedup": percentile(old_times, 0.95) / percentile(new_times, 0.95),
    }
    if geometry_old:
        result.update({
            "old_geometry_mean_seconds": statistics.mean(geometry_old),
            "new_geometry_mean_seconds": statistics.mean(geometry_new),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path,
        default=Path("/tmp/worldbridge4d-cache/kubric_geometry"),
    )
    parser.add_argument(
        "--mmap-root", type=Path,
        default=Path("/tmp/worldbridge4d-cache-v2/kubric_geometry_mmap"),
    )
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--geometry-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--minimum-mean-speedup", type=float, default=1.15)
    parser.add_argument("--minimum-p95-speedup", type=float, default=1.30)
    args = parser.parse_args()
    result = benchmark(args.source, args.mmap_root, args.count, args.geometry_count, args.seed)
    print(json.dumps(result, indent=2), flush=True)
    if float(result["load_mean_speedup"]) < args.minimum_mean_speedup:
        raise RuntimeError(
            f"mean speedup {result['load_mean_speedup']:.3f} is below "
            f"{args.minimum_mean_speedup:.3f}"
        )
    if float(result["load_p95_speedup"]) < args.minimum_p95_speedup:
        raise RuntimeError(
            f"p95 speedup {result['load_p95_speedup']:.3f} is below "
            f"{args.minimum_p95_speedup:.3f}"
        )
    print("KUBRIC_MMAP_BENCHMARK_OK", flush=True)


if __name__ == "__main__":
    main()
