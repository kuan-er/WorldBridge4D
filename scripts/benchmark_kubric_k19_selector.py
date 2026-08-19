#!/usr/bin/env python3
"""Matched real-cache A/B benchmark for the Kubric K19 selector repair.

The baseline uses the live route's two-sample/eight-shard cache and constructs
full geometry for every candidate source.  The candidate uses the optimized
metadata selector, 32-sample cache, and lazy mappings for every shard.  Both
routes use identical clips/source orders, audit exact selected geometry, and
copy the same 19 target tensors to the leased CUDA device.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import worldbridge.training256 as training256
from worldbridge.training256 import MOViF256Dataset, source_with_eligible_targets


def process_io() -> tuple[int, int, int, int]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    read_bytes = 0
    read_calls = 0
    try:
        for line in Path("/proc/self/io").read_text().splitlines():
            key, value = line.split(":", 1)
            if key == "read_bytes":
                read_bytes = int(value)
            elif key == "syscr":
                read_calls = int(value)
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return int(usage.ru_minflt), int(usage.ru_majflt), read_bytes, read_calls


def delta(after: tuple[int, ...], before: tuple[int, ...]) -> dict[str, int]:
    names = ("minor_faults", "major_faults", "read_bytes", "read_calls")
    return {name: int(right - left) for name, left, right in zip(names, before, after)}


def old_select(dataset: MOViF256Dataset, index: int, sources: np.ndarray,
               min_targets: int) -> tuple[int, np.ndarray, np.ndarray, int]:
    for attempts, source in enumerate(sources, start=1):
        xyz, valid = dataset.source_all_targets(index, int(source))
        eligible = valid.reshape(valid.shape[0], -1).any(axis=1)
        if int(eligible.sum()) >= min_targets:
            return int(source), xyz, valid, attempts
    raise ValueError(f"clip index {index} has no source with {min_targets} eligible targets")


def cuda_consume(xyz: np.ndarray, valid: np.ndarray, targets: int) -> float:
    eligible = np.flatnonzero(valid.reshape(valid.shape[0], -1).any(axis=1))
    if len(eligible) < targets:
        raise AssertionError("selected source lost K19 target capacity")
    chosen = eligible[:targets]
    started = time.perf_counter()
    xyz_cuda = torch.from_numpy(np.ascontiguousarray(xyz[chosen])).to("cuda")
    valid_cuda = torch.from_numpy(np.ascontiguousarray(valid[chosen])).to("cuda")
    # Exercise a minimal consumer so synchronization includes the transfer.
    checksum = torch.nan_to_num(xyz_cuda[:, :, ::32, ::32]).sum()
    checksum = checksum + valid_cuda[:, ::32, ::32].sum()
    if not bool(torch.isfinite(checksum)):
        raise AssertionError("non-finite CUDA checksum")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    del xyz_cuda, valid_cuda, checksum
    return elapsed


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GPU6 benchmark")
    training256.KUBRIC_GEOMETRY_CACHE = args.metadata_root
    common = dict(
        raw_root=args.raw_root,
        cache_root=args.cache_root,
        allow_missing_latents=True,
        geometry_mmap_root=args.mmap_root,
    )
    baseline = MOViF256Dataset(
        **common, geometry_sample_cache_size=2,
        geometry_mmap_max_open_shards=8,
    )
    candidate = MOViF256Dataset(
        **common, geometry_sample_cache_size=32,
        geometry_mmap_max_open_shards=None,
    )
    if len(baseline) != len(candidate):
        raise AssertionError("dataset lengths differ")

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(baseline), size=args.count, replace=False)
    source_orders = [rng.permutation(21).astype(np.int64) for _ in indices]
    torch.empty(1, device="cuda").add_(1)
    torch.cuda.synchronize()

    records: dict[str, list[dict[str, Any]]] = {"baseline": [], "candidate": []}
    selected_sources: list[int] = []
    attempts: list[int] = []
    for position, (index_value, sources) in enumerate(zip(indices, source_orders)):
        index = int(index_value)

        def run_baseline() -> tuple[int, np.ndarray, np.ndarray, int, dict[str, Any]]:
            before = process_io(); started = time.perf_counter()
            source, xyz, valid, tried = old_select(
                baseline, index, sources, args.targets,
            )
            transfer = cuda_consume(xyz, valid, args.targets)
            elapsed = time.perf_counter() - started
            metrics: dict[str, Any] = {
                "index": index, "source": source, "attempts": tried,
                "seconds": elapsed, "cuda_transfer_seconds": transfer,
                **delta(process_io(), before),
            }
            return source, xyz, valid, tried, metrics

        def run_candidate() -> tuple[int, np.ndarray, np.ndarray, dict[str, Any]]:
            before = process_io(); started = time.perf_counter()
            source, xyz, valid = source_with_eligible_targets(
                candidate, index, sources, args.targets,
            )
            transfer = cuda_consume(xyz, valid, args.targets)
            elapsed = time.perf_counter() - started
            metrics: dict[str, Any] = {
                "index": index, "source": source,
                "seconds": elapsed, "cuda_transfer_seconds": transfer,
                **delta(process_io(), before),
            }
            return source, xyz, valid, metrics

        # Alternating order balances the unavoidable shared page-cache warmup.
        if position % 2:
            new_source, new_xyz, new_valid, new_metrics = run_candidate()
            old_source, old_xyz, old_valid, tried, old_metrics = run_baseline()
        else:
            old_source, old_xyz, old_valid, tried, old_metrics = run_baseline()
            new_source, new_xyz, new_valid, new_metrics = run_candidate()
        if old_source != new_source:
            raise AssertionError(
                f"selected source mismatch at index {index}: {old_source} != {new_source}"
            )
        if not np.array_equal(old_valid, new_valid):
            raise AssertionError(f"validity mismatch at index {index}")
        if not np.array_equal(old_xyz, new_xyz, equal_nan=True):
            raise AssertionError(f"XYZ mismatch at index {index}")
        old_metrics["executed_first"] = position % 2 == 0
        new_metrics["executed_first"] = position % 2 == 1
        records["baseline"].append(old_metrics)
        records["candidate"].append(new_metrics)
        selected_sources.append(old_source)
        attempts.append(tried)
        print(json.dumps({
            "event": "k19_selector_progress", "done": position + 1,
            "count": args.count, "index": index, "attempts": tried,
            "baseline_seconds": old_metrics["seconds"],
            "candidate_seconds": new_metrics["seconds"],
        }), flush=True)

    baseline_times = [float(row["seconds"]) for row in records["baseline"]]
    candidate_times = [float(row["seconds"]) for row in records["candidate"]]
    old_summary = summary(baseline_times)
    new_summary = summary(candidate_times)
    result: dict[str, Any] = {
        "event": "kubric_k19_selector_benchmark",
        "exact": True,
        "seed": args.seed,
        "samples": args.count,
        "targets_per_source": args.targets,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "baseline": old_summary,
        "candidate": new_summary,
        "mean_speedup": old_summary["mean"] / new_summary["mean"],
        "p50_speedup": old_summary["p50"] / new_summary["p50"],
        "p95_speedup": old_summary["p95"] / new_summary["p95"],
        "source_attempts": {
            "mean": float(statistics.mean(attempts)),
            "max": max(attempts),
            "more_than_one": sum(value > 1 for value in attempts),
            "histogram": {str(value): attempts.count(value) for value in sorted(set(attempts))},
        },
        "selected_sources": selected_sources,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path,
                        default=Path("/dataset/nas0/yejun/MOVi-F/512x512"))
    parser.add_argument("--cache-root", type=Path, default=Path(
        "/data/WorldBridge4D-persistent/worldbridge4d_256_three_dataset_v1/kubric"))
    parser.add_argument("--metadata-root", type=Path,
                        default=Path("/tmp/worldbridge4d-cache/kubric_geometry"))
    parser.add_argument("--mmap-root", type=Path,
                        default=Path("/tmp/worldbridge4d-cache-v2/kubric_geometry_mmap"))
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--targets", type=int, default=19)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.count <= 256:
        raise ValueError("count must be in [1,256]")
    if not 1 <= args.targets <= 21:
        raise ValueError("targets must be in [1,21]")
    result = benchmark(args)
    print(json.dumps({key: result[key] for key in (
        "event", "exact", "samples", "mean_speedup", "p50_speedup",
        "p95_speedup", "source_attempts",
    )}, indent=2), flush=True)
    print("KUBRIC_K19_SELECTOR_BENCHMARK_OK", flush=True)


if __name__ == "__main__":
    main()
