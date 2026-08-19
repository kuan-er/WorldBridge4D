#!/usr/bin/env python3
"""Matched real-Kubric benchmark for bounded geometry prefetch pipelines.

Three Latin-square rounds rotate which route sees each fresh clip set first:

* baseline: 8 geometry workers, 16 tasks in flight, sample LRU 2, 8 mmap shards;
* conservative: 4 workers, 8 in flight, sample LRU 16, all shards mapped;
* staged: one serial sample loader feeding 4 compute workers through a bounded
  16-clip ring, sample LRU 16, all shards mapped.

Every route uses the same clips/source order, requires K19, copies identical
outputs to CUDA, and verifies full-array SHA-256 equality.
"""
from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import time
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import worldbridge.training256 as training256
from worldbridge.data import MOViSample
from worldbridge.training256 import MOViF256Dataset


ROUTE_ORDERS = (
    ("baseline", "conservative", "staged"),
    ("conservative", "staged", "baseline"),
    ("staged", "baseline", "conservative"),
)


def io_snapshot() -> tuple[int, int, int]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    read_bytes = 0
    try:
        for line in Path("/proc/self/io").read_text().splitlines():
            key, value = line.split(":", 1)
            if key == "read_bytes":
                read_bytes = int(value)
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return int(usage.ru_minflt), int(usage.ru_majflt), read_bytes


def io_delta(before: tuple[int, ...], after: tuple[int, ...]) -> dict[str, int]:
    return {
        name: int(right - left)
        for name, left, right in zip(
            ("minor_faults", "major_faults", "read_bytes"), before, after,
        )
    }


def select_from_sample(sample: MOViSample, sources: np.ndarray, targets: int
                       ) -> tuple[int, np.ndarray, np.ndarray, int]:
    for attempts, source in enumerate(sources, start=1):
        xyz, valid, _ = MOViF256Dataset._geometry_from_sample(
            sample, int(source), False,
        )
        eligible = valid.reshape(valid.shape[0], -1).any(axis=1)
        if int(eligible.sum()) >= targets:
            return int(source), xyz, valid, attempts
    raise ValueError(f"sample has no source with {targets} eligible targets")


def load_and_compute(dataset: MOViF256Dataset, index: int, sources: np.ndarray,
                     targets: int) -> dict[str, Any]:
    started = time.perf_counter()
    load_started = time.perf_counter()
    sample = dataset.sample(index)
    load_seconds = time.perf_counter() - load_started
    compute_started = time.perf_counter()
    source, xyz, valid, attempts = select_from_sample(sample, sources, targets)
    return {
        "index": index, "source": source, "attempts": attempts,
        "xyz": xyz, "valid": valid, "load_seconds": load_seconds,
        "compute_seconds": time.perf_counter() - compute_started,
        "task_seconds": time.perf_counter() - started,
    }


def staged_compute(load_future: Future[tuple[MOViSample, float]], index: int,
                   sources: np.ndarray, targets: int) -> dict[str, Any]:
    started = time.perf_counter()
    sample, load_seconds = load_future.result()
    compute_started = time.perf_counter()
    source, xyz, valid, attempts = select_from_sample(sample, sources, targets)
    return {
        "index": index, "source": source, "attempts": attempts,
        "xyz": xyz, "valid": valid, "load_seconds": load_seconds,
        "compute_seconds": time.perf_counter() - compute_started,
        "task_seconds": time.perf_counter() - started,
    }


def timed_sample(dataset: MOViF256Dataset, index: int) -> tuple[MOViSample, float]:
    started = time.perf_counter()
    return dataset.sample(index), time.perf_counter() - started


def consume_cuda(result: dict[str, Any], targets: int) -> tuple[float, str]:
    xyz = result.pop("xyz")
    valid = result.pop("valid")
    eligible = np.flatnonzero(valid.reshape(valid.shape[0], -1).any(axis=1))
    chosen = eligible[:targets]
    if len(chosen) != targets:
        raise AssertionError("selected geometry lost K19 capacity")
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(xyz).view(np.uint8))
    digest.update(np.ascontiguousarray(valid).view(np.uint8))
    started = time.perf_counter()
    xyz_cuda = torch.from_numpy(np.ascontiguousarray(xyz[chosen])).to("cuda")
    valid_cuda = torch.from_numpy(np.ascontiguousarray(valid[chosen])).to("cuda")
    check = torch.nan_to_num(xyz_cuda[:, :, ::32, ::32]).sum()
    check = check + valid_cuda[:, ::32, ::32].sum()
    if not bool(torch.isfinite(check)):
        raise AssertionError("non-finite CUDA checksum")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    del xyz_cuda, valid_cuda, check, xyz, valid
    return elapsed, digest.hexdigest()


def direct_route(dataset: MOViF256Dataset, plans: list[tuple[int, np.ndarray]],
                 targets: int, workers: int, inflight: int
                 ) -> tuple[list[dict[str, Any]], float]:
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures: dict[int, Future[dict[str, Any]]] = {}
        next_submit = 0
        while next_submit < min(inflight, len(plans)):
            index, sources = plans[next_submit]
            futures[next_submit] = pool.submit(
                load_and_compute, dataset, index, sources, targets,
            )
            next_submit += 1
        for position in range(len(plans)):
            result = futures.pop(position).result()
            transfer, digest = consume_cuda(result, targets)
            result["cuda_seconds"] = transfer
            result["digest"] = digest
            records.append(result)
            if next_submit < len(plans):
                index, sources = plans[next_submit]
                futures[next_submit] = pool.submit(
                    load_and_compute, dataset, index, sources, targets,
                )
                next_submit += 1
    return records, time.perf_counter() - started


def staged_route(dataset: MOViF256Dataset, plans: list[tuple[int, np.ndarray]],
                 targets: int, compute_workers: int, inflight: int
                 ) -> tuple[list[dict[str, Any]], float]:
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as loader, ThreadPoolExecutor(
        max_workers=compute_workers,
    ) as compute:
        futures: dict[int, Future[dict[str, Any]]] = {}
        next_submit = 0

        def submit(position: int) -> None:
            index, sources = plans[position]
            loaded = loader.submit(timed_sample, dataset, index)
            futures[position] = compute.submit(
                staged_compute, loaded, index, sources, targets,
            )

        while next_submit < min(inflight, len(plans)):
            submit(next_submit); next_submit += 1
        for position in range(len(plans)):
            result = futures.pop(position).result()
            transfer, digest = consume_cuda(result, targets)
            result["cuda_seconds"] = transfer
            result["digest"] = digest
            records.append(result)
            if next_submit < len(plans):
                submit(next_submit); next_submit += 1
    return records, time.perf_counter() - started


def numeric_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def make_dataset(args: argparse.Namespace, cache_size: int,
                 open_shards: int | None) -> MOViF256Dataset:
    return MOViF256Dataset(
        args.raw_root, args.cache_root, allow_missing_latents=True,
        geometry_mmap_root=args.mmap_root,
        geometry_sample_cache_size=cache_size,
        geometry_mmap_max_open_shards=open_shards,
    )


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    training256.KUBRIC_GEOMETRY_CACHE = args.metadata_root
    datasets = {
        "baseline": make_dataset(args, 2, 8),
        "conservative": make_dataset(args, 16, None),
        "staged": make_dataset(args, 16, None),
    }
    length = len(datasets["baseline"])
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(length, size=args.rounds * args.count, replace=False)
    torch.empty(1, device="cuda").add_(1)
    torch.cuda.synchronize()

    route_records: dict[str, list[dict[str, Any]]] = {
        name: [] for name in datasets
    }
    round_results: list[dict[str, Any]] = []
    for round_index in range(args.rounds):
        values = chosen[round_index * args.count:(round_index + 1) * args.count]
        plans = [
            (int(index), rng.permutation(21).astype(np.int64)) for index in values
        ]
        expected: dict[int, tuple[int, str]] = {}
        order = ROUTE_ORDERS[round_index % len(ROUTE_ORDERS)]
        current: dict[str, Any] = {"round": round_index, "order": list(order)}
        for order_position, route in enumerate(order):
            before = io_snapshot()
            if route == "baseline":
                records, wall = direct_route(
                    datasets[route], plans, args.targets, workers=8, inflight=16,
                )
            elif route == "conservative":
                records, wall = direct_route(
                    datasets[route], plans, args.targets, workers=4, inflight=8,
                )
            else:
                records, wall = staged_route(
                    datasets[route], plans, args.targets,
                    compute_workers=4, inflight=16,
                )
            faults = io_delta(before, io_snapshot())
            for record in records:
                key = int(record["index"])
                identity = (int(record["source"]), str(record["digest"]))
                if key in expected and expected[key] != identity:
                    raise AssertionError(
                        f"route mismatch index={key}: {expected[key]} != {identity}"
                    )
                expected[key] = identity
                record.update({
                    "round": round_index, "order_position": order_position,
                })
            route_records[route].extend(records)
            current[route] = {"wall_seconds": wall, **faults}
            print(json.dumps({
                "event": "geometry_prefetch_progress", "round": round_index,
                "route": route, "order_position": order_position,
                "wall_seconds": wall, **faults,
            }), flush=True)
        current["exact"] = True
        round_results.append(current)

    aggregates: dict[str, Any] = {}
    for route, records in route_records.items():
        walls = [
            float(round_result[route]["wall_seconds"])
            for round_result in round_results
        ]
        cold_walls = [
            float(round_result[route]["wall_seconds"])
            for round_result in round_results
            if round_result["order"][0] == route
        ]
        aggregates[route] = {
            "wall": numeric_summary(walls),
            "cold_first_wall_seconds": cold_walls,
            "task": numeric_summary([float(row["task_seconds"]) for row in records]),
            "load": numeric_summary([float(row["load_seconds"]) for row in records]),
            "compute": numeric_summary([float(row["compute_seconds"]) for row in records]),
            "cuda": numeric_summary([float(row["cuda_seconds"]) for row in records]),
            "attempts": {
                "mean": float(statistics.mean(row["attempts"] for row in records)),
                "max": max(int(row["attempts"]) for row in records),
            },
            "major_faults": sum(int(item[route]["major_faults"]) for item in round_results),
            "read_bytes": sum(int(item[route]["read_bytes"]) for item in round_results),
        }
    result = {
        "event": "geometry_prefetch_pipeline_benchmark",
        "exact": True,
        "seed": args.seed,
        "rounds": args.rounds,
        "clips_per_round": args.count,
        "targets_per_source": args.targets,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device": torch.cuda.get_device_name(0),
        "protocol": {
            "baseline": {"workers": 8, "inflight": 16, "cache": 2, "shards": 8},
            "conservative": {"workers": 4, "inflight": 8, "cache": 16, "shards": "all"},
            "staged": {"loaders": 1, "compute_workers": 4, "ring": 16,
                       "cache": 16, "shards": "all"},
        },
        "round_results": round_results,
        "aggregates": aggregates,
        "records": route_records,
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
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--targets", type=int, default=19)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 9 or not 4 <= args.count <= 64:
        raise ValueError("rounds must be in [1,9] and count in [4,64]")
    result = benchmark(args)
    print(json.dumps({
        "event": result["event"], "exact": result["exact"],
        "aggregates": result["aggregates"],
    }, indent=2), flush=True)
    print("GEOMETRY_PREFETCH_PIPELINE_BENCHMARK_OK", flush=True)


if __name__ == "__main__":
    main()
