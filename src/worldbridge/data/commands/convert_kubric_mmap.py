#!/usr/bin/env python3
"""Convert compact Kubric geometry NPZs into resumable mmap shards.

The existing NPZ cache remains authoritative and untouched.  Each output
shard is published only after all three NPY files are flushed, fsynced, and
checksummed.  The root manifest is published last, so readers fail closed on
partial conversions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable

import numpy as np

from worldbridge.data.constants import KUBRIC_GEOMETRY_CACHE, KUBRIC_GEOMETRY_MMAP_CACHE

FIELDS = ("depth", "depth_valid", "segmentation")
FORMAT = "worldbridge4d.kubric_geometry_mmap.v1"
DEFAULT_SOURCE = KUBRIC_GEOMETRY_CACHE
DEFAULT_DESTINATION = KUBRIC_GEOMETRY_MMAP_CACHE
DEFAULT_INDEX = Path(
    "/data/WorldBridge4D-persistent/worldbridge4d_256_three_dataset_v1/"
    "kubric/splits/train.jsonl"
)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fsync_directory(path.parent)


def sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def training_indices(index_path: Path | None, source: Path) -> list[int]:
    if index_path is None:
        values = sorted(int(path.stem.split("_")[-1]) for path in source.glob("geom_*.npz"))
    else:
        rows = [json.loads(line) for line in index_path.read_text().splitlines() if line.strip()]
        values = sorted({int(row["raw_index"]) for row in rows})
    if values != list(range(len(values))):
        raise ValueError(
            "Kubric mmap v1 requires a contiguous zero-based raw index; "
            f"got count={len(values)}, first={values[:3]}, last={values[-3:]}"
        )
    missing = [index for index in values if not (source / f"geom_{index:06d}.npz").is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} compact Kubric archives, first={missing[:10]}")
    return values


def inspect_layout(path: Path) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            field: (tuple(archive[field].shape), np.dtype(archive[field].dtype))
            for field in FIELDS
        }


def output_name(destination: Path, shard: int, field: str) -> Path:
    return destination / f"shard_{shard:05d}_{field}.npy"


def marker_name(destination: Path, shard: int) -> Path:
    return destination / f"shard_{shard:05d}.json"


def marker_valid(destination: Path, shard: int, count: int) -> bool:
    marker = marker_name(destination, shard)
    if not marker.is_file():
        return False
    try:
        value = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if value.get("format") != FORMAT or int(value.get("count", -1)) != count:
        return False
    return all(
        output_name(destination, shard, field).is_file()
        and output_name(destination, shard, field).stat().st_size == int(value["files"][field]["bytes"])
        and sha256(output_name(destination, shard, field)) == value["files"][field]["sha256"]
        for field in FIELDS
    )


def remove_incomplete_shard(destination: Path, shard: int) -> None:
    marker_name(destination, shard).unlink(missing_ok=True)
    for field in FIELDS:
        output_name(destination, shard, field).unlink(missing_ok=True)
        for temporary in destination.glob(f"shard_{shard:05d}_{field}.npy.*.tmp"):
            temporary.unlink(missing_ok=True)


def convert_shard(
    source: Path,
    destination: Path,
    indices: list[int],
    shard: int,
    layout: dict[str, tuple[tuple[int, ...], np.dtype]],
) -> dict[str, Any]:
    if marker_valid(destination, shard, len(indices)):
        return json.loads(marker_name(destination, shard).read_text())
    remove_incomplete_shard(destination, shard)
    temporary_paths = {
        field: destination / f"shard_{shard:05d}_{field}.npy.{os.getpid()}.tmp"
        for field in FIELDS
    }
    arrays = {
        field: np.lib.format.open_memmap(
            temporary_paths[field], mode="w+", dtype=layout[field][1],
            shape=(len(indices), *layout[field][0]),
        )
        for field in FIELDS
    }
    try:
        for local, index in enumerate(indices):
            path = source / f"geom_{index:06d}.npz"
            with np.load(path, allow_pickle=False) as archive:
                for field in FIELDS:
                    value = archive[field]
                    shape, dtype = layout[field]
                    if value.shape != shape or value.dtype != dtype:
                        raise ValueError(
                            f"inconsistent {field} in {path}: "
                            f"got {value.shape}/{value.dtype}, expected {shape}/{dtype}"
                        )
                    arrays[field][local] = value
        for array in arrays.values():
            array.flush()
        arrays.clear()
        for path in temporary_paths.values():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        for field, path in temporary_paths.items():
            path.replace(output_name(destination, shard, field))
        fsync_directory(destination)
        files = {
            field: {
                "path": output_name(destination, shard, field).name,
                "bytes": output_name(destination, shard, field).stat().st_size,
                "sha256": sha256(output_name(destination, shard, field)),
            }
            for field in FIELDS
        }
        marker = {
            "format": FORMAT,
            "shard": shard,
            "first_index": indices[0],
            "count": len(indices),
            "files": files,
        }
        atomic_json(marker_name(destination, shard), marker)
        return marker
    except BaseException:
        arrays.clear()
        remove_incomplete_shard(destination, shard)
        raise


def estimate_output_bytes(count: int, layout: dict[str, tuple[tuple[int, ...], np.dtype]]) -> int:
    # NPY headers are less than 4 KiB per field/shard; reserve 1 MiB globally
    # on top of exact array payloads to keep the safety check conservative.
    return count * sum(int(np.prod(shape)) * dtype.itemsize for shape, dtype in layout.values()) + (1 << 20)


def nearest_existing_parent(path: Path) -> Path:
    value = path
    while not value.exists():
        value = value.parent
    return value


def convert(
    source: Path,
    destination: Path,
    indices: Iterable[int],
    shard_size: int = 64,
    min_free_gib: float = 200.0,
) -> dict[str, Any]:
    source, destination = Path(source), Path(destination)
    indices = list(map(int, indices))
    if not indices or indices != list(range(len(indices))):
        raise ValueError("indices must be a non-empty contiguous zero-based sequence")
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    layout = inspect_layout(source / f"geom_{indices[0]:06d}.npz")
    projected = estimate_output_bytes(len(indices), layout)
    existing = sum(path.stat().st_size for path in destination.glob("shard_*.npy")) if destination.exists() else 0
    required = max(0, projected - existing)
    free = shutil.disk_usage(nearest_existing_parent(destination)).free
    minimum = int(float(min_free_gib) * 2**30)
    if free - required < minimum:
        raise OSError(
            f"refusing Kubric mmap conversion: free={free / 2**30:.1f} GiB, "
            f"additional={required / 2**30:.1f} GiB, floor={min_free_gib:.1f} GiB"
        )
    destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    shards = []
    total_shards = (len(indices) + shard_size - 1) // shard_size
    for shard in range(total_shards):
        subset = indices[shard * shard_size:(shard + 1) * shard_size]
        marker = convert_shard(source, destination, subset, shard, layout)
        shards.append(marker)
        print(json.dumps({
            "event": "kubric_mmap_progress", "shard": shard + 1,
            "total_shards": total_shards, "elapsed_seconds": time.perf_counter() - started,
        }), flush=True)
    manifest = {
        "format": FORMAT,
        "complete": True,
        "count": len(indices),
        "shard_size": shard_size,
        "fields": {
            field: {"shape": list(shape), "dtype": dtype.str}
            for field, (shape, dtype) in layout.items()
        },
        "bytes": sum(
            int(value["files"][field]["bytes"])
            for value in shards for field in FIELDS
        ),
        "shards": total_shards,
    }
    atomic_json(destination / "manifest.json", manifest)
    print(json.dumps({"event": "kubric_mmap_complete", **manifest}), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--min-free-gib", type=float, default=200.0)
    args = parser.parse_args()
    values = training_indices(args.index, args.source)
    convert(args.source, args.destination, values, args.shard_size, args.min_free_gib)


if __name__ == "__main__":
    main()
