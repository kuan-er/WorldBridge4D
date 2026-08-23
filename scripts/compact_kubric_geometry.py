#!/usr/bin/env python3
"""Compact Kubric (MOVi-F) geometry inputs into per-clip .npz files on /tmp.

Groups clips by TFRecord shard and reads each shard once (instead of a separate
skip+take per clip), then resizes each clip to the 256 training grid, drops the
RGB frames, and writes the geometry fields into one .npz per clip.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldbridge.data import MOViFDataset
from worldbridge.data.datasets.movif256 import MOViF256Dataset

RAW_ROOT = "/dataset/nas0/yejun/MOVi-F/512x512"
TRAIN_JSONL = Path(
    "/data/WorldBridge4D-persistent/worldbridge4d_256_three_dataset_v1/"
    "kubric/splits/train.jsonl"
)
DST = Path("/tmp/worldbridge4d-cache/kubric_geometry")

_NATIVE: MOViFDataset | None = None


def get_native() -> MOViFDataset:
    global _NATIVE
    if _NATIVE is None:
        _NATIVE = MOViFDataset(RAW_ROOT, split="train", clip_length=21, clip_start=0)
    return _NATIVE


def write_compact(global_index: int, sample) -> int:
    out = DST / f"geom_{int(global_index):06d}.npz"
    if out.is_file():
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(f".{global_index}.tmp.npz")
    np.savez_compressed(
        temporary,
        depth=sample.depth.astype(np.float32),
        depth_valid=sample.depth_valid.astype(np.uint8),
        segmentation=sample.segmentation.astype(np.int32),
        camera_positions=sample.camera_positions.astype(np.float32),
        camera_quaternions=sample.camera_quaternions.astype(np.float32),
        focal_length=np.float32(sample.focal_length),
        sensor_width=np.float32(sample.sensor_width),
        field_of_view=np.float32(sample.field_of_view),
        instance_positions=sample.instance_positions.astype(np.float32),
        instance_quaternions=sample.instance_quaternions.astype(np.float32),
        instance_dynamic=sample.instance_dynamic.astype(np.uint8),
        instance_visibility=sample.instance_visibility.astype(np.uint16),
        depth_range=sample.depth_range.astype(np.float32),
        clip_start=np.int64(sample.clip_start),
    )
    temporary.replace(out)
    return out.stat().st_size


def compact_shard(args) -> tuple[str, int, int]:
    shard_path, entries = args
    native = get_native()
    tf = native._tf()
    # Read the whole shard once; entries reference records by local_index.
    records = list(tf.data.TFRecordDataset([str(shard_path)]))
    written = 0
    bytes_written = 0
    for global_index, local_index in entries:
        raw = bytes(records[local_index].numpy())
        sample = native._decode(raw, global_index, decode_rgb=False)
        resized = MOViF256Dataset._resize_sample(sample, skip_rgb=True)
        bytes_written += write_compact(global_index, resized)
        written += 1
    return str(shard_path), written, bytes_written


def audit_outputs(wanted: set[int], destination: Path = DST) -> dict[str, int]:
    """Fail closed unless every requested hot-cache file was atomically published."""
    expected = [destination / f"geom_{index:06d}.npz" for index in sorted(wanted)]
    missing = [path.name for path in expected if not path.is_file()]
    empty = [path.name for path in expected if path.is_file() and path.stat().st_size == 0]
    temporaries = sorted(path.name for path in destination.glob("*.tmp.npz"))
    if missing or empty or temporaries:
        raise RuntimeError(
            "Kubric compact geometry audit failed: "
            f"expected={len(expected)}, missing={len(missing)}, empty={len(empty)}, "
            f"temporaries={len(temporaries)}, examples="
            f"{(missing + empty + temporaries)[:10]}"
        )
    return {
        "expected": len(expected),
        "files": len(expected),
        "bytes": sum(path.stat().st_size for path in expected),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    native = get_native()
    rows = [json.loads(line) for line in TRAIN_JSONL.read_text().splitlines() if line.strip()]
    wanted = {int(row["raw_index"]) for row in rows}

    # Group wanted clips by shard, one task per shard.
    groups: dict[Path, list[tuple[int, int]]] = {}
    for global_index, (path, local_index) in enumerate(native.records):
        if global_index in wanted:
            groups.setdefault(path, []).append((global_index, local_index))

    tasks = [(path, entries) for path, entries in groups.items()]
    print(f"compacting {len(wanted)} Kubric clips across {len(tasks)} shards with {args.workers} workers", flush=True)

    started = time.perf_counter()
    done_clips = 0
    done_shards = 0
    total_bytes = 0
    errors: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(compact_shard, t): t for t in tasks}
        for future in concurrent.futures.as_completed(futures):
            try:
                _, written, size = future.result()
                done_clips += written
                total_bytes += size
            except Exception as exc:  # noqa: BLE001
                error = f"error shard {futures[future][0]}: {exc}"
                errors.append(error)
                print(error, flush=True)
            done_shards += 1
            if done_shards % 50 == 0 or done_shards == len(tasks):
                print(f"compacted {done_clips}/{len(wanted)} clips ({done_shards}/{len(tasks)} shards), {total_bytes/1e9:.1f} GB, {time.perf_counter()-started:.0f}s", flush=True)
    if errors:
        raise RuntimeError(
            f"Kubric geometry compaction failed in {len(errors)}/{len(tasks)} shards; "
            f"first error: {errors[0]}"
        )
    audit = audit_outputs(wanted)
    print(json.dumps({"event": "kubric_geometry_compact_audit", **audit}), flush=True)
    print("KUBRIC_GEOMETRY_COMPACT_OK", flush=True)


if __name__ == "__main__":
    main()
