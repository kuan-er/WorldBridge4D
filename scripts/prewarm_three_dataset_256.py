#!/usr/bin/env python3
"""Pre-warm the page cache with the scene-level geometry data sources.

Reads the PointOdyssey scene annotation files once so that subsequent VAE
encoding and per-step geometry computation hit memory instead of the slow NFS
mount.  Best-effort: it only reads, never writes, and any unreadable path is
reported as a warning without failing the run.

Deliberately excluded (measured too large for the ~835G free memory):
  * Dynamic Replica trajectory .pth files: 6,090 clips x 21 frames x ~5.4 MB
    ~ 690 GB total.  These are already stream-cached in-process; see
    dynamic_replica.py `_load_stream` and its `_max_stream_cache`.
  * Kubric TFRecord shards: read via tensorflow's shard iterator.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

TRAIN_JSONL = "/data/WorldBridge4D-persistent/worldbridge4d_256_three_dataset_v1/pointodyssey/splits/train.jsonl"
POINTODYSSEY_RAW = Path("/dataset/nas0/PointOdyssey")


def read_bytes(path: Path) -> int:
    total = 0
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            total += len(chunk)
    return total


def collect_files() -> list[Path]:
    files: set[Path] = set()
    # The PointOdyssey adapter rewrites source_scene to
    # raw_root/train/<scene> in its __init__ (see pointodyssey.py), so mirror
    # that mapping here.
    for line in Path(TRAIN_JSONL).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        scene = POINTODYSSEY_RAW / "train" / Path(row["source_scene"]).name
        files.add(scene / "anno.npz")
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    files = collect_files()
    print(f"prewarming {len(files)} PointOdyssey scene annotations with {args.workers} workers", flush=True)

    total = 0
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(read_bytes, f): f for f in files}
        for future in concurrent.futures.as_completed(futures):
            path = futures[future]
            try:
                total += future.result()
            except OSError as exc:
                print(f"warning: {path}: {exc}", flush=True)
            done += 1
            if done % 20 == 0:
                print(f"prewarmed {done}/{len(files)} files, {total / 1e9:.1f} GB", flush=True)

    print(f"prewarm complete: {len(files)} files, {total / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
