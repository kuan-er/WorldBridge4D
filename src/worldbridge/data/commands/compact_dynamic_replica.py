#!/usr/bin/env python3
"""Compact Dynamic Replica trajectory archives into per-stream .npz files.

The raw release stores one torch .pth per frame; each .pth embeds a full
[720,1280,3] uint8 image (~2.76 MB) that training never reads.  This stage
extracts only the four geometry fields used by _load_stream, drops the image,
and merges all 294 frames of a stream into a single .npz so the training
reader does one sequential read instead of ~294 small random reads.

Output layout (one file per stream):
    /tmp/worldbridge4d-cache/trajectories/<stream>.npz
with:
    traj_3d_world  [T, N, 3] float32
    traj_2d        [T, N, 2] float32  (only xy; z is unused)
    verts_inds_vis [T, N]     uint8
    instances      [N]        int32   (constant across frames within a stream)
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

DYNAMIC_REPLICA_TRAIN = Path("/dataset/data/Dynamic_dataset/dynamic_stereo/train")
OUTPUT_ROOT = Path("/tmp/worldbridge4d-cache/trajectories")
SPLITS_JSONL = Path(
    "/data/WorldBridge4D-persistent/worldbridge4d_256_three_dataset_v1/"
    "dynamic_replica/splits/train.jsonl"
)


def collect_streams() -> list[str]:
    streams: set[str] = set()
    for line in SPLITS_JSONL.read_text().splitlines():
        if line.strip():
            streams.add(json.loads(line)["stream"])
    return sorted(streams)


def compact_stream(stream: str) -> tuple[str, int, int]:
    trajectory_dir = DYNAMIC_REPLICA_TRAIN / stream / "trajectories"
    paths = sorted(trajectory_dir.glob("*.pth"))
    if not paths:
        raise FileNotFoundError(stream)

    uv, world, visible, instances = [], [], [], None
    expected_points: int | None = None
    for path in paths:
        value = torch.load(path, map_location="cpu", weights_only=True)
        n = int(value["traj_3d_world"].shape[0])
        if expected_points is None:
            expected_points = n
        if n != expected_points:
            raise ValueError(f"track count changed in stream {stream}: {path}")
        uv.append(value["traj_2d"][:, :2].numpy().astype(np.float32))
        world.append(value["traj_3d_world"].numpy().astype(np.float32))
        visible.append(value["verts_inds_vis"].numpy().astype(np.uint8))
        if instances is None:
            instances = value["instances"].numpy().astype(np.int32)

    out = OUTPUT_ROOT / f"{stream}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(f".{stream}.tmp.npz")
    rel_paths = [str(p.relative_to(DYNAMIC_REPLICA_TRAIN)) for p in paths]
    np.savez_compressed(
        temporary,
        traj_3d_world=np.stack(world),
        traj_2d=np.stack(uv),
        verts_inds_vis=np.stack(visible),
        instances=instances,
        paths=json.dumps(rel_paths),
    )
    temporary.replace(out)
    return stream, len(paths), expected_points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    streams = collect_streams()
    print(f"compacting {len(streams)} streams with {args.workers} workers", flush=True)
    started = time.perf_counter()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(compact_stream, s): s for s in streams}
        for future in concurrent.futures.as_completed(futures):
            stream = futures[future]
            try:
                name, frames, points = future.result()
                done += 1
                if done % 20 == 0 or done == len(streams):
                    elapsed = time.perf_counter() - started
                    print(f"compacted {done}/{len(streams)} streams, {elapsed:.0f}s elapsed", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"error {stream}: {exc}", flush=True)
    print("TRAJECTORY_COMPACT_OK", flush=True)


if __name__ == "__main__":
    main()
