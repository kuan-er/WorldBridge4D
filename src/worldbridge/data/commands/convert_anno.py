#!/usr/bin/env python3
"""Convert PointOdyssey anno.npz archives into per-scene .npy files.

The compressed scene-level anno.npz must be fully decompressed (~445 MB) to
read even one frame.  Storing each array as an uncompressed .npy lets the
training adapter mmap it and lazily read only the 21 frames a clip needs.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import time

import numpy as np

TRAIN_JSONL = Path(
    "/data/WorldBridge4D-persistent/worldbridge4d_256_three_dataset_v1/"
    "pointodyssey/splits/train.jsonl"
)
ANNO_NPZ = Path("/tmp/worldbridge4d-cache/anno")
DST = Path("/tmp/worldbridge4d-cache/anno_npy")

KEYS = ("trajs_2d", "trajs_3d", "valids", "visibs", "intrinsics", "extrinsics")


def convert_scene(scene: str) -> tuple[str, str]:
    src = ANNO_NPZ / f"{scene}.npz"
    dst_dir = DST / scene
    if dst_dir.is_dir() and all((dst_dir / f"{k}.npy").is_file() for k in KEYS):
        return scene, "skip"
    dst_dir.mkdir(parents=True, exist_ok=True)
    with np.load(src) as z:
        for k in KEYS:
            np.save(dst_dir / f"{k}.npy", z[k])
    return scene, "converted"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    scenes = sorted(p.stem for p in ANNO_NPZ.glob("*.npz"))
    print(f"converting {len(scenes)} scenes with {args.workers} workers", flush=True)
    started = time.perf_counter()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for scene, status in pool.map(convert_scene, scenes):
            done += 1
            if done % 20 == 0 or done == len(scenes):
                print(f"converted {done}/{len(scenes)}, {time.perf_counter()-started:.0f}s", flush=True)
    print("ANNO_NPY_CONVERT_OK", flush=True)


if __name__ == "__main__":
    main()
