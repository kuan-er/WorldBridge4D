#!/usr/bin/env python3
"""Copy PointOdyssey scene annotations to a /tmp SSD hot cache.

The authoritative anno.npz files remain on NFS; this stage copies the scenes
used by the training split into /tmp so geometry reads hit SSD instead of the
network mount.  Re-runnable and idempotent.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import shutil
import time

TRAIN_JSONL = Path(
    "/data/WorldBridge4D-persistent/worldbridge4d_256_three_dataset_v1/"
    "pointodyssey/splits/train.jsonl"
)
RAW_TRAIN = Path("/dataset/nas0/PointOdyssey/train")
DST = Path("/tmp/worldbridge4d-cache/anno")


def copy_anno(scene: str) -> tuple[str, int, str]:
    src = RAW_TRAIN / scene / "anno.npz"
    dst = DST / f"{scene}.npz"
    if dst.is_file() and dst.stat().st_size == src.stat().st_size:
        return scene, src.stat().st_size, "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return scene, src.stat().st_size, "copied"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    scenes = sorted({
        Path(json.loads(line)["source_scene"]).name
        for line in TRAIN_JSONL.read_text().splitlines() if line.strip()
    })
    print(f"copying {len(scenes)} PointOdyssey scene annotations", flush=True)
    started = time.perf_counter()
    total = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for scene, size, status in pool.map(copy_anno, scenes):
            total += size
            if status == "copied":
                print(f"  {scene}: {size/1e6:.0f} MB", flush=True)
    print(f"anno copy complete: {len(scenes)} scenes, {total/1e9:.1f} GB, {time.perf_counter()-started:.0f}s", flush=True)
    print("ANNO_COPY_OK", flush=True)


if __name__ == "__main__":
    main()
