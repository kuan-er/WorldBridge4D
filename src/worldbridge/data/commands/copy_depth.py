#!/usr/bin/env python3
"""Copy PointOdyssey and Dynamic Replica depth images to durable storage.

Training geometry reads one source-frame depth PNG per clip. The authoritative
files stay on NFS/HDD; this stage copies the per-scene/per-stream depth
directories used by the training split into the persistent cache.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import shutil
import threading
import time

from worldbridge.data.constants import PERSISTENT_CACHE_ROOT

PO_TRAIN_JSONL = Path(
    "/data/WorldBridge4D-persistent/worldbridge4d_256_three_dataset_v1/"
    "pointodyssey/splits/train.jsonl"
)
DR_TRAIN_JSONL = Path(
    "/data/WorldBridge4D-persistent/worldbridge4d_256_three_dataset_v1/"
    "dynamic_replica/splits/train.jsonl"
)
PO_RAW = Path("/dataset/nas0/PointOdyssey/train")
DR_RAW = Path("/dataset/data/Dynamic_dataset/dynamic_stereo/train")
DST_ROOT = PERSISTENT_CACHE_ROOT / "depth"


def tree_manifest(root: Path) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(
        (path.relative_to(root).as_posix(), path.stat().st_size)
        for path in root.rglob("*") if path.is_file()
    ))


def copy_tree(src: Path, dst: Path) -> tuple[str, int, int, str]:
    source_manifest = tree_manifest(src)
    if not source_manifest:
        raise ValueError(f"depth source directory is empty: {src}")
    if dst.is_dir() and tree_manifest(dst) == source_manifest:
        return str(src), len(source_manifest), sum(size for _, size in source_manifest), "skip"

    dst.parent.mkdir(parents=True, exist_ok=True)
    temporary = dst.with_name(
        f".{dst.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        shutil.copytree(src, temporary)
        if tree_manifest(temporary) != source_manifest:
            raise RuntimeError(f"depth copy verification failed: {src} -> {temporary}")
        if dst.exists():
            shutil.rmtree(dst)
        temporary.replace(dst)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return str(src), len(source_manifest), sum(size for _, size in source_manifest), "copied"


def collect_dirs() -> tuple[list[tuple[Path, Path]], dict[str, int]]:
    dirs: list[tuple[Path, Path]] = []
    missing: list[Path] = []
    # PointOdyssey: per-scene depths directories.
    scenes = sorted({
        Path(json.loads(line)["source_scene"]).name
        for line in PO_TRAIN_JSONL.read_text().splitlines() if line.strip()
    })
    for scene in scenes:
        src = PO_RAW / scene / "depths"
        dst = DST_ROOT / "pointodyssey" / scene / "depths"
        if src.is_dir():
            dirs.append((src, dst))
        else:
            missing.append(src)
    # Dynamic Replica: per-stream depths directories.
    streams = sorted({
        json.loads(line)["stream"]
        for line in DR_TRAIN_JSONL.read_text().splitlines() if line.strip()
    })
    for stream in streams:
        src = DR_RAW / stream / "depths"
        dst = DST_ROOT / "dynamic_replica" / stream / "depths"
        if src.is_dir():
            dirs.append((src, dst))
        else:
            missing.append(src)
    if missing:
        sample = ", ".join(str(path) for path in missing[:3])
        raise FileNotFoundError(
            f"{len(missing)}/{len(scenes) + len(streams)} depth source directories are missing: {sample}"
        )
    expected = {"pointodyssey": len(scenes), "dynamic_replica": len(streams)}
    if len(dirs) != sum(expected.values()):
        raise RuntimeError(f"depth directory planning mismatch: {len(dirs)} != {expected}")
    return dirs, expected


def atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def run_copy(workers: int) -> None:
    dirs, expected = collect_dirs()
    print(f"copying {len(dirs)} depth directories with {workers} workers", flush=True)
    started = time.perf_counter()
    copied = files = size = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for src, count, num_bytes, status in pool.map(lambda d: copy_tree(*d), dirs):
            files += count
            size += num_bytes
            if status == "copied":
                copied += 1
    missing_outputs = [str(dst) for _, dst in dirs if not dst.is_dir()]
    temporary_dirs = [
        str(path) for path in DST_ROOT.glob("*/*/.depths.tmp.*") if path.is_dir()
    ]
    if missing_outputs or temporary_dirs:
        raise RuntimeError(
            f"depth completion audit failed: missing={missing_outputs[:3]} temporary={temporary_dirs[:3]}"
        )
    elapsed = time.perf_counter() - started
    metadata: dict[str, object] = {
        "version": 1,
        "directories": len(dirs),
        "dataset_directories": expected,
        "files": files,
        "bytes": size,
        "copied_directories": copied,
        "elapsed_seconds": elapsed,
    }
    atomic_json(DST_ROOT / "metadata.json", metadata)
    print(
        f"depth copy complete: {len(dirs)} dirs, {copied} copied, "
        f"{files} files, {size} bytes, {elapsed:.0f}s",
        flush=True,
    )
    print("DEPTH_COPY_OK", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    DST_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = DST_ROOT.parent / ".depth-copy.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        run_copy(args.workers)


if __name__ == "__main__":
    main()
