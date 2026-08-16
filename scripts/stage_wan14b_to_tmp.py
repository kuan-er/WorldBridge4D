#!/usr/bin/env python3
"""Atomically stage the Wan2.1-14B training files from NAS to node-local /tmp."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

FILES = (
    "config.json",
    "configuration.json",
    "diffusion_pytorch_model.safetensors.index.json",
    "diffusion_pytorch_model-00001-of-00006.safetensors",
    "diffusion_pytorch_model-00002-of-00006.safetensors",
    "diffusion_pytorch_model-00003-of-00006.safetensors",
    "diffusion_pytorch_model-00004-of-00006.safetensors",
    "diffusion_pytorch_model-00005-of-00006.safetensors",
    "diffusion_pytorch_model-00006-of-00006.safetensors",
    "Wan2.1_VAE.pth",
)


def copy_and_hash(source: Path, destination: Path, chunk_size: int = 16 << 20) -> str:
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    digest = hashlib.sha256()
    with source.open("rb") as reader, temporary.open("wb") as writer:
        while block := reader.read(chunk_size):
            writer.write(block)
            digest.update(block)
        writer.flush()
        os.fsync(writer.fileno())
    if temporary.stat().st_size != source.stat().st_size:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"staged size mismatch for {source}")
    temporary.replace(destination)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("/dataset/nas0/yejun/Wan2.1-T2V-14B"))
    parser.add_argument("--destination", type=Path, default=Path("/tmp/worldbridge4d-models/Wan2.1-T2V-14B"))
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.destination.resolve()
    missing = [name for name in FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"source checkpoint is incomplete: {missing}")
    required = sum((source / name).stat().st_size for name in FILES)
    destination.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(destination).free
    existing = sum((destination / name).stat().st_size for name in FILES if (destination / name).is_file())
    if free + existing < required + (5 << 30):
        raise OSError(f"insufficient local space: need {required / 2**30:.1f} GiB plus 5 GiB margin")

    records = []
    started = time.time()
    for index, name in enumerate(FILES, 1):
        src, dst = source / name, destination / name
        if dst.is_file() and dst.stat().st_size == src.stat().st_size:
            status, checksum = "reused_size_match", None
        else:
            checksum = copy_and_hash(src, dst)
            status = "copied_and_hashed"
        record = {"name": name, "bytes": src.stat().st_size, "status": status, "sha256": checksum}
        records.append(record)
        print(json.dumps({"event": "wan14b_stage_file", "index": index, "count": len(FILES), **record}), flush=True)

    manifest = {
        "contract": "worldbridge4d.wan14b.local-stage.v1",
        "source": str(source),
        "destination": str(destination),
        "total_bytes": required,
        "elapsed_seconds": time.time() - started,
        "files": records,
    }
    temporary = destination / f".stage_manifest.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(destination / "stage_manifest.json")
    print("WAN14B_STAGE_OK", flush=True)


if __name__ == "__main__":
    main()
