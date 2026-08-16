#!/usr/bin/env python3
"""Checksum-verified SSD staging for 256px Wan training cold starts."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import yaml


def sha256(path: Path, chunk_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def source_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "source": str(path.resolve()), "device": stat.st_dev, "inode": stat.st_ino,
        "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def copy_with_sha256(source: Path, temporary: Path,
                     chunk_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as reader, temporary.open("wb") as writer:
        while block := reader.read(chunk_size):
            digest.update(block)
            writer.write(block)
        writer.flush()
        os.fsync(writer.fileno())
    shutil.copystat(source, temporary)
    return digest.hexdigest()


def stage_file(source: str | Path, staging_root: str | Path) -> tuple[Path, str, bool]:
    """Return exact staged object, content SHA-256, and whether it was copied."""
    source = Path(source).resolve()
    root = Path(staging_root).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    objects = root / "objects"
    manifests = root / "manifests"
    locks = root / "locks"
    for directory in (objects, manifests, locks):
        directory.mkdir(parents=True, exist_ok=True)
    source_key = hashlib.sha256(str(source).encode()).hexdigest()
    manifest_path = manifests / f"{source_key}.json"
    lock_path = locks / f"{source_key}.lock"
    identity = source_identity(source)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            candidate = Path(manifest.get("staged", ""))
            candidate_stat = candidate.stat() if candidate.is_file() else None
            source_unchanged = all(
                manifest.get(key) == value for key, value in identity.items()
            )
            # Fast path: the source identity and the staged file's mtime/size
            # are unchanged, so the previously verified SHA-256 still holds.
            # This avoids re-hashing the multi-GiB object on every cold start.
            cached_ok = (
                source_unchanged
                and candidate_stat is not None
                and candidate_stat.st_size == identity["size"]
                and manifest.get("candidate_mtime_ns") == candidate_stat.st_mtime_ns
            )
            if cached_ok or (
                source_unchanged
                and candidate_stat is not None
                and candidate_stat.st_size == identity["size"]
                and sha256(candidate) == manifest.get("sha256")
            ):
                print(json.dumps({
                    "event": "staging_reuse", "source": str(source),
                    "staged": str(candidate), "bytes": identity["size"],
                    "sha256": manifest["sha256"],
                }), file=sys.stderr, flush=True)
                return candidate, str(manifest["sha256"]), False
        free = shutil.disk_usage(root).free
        if free < identity["size"] + (1 << 30):
            raise OSError(
                f"insufficient staging space under {root}: free={free}, need={identity['size'] + (1 << 30)}"
            )
        temporary = objects / f".{source_key}.{os.getpid()}.tmp"
        temporary.unlink(missing_ok=True)
        try:
            digest = copy_with_sha256(source, temporary)
            destination = objects / f"{digest}-{source.name}"
            if destination.is_file():
                if destination.stat().st_size != identity["size"] or sha256(destination) != digest:
                    raise RuntimeError(f"existing staged object failed checksum: {destination}")
                temporary.unlink()
            else:
                temporary.replace(destination)
            if sha256(destination) != digest:
                raise RuntimeError(f"staged copy checksum mismatch: {destination}")
            manifest = {
                **identity, "staged": str(destination), "sha256": digest,
                "candidate_mtime_ns": destination.stat().st_mtime_ns,
            }
            atomic_json(manifest_path, manifest)
        finally:
            temporary.unlink(missing_ok=True)
        print(json.dumps({
            "event": "staging_copy", "source": str(source),
            "staged": str(destination), "bytes": identity["size"], "sha256": digest,
        }), file=sys.stderr, flush=True)
        return destination, digest, True


def stage_training_inputs(config_path: str | Path, staging_root: str | Path,
                          resume: str | Path | None = None) -> tuple[Path, Path | None]:
    config_path = Path(config_path).resolve()
    root = Path(staging_root).resolve()
    config = yaml.safe_load(config_path.read_text())
    wan_root = Path(config["wan_root"])
    dit_source = Path(config.get("wan_checkpoint", wan_root / "diffusion_pytorch_model.safetensors"))
    vae_source = Path(config.get("vae_checkpoint", wan_root / "Wan2.1_VAE.pth"))
    dit, dit_sha, _ = stage_file(dit_source, root)
    vae, vae_sha, _ = stage_file(vae_source, root)
    staged_resume = None
    resume_sha = None
    if resume is not None:
        resume_source = Path(resume).resolve()
        # The resume checkpoint already lives on /data; staging a second copy
        # to the same filesystem wastes 9+ GiB and minutes of HDD IO.
        staged_resume = resume_source
        resume_sha = None
        config["resume_status_path"] = str(resume_source.parent / "train_status.json")
    config["wan_checkpoint"] = str(dit)
    config["vae_checkpoint"] = str(vae)
    config["staged_inputs"] = {
        "source_config": str(config_path), "staging_root": str(root),
        "wan_sha256": dit_sha, "vae_sha256": vae_sha,
        "resume_sha256": resume_sha,
    }
    serialized = yaml.safe_dump(config, sort_keys=False)
    runtime_hash = hashlib.sha256(serialized.encode()).hexdigest()[:20]
    runtime_dir = root / "runtime_configs"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_config = runtime_dir / f"three_dataset_256_{runtime_hash}.yaml"
    if not runtime_config.is_file() or runtime_config.read_text() != serialized:
        temporary = runtime_config.with_suffix(f".{os.getpid()}.tmp.yaml")
        temporary.write_text(serialized)
        temporary.replace(runtime_config)
    return runtime_config, staged_resume


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--staging-root",
        default="/data/WorldBridge4D-persistent/worldbridge4d_staging/checkpoints",
    )
    parser.add_argument("--resume")
    args = parser.parse_args()
    config, resume = stage_training_inputs(args.config, args.staging_root, args.resume)
    # Exactly two stdout lines form a shell-safe interface; diagnostics use stderr.
    print(config)
    print(resume or "")


if __name__ == "__main__":
    main()
