"""Wan latent shard and sparse cache stores."""
from __future__ import annotations

from collections import OrderedDict
import fcntl
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np

class LatentShardStore:
    """Bounded read-only shard LRU supporting both existing naming schemes."""
    def __init__(self, root: str | Path, expected_shape: tuple[int, ...] = (16, 6, 32, 32),
                 max_shards: int = 2) -> None:
        self.root = Path(root)
        self.expected_shape = tuple(expected_shape)
        self.max_shards = max(1, int(max_shards))
        self.entries: list[tuple[int, int, Path, str]] = []
        self.cache: OrderedDict[Path, np.ndarray] = OrderedDict()
        for path in sorted(self.root.glob("*.safetensors")):
            match = re.fullmatch(r"shard[-_](\d+)[-_](\d+)", path.stem)
            if not match:
                continue
            first, second = map(int, match.groups())
            if "-" in path.stem:
                count = second - first + 1
                key = "latent"
            else:
                count = second
                key = "latents"
            self.entries.append((first, count, path, key))
        if not self.entries:
            raise FileNotFoundError(f"no latent shards under {self.root}")
        self.entries.sort()

    def __getitem__(self, global_index: int) -> np.ndarray:
        from safetensors import safe_open
        index = int(global_index)
        for first, count, path, key in self.entries:
            if first <= index < first + count:
                value = self.cache.pop(path, None)
                if value is None:
                    with safe_open(str(path), framework="np") as handle:
                        value = handle.get_tensor(key)
                    if tuple(value.shape[1:]) != self.expected_shape:
                        raise RuntimeError(f"latent shape mismatch in {path}: {value.shape}")
                self.cache[path] = value
                while len(self.cache) > self.max_shards:
                    self.cache.popitem(last=False)
                return np.asarray(value[index - first], dtype=np.float32)
        raise IndexError(f"latent index {index} is not covered by {self.root}")

class LazyLatentCache:
    """Per-clip read-through cache with process locks and atomic publication."""
    CONTRACT = "wan2.1_vae_posterior_mean_fp32_256_v1"

    def __init__(self, root: str | Path, dataset: str,
                 expected_shape: tuple[int, ...] = (16, 6, 32, 32)) -> None:
        self.root = Path(root)
        self.dataset = str(dataset)
        self.expected_shape = tuple(expected_shape)
        self.expected_vae_sha256: str | None = None

    def path(self, index: int) -> Path:
        return self.root / f"latent_{int(index):08d}.safetensors"

    def set_vae_sha256(self, value: str) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", str(value)):
            raise ValueError("VAE SHA-256 must be 64 lowercase hex characters")
        self.expected_vae_sha256 = str(value)

    def read(self, index: int, clip_id: str) -> np.ndarray:
        from safetensors import safe_open
        path = self.path(index)
        if not path.is_file():
            raise FileNotFoundError(path)
        with safe_open(str(path), framework="np") as handle:
            metadata = handle.metadata() or {}
            if metadata.get("contract") != self.CONTRACT:
                raise RuntimeError(f"lazy latent contract mismatch: {path}")
            if metadata.get("dataset") != self.dataset or metadata.get("clip_id") != str(clip_id):
                raise RuntimeError(f"lazy latent identity mismatch: {path}")
            if metadata.get("index") != str(int(index)):
                raise RuntimeError(f"lazy latent index mismatch: {path}")
            if self.expected_vae_sha256 is not None and metadata.get("vae_sha256") != self.expected_vae_sha256:
                raise RuntimeError(f"lazy latent VAE checksum mismatch: {path}")
            value = handle.get_tensor("latent")
        if tuple(value.shape) != self.expected_shape or value.dtype != np.float32:
            raise RuntimeError(f"lazy latent shape/dtype mismatch in {path}: {value.shape}/{value.dtype}")
        if not np.isfinite(value).all():
            raise RuntimeError(f"non-finite lazy latent: {path}")
        return np.asarray(value, dtype=np.float32)

    def write(self, index: int, clip_id: str, value: np.ndarray,
              vae_sha256: str) -> bool:
        """Write once; return True only when this process produced the file."""
        from safetensors.numpy import save_file
        value = np.asarray(value, dtype=np.float32)
        if tuple(value.shape) != self.expected_shape or not np.isfinite(value).all():
            raise ValueError(f"invalid lazy latent for {self.dataset}/{index}: {value.shape}")
        self.set_vae_sha256(vae_sha256)
        path = self.path(index)
        lock_path = self.root / ".locks" / f"latent_{int(index):08d}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.root.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if path.is_file():
                self.read(index, clip_id)
                return False
            temporary = path.with_suffix(f".{os.getpid()}.tmp.safetensors")
            save_file({"latent": np.ascontiguousarray(value)}, str(temporary), metadata={
                "contract": self.CONTRACT, "dataset": self.dataset,
                "clip_id": str(clip_id), "index": str(int(index)),
                "vae_sha256": vae_sha256,
            })
            temporary.replace(path)
            self.read(index, clip_id)
            return True
