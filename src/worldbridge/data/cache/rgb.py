"""Memory-mapped uint8 source-RGB shards."""
from __future__ import annotations

from collections import OrderedDict
import fcntl
import json
import os
from pathlib import Path
import re
import threading

import numpy as np

from ..constants import DATASET_NAMES

class RGBUInt8ShardStore:
    """Bounded mmap LRU over audited NTHWC uint8 RGB shards.

    Startup validates the cache contract, completion marker, exact shard
    coverage, NPY headers, and checksum-sidecar structure.  It deliberately
    does not reread the 83 GiB payload to recompute shard SHA-256 values.
    """

    CONTRACT = "worldbridge4d_rgb_uint8_256_v1"
    CLIP_SHAPE = (21, 256, 256, 3)

    def __init__(self, root: str | Path, dataset: str, count: int,
                 max_open_shards: int = 16) -> None:
        self.root = Path(root)
        self.dataset = str(dataset)
        self.count = int(count)
        self.max_open_shards = max(1, int(max_open_shards))
        if self.dataset not in DATASET_NAMES or self.count < 1:
            raise ValueError("invalid RGB shard dataset/count")
        contract_path = self.root / "contract.json"
        progress_path = self.root / "progress.json"
        if not contract_path.is_file() or not progress_path.is_file():
            raise FileNotFoundError(
                f"RGB cache metadata missing under {self.root}"
            )
        contract = json.loads(contract_path.read_text())
        progress = json.loads(progress_path.read_text())
        expected_contract = {
            "contract": self.CONTRACT,
            "dtype": "uint8",
            "layout": "NTHWC",
            "clip_shape": list(self.CLIP_SHAPE),
            "compression": "none",
            "shard_size": 64,
        }
        mismatches = {
            key: (contract.get(key), value)
            for key, value in expected_contract.items()
            if contract.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"RGB cache contract mismatch: {mismatches}")
        dataset_contract = contract.get("indexes", {}).get(self.dataset, {})
        if int(dataset_contract.get("clips", -1)) != self.count:
            raise RuntimeError(
                f"RGB cache count mismatch for {self.dataset}: "
                f"{dataset_contract.get('clips')} != {self.count}"
            )
        dataset_progress = progress.get("datasets", {}).get(self.dataset, {})
        if progress.get("status") != "complete" \
                or int(progress.get("clips_complete", -1)) != int(progress.get("clips_total", -2)) \
                or int(dataset_progress.get("clips_complete", -1)) != self.count \
                or int(dataset_progress.get("clips_total", -2)) != self.count:
            raise RuntimeError(f"RGB cache is incomplete for {self.dataset}")
        self.shard_size = int(contract["shard_size"])
        self.directory = self.root / self.dataset / "train"
        if not self.directory.is_dir():
            raise FileNotFoundError(self.directory)
        expected_paths: dict[Path, tuple[int, int]] = {}
        for first in range(0, self.count, self.shard_size):
            end = min(first + self.shard_size, self.count)
            path = self.directory / f"shard_{first:06d}_{end:06d}.npy"
            expected_paths[path] = (first, end)
        actual_paths = set(self.directory.glob("shard_*.npy"))
        if actual_paths != set(expected_paths):
            missing = sorted(str(path.name) for path in set(expected_paths) - actual_paths)
            extra = sorted(str(path.name) for path in actual_paths - set(expected_paths))
            raise RuntimeError(
                f"RGB shard coverage mismatch for {self.dataset}; "
                f"missing={missing[:4]}, extra={extra[:4]}"
            )
        temporaries = list(self.directory.glob("*.tmp*"))
        if temporaries:
            raise RuntimeError(f"RGB cache has temporary files: {temporaries[:4]}")
        for path, (first, end) in expected_paths.items():
            sidecar = path.with_suffix(path.suffix + ".sha256")
            if not sidecar.is_file():
                raise FileNotFoundError(sidecar)
            record = sidecar.read_text().strip()
            if not re.fullmatch(rf"[0-9a-f]{{64}}  {re.escape(path.name)}", record):
                raise RuntimeError(f"invalid RGB checksum sidecar: {sidecar}")
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            expected_shape = (end - first, *self.CLIP_SHAPE)
            if tuple(value.shape) != expected_shape or value.dtype != np.uint8 \
                    or not value.flags.c_contiguous:
                raise RuntimeError(
                    f"RGB shard header mismatch in {path}: "
                    f"{value.shape}/{value.dtype}/contiguous={value.flags.c_contiguous}"
                )
            if isinstance(value, np.memmap):
                value._mmap.close()
        self.paths = expected_paths
        self.cache: OrderedDict[Path, np.ndarray] = OrderedDict()
        self.lock = threading.RLock()

    def _path(self, index: int) -> tuple[Path, int]:
        index = int(index)
        if not 0 <= index < self.count:
            raise IndexError(f"RGB index {index} outside [0,{self.count})")
        first = index // self.shard_size * self.shard_size
        end = min(first + self.shard_size, self.count)
        return self.directory / f"shard_{first:06d}_{end:06d}.npy", index - first

    def _open(self, path: Path) -> np.ndarray:
        with self.lock:
            value = self.cache.pop(path, None)
            if value is None:
                value = np.load(path, mmap_mode="r", allow_pickle=False)
            self.cache[path] = value
            while len(self.cache) > self.max_open_shards:
                self.cache.popitem(last=False)
            return value

    def source_rgb(self, index: int, source: int) -> np.ndarray:
        source = int(source)
        if not 0 <= source < self.CLIP_SHAPE[0]:
            raise ValueError(f"source must be in [0,20], got {source}")
        path, local = self._path(index)
        # Copy exactly one 192-KiB frame so a returned batch never depends on
        # the mmap remaining in the LRU after another worker opens a shard.
        return np.array(self._open(path)[local, source], dtype=np.uint8, copy=True)

    def clip(self, index: int) -> np.ndarray:
        path, local = self._path(index)
        return np.array(self._open(path)[local], dtype=np.uint8, copy=True)
