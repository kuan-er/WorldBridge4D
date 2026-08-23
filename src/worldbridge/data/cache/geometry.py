"""Memory-mapped Kubric geometry shards."""
from __future__ import annotations

from collections import OrderedDict
import json
import math
from pathlib import Path
import threading

import numpy as np

from ..constants import KUBRIC_MMAP_FIELDS

class KubricGeometryMmapStore:
    """Read fixed-size Kubric arrays from atomically published mmap shards.

    Shard mappings are cheap virtual-address reservations.  Keeping all shards
    mapped avoids mmap-lock churn when geometry workers sample random clips;
    physical pages remain demand-paged by the kernel.
    """

    def __init__(self, root: str | Path,
                 max_open_shards: int | None = None) -> None:
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Kubric mmap manifest is missing: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("format") != "worldbridge4d.kubric_geometry_mmap.v1":
            raise ValueError(f"unsupported Kubric mmap manifest: {manifest_path}")
        if not self.manifest.get("complete"):
            raise ValueError(f"incomplete Kubric mmap cache: {manifest_path}")
        self.count = int(self.manifest["count"])
        self.shard_size = int(self.manifest["shard_size"])
        if self.count < 1 or self.shard_size < 1:
            raise ValueError(f"invalid Kubric mmap dimensions: {manifest_path}")
        self.shard_count = math.ceil(self.count / self.shard_size)
        self.max_open_shards = (
            self.shard_count if max_open_shards is None
            else max(1, int(max_open_shards))
        )
        self._shards: OrderedDict[int, tuple[np.ndarray, ...]] = OrderedDict()
        self._lock = threading.RLock()

    def _open_shard(self, shard: int) -> tuple[np.ndarray, ...]:
        with self._lock:
            value = self._shards.pop(shard, None)
            if value is None:
                prefix = self.root / f"shard_{shard:05d}"
                value = tuple(
                    np.load(f"{prefix}_{field}.npy", mmap_mode="r", allow_pickle=False)
                    for field in KUBRIC_MMAP_FIELDS
                )
                expected = min(self.shard_size, self.count - shard * self.shard_size)
                if any(array.shape[0] != expected for array in value):
                    raise ValueError(f"Kubric mmap shard {shard} has an invalid leading dimension")
            self._shards[shard] = value
            while len(self._shards) > self.max_open_shards:
                self._shards.popitem(last=False)
            return value

    def read(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        index = int(index)
        if not 0 <= index < self.count:
            raise IndexError(f"Kubric mmap index {index} outside [0,{self.count})")
        shard, local = divmod(index, self.shard_size)
        arrays = self._open_shard(shard)
        return arrays[0][local], arrays[1][local], arrays[2][local]
