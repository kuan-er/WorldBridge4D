"""Cached adapter shared by PointOdyssey and Dynamic Replica."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..cache.latent import LazyLatentCache, LatentShardStore
from ..cache.rgb import RGBUInt8ShardStore
from .dynamic_replica import DynamicReplicaDataset
from .pointodyssey import PointOdysseyDataset

class CachedExternalDataset:
    """Attach canonical sharded and optional lazy 256 latents to PO/DR geometry."""
    def __init__(self, geometry: PointOdysseyDataset | DynamicReplicaDataset,
                 cache_root: str | Path, dataset_name: str,
                 split: str = "train", allow_missing_latents: bool = False,
                 rgb_cache_root: str | Path | None = None,
                 rgb_cache_max_open_shards: int = 16) -> None:
        self.geometry = geometry
        self.dataset_name = str(dataset_name)
        cache_root = Path(cache_root)
        index_path = cache_root / "splits" / f"{split}.jsonl"
        if not index_path.is_file():
            raise FileNotFoundError(index_path)
        self.rows = [json.loads(line) for line in index_path.read_text().splitlines() if line]
        self.rgb_shards = (
            RGBUInt8ShardStore(
                rgb_cache_root, self.dataset_name, len(self.rows),
                max_open_shards=rgb_cache_max_open_shards,
            )
            if rgb_cache_root is not None else None
        )
        by_clip = {str(row["clip_id"]): index for index, row in enumerate(geometry.rows)}
        try:
            self.geometry_indices = [by_clip[str(row["clip_id"])] for row in self.rows]
        except KeyError as exc:
            raise ValueError(f"256 index is not a subset of its geometry cache: {exc}") from exc
        latent_name = (
            "wan2.1_1.3b_fp32_256"
            if split == "train" else f"wan2.1_1.3b_fp32_256_{split}"
        )
        try:
            self.latents: LatentShardStore | None = LatentShardStore(
                cache_root / "latents" / latent_name
            )
        except FileNotFoundError:
            if not allow_missing_latents:
                raise
            self.latents = None
        lazy_name = (
            "wan2.1_1.3b_fp32_256_lazy"
            if split == "train" else f"wan2.1_1.3b_fp32_256_{split}_lazy"
        )
        self.lazy_latents = LazyLatentCache(
            cache_root / "latents" / lazy_name, self.dataset_name
        )

    def __len__(self) -> int:
        return len(self.rows)

    def source_all_targets(self, index: int, source: int) -> tuple[np.ndarray, np.ndarray]:
        return self.geometry.source_all_targets(self.geometry_indices[index], source)

    def source_all_targets_with_visibility(
        self, index: int, source: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.geometry.source_all_targets_with_visibility(self.geometry_indices[index], source)

    def cycle_camera(self, index: int) -> dict[str, np.ndarray | float]:
        return self.geometry.cycle_camera(self.geometry_indices[int(index)])

    def rgb(self, index: int) -> np.ndarray:
        if self.rgb_shards is not None:
            return self.rgb_shards.clip(int(index))
        return self.geometry.rgb(self.geometry_indices[int(index)])

    def source_rgb(self, index: int, source: int) -> np.ndarray:
        source = int(source)
        if not 0 <= source < 21:
            raise ValueError(f"source must be in [0,20], got {source}")
        if self.rgb_shards is not None:
            return self.rgb_shards.source_rgb(int(index), source)
        geometry_index = self.geometry_indices[int(index)]
        if hasattr(self.geometry, "source_rgb"):
            return self.geometry.source_rgb(geometry_index, source)
        return self.geometry.rgb(geometry_index)[source]

    def set_lazy_vae_sha256(self, value: str) -> None:
        self.lazy_latents.set_vae_sha256(value)

    def clean_latent(self, index: int) -> np.ndarray:
        # Prefer immutable compact shards, then the atomic per-clip lazy tier.
        if self.latents is not None:
            try:
                return self.latents[int(index)]
            except IndexError:
                pass
        return self.lazy_latents.read(int(index), str(self.rows[int(index)]["clip_id"]))

    def latent_cached(self, index: int) -> bool:
        try:
            self.clean_latent(index)
            return True
        except (FileNotFoundError, IndexError):
            return False

    def cache_latent(self, index: int, value: np.ndarray, vae_sha256: str) -> bool:
        return self.lazy_latents.write(
            int(index), str(self.rows[int(index)]["clip_id"]), value, vae_sha256
        )
