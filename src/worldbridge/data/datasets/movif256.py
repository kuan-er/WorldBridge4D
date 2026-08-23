"""MOVi-F adapter for the 256px three-dataset route."""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
import threading
from typing import Any

import numpy as np

from ..cache.geometry import KubricGeometryMmapStore
from ..cache.latent import LazyLatentCache, LatentShardStore
from ..cache.rgb import RGBUInt8ShardStore
from ..constants import KUBRIC_GEOMETRY_CACHE
from ..geometry import GeometryBuilder
from ..movif import MOViFDataset, MOViSample

class MOViF256Dataset:
    """Read-only MOVi-F 512 source -> audited 256 geometry and latent cache."""
    def __init__(self, raw_root: str | Path, cache_root: str | Path,
                 split: str = "train", allow_missing_latents: bool = False,
                 geometry_mmap_root: str | Path | None = None,
                 geometry_compact_root: str | Path = KUBRIC_GEOMETRY_CACHE,
                 geometry_sample_cache_size: int = 16,
                 geometry_mmap_max_open_shards: int | None = None,
                 rgb_cache_root: str | Path | None = None,
                 rgb_cache_max_open_shards: int = 16) -> None:
        cache_root = Path(cache_root)
        self.cache_root = cache_root
        index_path = cache_root / "splits" / f"{split}.jsonl"
        if not index_path.is_file():
            raise FileNotFoundError(index_path)
        self.rows = [json.loads(line) for line in index_path.read_text().splitlines() if line]
        if not self.rows:
            raise RuntimeError(f"empty MOVi-F 256 index: {index_path}")
        self.rgb_shards = (
            RGBUInt8ShardStore(
                rgb_cache_root, "kubric", len(self.rows),
                max_open_shards=rgb_cache_max_open_shards,
            )
            if rgb_cache_root is not None else None
        )
        max_index = max(int(row["raw_index"]) for row in self.rows)
        self.native = MOViFDataset(raw_root, split=split, clip_length=21, clip_start=0,
                                   max_examples=max_index + 1)
        latent_root = cache_root / "latents" / "wan2.1_1.3b_fp32_256"
        self.latents = LatentShardStore(latent_root) if latent_root.is_dir() and any(latent_root.glob("*.safetensors")) else None
        if self.latents is None and not allow_missing_latents:
            raise FileNotFoundError("MOVi-F 256 latent cache has not been generated")
        self.lazy_latents = LazyLatentCache(
            cache_root / "latents" / "wan2.1_1.3b_fp32_256_lazy", "kubric"
        )
        self.geometry_compact_root = Path(geometry_compact_root)
        self.geometry_mmap = (
            KubricGeometryMmapStore(
                geometry_mmap_root,
                max_open_shards=geometry_mmap_max_open_shards,
            )
            if geometry_mmap_root is not None else None
        )
        self._sample_cache_size = max(1, int(geometry_sample_cache_size))
        self._sample_cache: OrderedDict[int, MOViSample] = OrderedDict()
        self._sample_cache_lock = threading.RLock()
        self._sample_load_locks: dict[int, threading.Lock] = {}
        # Geometry-only compact samples intentionally omit RGB.  Keep a small,
        # separate resized RGB-clip LRU for source-pyramid training without
        # inflating the much larger geometry sample cache.
        self._rgb_cache_size = max(1, min(4, self._sample_cache_size))
        self._rgb_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._rgb_cache_lock = threading.RLock()
        self._rgb_load_locks: dict[int, threading.Lock] = {}

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _resize_sample(sample: MOViSample, skip_rgb: bool = False) -> MOViSample:
        from PIL import Image
        size = 256
        if skip_rgb:
            rgb = np.zeros((len(sample.depth), 0, 0, 3), np.uint8)
        else:
            rgb = np.stack([np.asarray(Image.fromarray(frame).resize(
                (size, size), Image.Resampling.BICUBIC), np.uint8) for frame in sample.rgb])
        depth = np.stack([np.asarray(Image.fromarray(frame).resize(
            (size, size), Image.Resampling.NEAREST), np.float32) for frame in sample.depth])
        segmentation = np.stack([np.asarray(Image.fromarray(frame.astype(np.int32)).resize(
            (size, size), Image.Resampling.NEAREST), np.int64) for frame in sample.segmentation])
        depth_valid = np.isfinite(depth) & (depth > 0)
        return MOViSample(
            sample.video_name, rgb, depth, depth_valid, segmentation,
            sample.camera_positions, sample.camera_quaternions, sample.focal_length,
            sample.sensor_width, sample.field_of_view, sample.instance_positions,
            sample.instance_quaternions, sample.instance_dynamic,
            sample.instance_visibility, sample.depth_range, sample.clip_start,
        )

    def sample(self, index: int) -> MOViSample:
        raw_index = int(self.rows[index]["raw_index"])
        with self._sample_cache_lock:
            value = self._sample_cache.get(raw_index)
            if value is not None:
                self._sample_cache.move_to_end(raw_index)
                return value
            load_lock = self._sample_load_locks.setdefault(raw_index, threading.Lock())
        # Coalesce duplicate fallback/source requests without serializing loads
        # for unrelated clips.  Recheck after acquiring the per-clip lock.
        with load_lock:
            with self._sample_cache_lock:
                value = self._sample_cache.get(raw_index)
                if value is not None:
                    self._sample_cache.move_to_end(raw_index)
                    return value
            compact = self.geometry_compact_root / f"geom_{raw_index:06d}.npz"
            if self.geometry_mmap is not None:
                if not compact.is_file():
                    raise FileNotFoundError(compact)
                value = self._load_compact_sample(compact, self.geometry_mmap.read(raw_index))
            elif compact.is_file():
                value = self._load_compact_sample(compact)
            else:
                value = self._resize_sample(self.native[raw_index])
            with self._sample_cache_lock:
                self._sample_cache[raw_index] = value
                self._sample_cache.move_to_end(raw_index)
                while len(self._sample_cache) > self._sample_cache_size:
                    self._sample_cache.popitem(last=False)
            return value

    @staticmethod
    def _load_compact_sample(
        path: Path,
        heavy: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> MOViSample:
        """Rebuild a MOViSample from compact metadata and optional mmap arrays."""
        with np.load(path) as z:
            depth, depth_valid, segmentation = (
                heavy if heavy is not None
                else (z["depth"], z["depth_valid"], z["segmentation"])
            )
            return MOViSample(
                video_name="", rgb=np.zeros((0, 0, 0, 3), np.uint8),
                depth=depth.astype(np.float32),
                depth_valid=depth_valid.astype(bool),
                segmentation=segmentation.astype(np.int64),
                camera_positions=z["camera_positions"].astype(np.float32),
                camera_quaternions=z["camera_quaternions"].astype(np.float32),
                focal_length=float(z["focal_length"]),
                sensor_width=float(z["sensor_width"]),
                field_of_view=float(z["field_of_view"]),
                instance_positions=z["instance_positions"].astype(np.float32),
                instance_quaternions=z["instance_quaternions"].astype(np.float32),
                instance_dynamic=z["instance_dynamic"].astype(bool),
                instance_visibility=z["instance_visibility"].astype(np.uint16),
                depth_range=z["depth_range"].astype(np.float32),
                clip_start=int(z["clip_start"]),
            )

    @staticmethod
    def _geometry_from_sample(
        sample: MOViSample, source: int, compute_visibility: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        xyz, visible, valid, _ = GeometryBuilder(sample).trajectory_block(
            int(source), coordinate_frame="source", compute_visibility=compute_visibility
        )
        height, width, frames = sample.height, sample.width, sample.num_frames
        xyz = xyz.reshape(height, width, frames, 3).transpose(2, 3, 0, 1).astype(np.float32)
        valid = valid.reshape(height, width, frames).transpose(2, 0, 1).astype(bool)
        visible_out = (
            visible.reshape(height, width, frames).transpose(2, 0, 1).astype(bool)
            if visible is not None else None
        )
        return xyz, valid, visible_out

    def _geometry(self, index: int, source: int, compute_visibility: bool
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        return self._geometry_from_sample(
            self.sample(index), int(source), compute_visibility,
        )

    def source_all_targets(self, index: int, source: int) -> tuple[np.ndarray, np.ndarray]:
        xyz, valid, _ = self._geometry(index, source, False)
        return xyz, valid

    def source_all_targets_with_visibility(
        self, index: int, source: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xyz, valid, visible = self._geometry(index, source, True)
        assert visible is not None
        return xyz, valid, visible

    def rgb(self, index: int) -> np.ndarray:
        if self.rgb_shards is not None:
            return self.rgb_shards.clip(int(index))
        sample = self.sample(index)
        if sample.rgb.size:
            return sample.rgb
        return self._rgb_clip(index)

    def _rgb_clip(self, index: int) -> np.ndarray:
        from PIL import Image
        raw_index = int(self.rows[int(index)]["raw_index"])
        with self._rgb_cache_lock:
            cached = self._rgb_cache.get(raw_index)
            if cached is not None:
                self._rgb_cache.move_to_end(raw_index)
                return cached
            load_lock = self._rgb_load_locks.setdefault(raw_index, threading.Lock())
        with load_lock:
            with self._rgb_cache_lock:
                cached = self._rgb_cache.get(raw_index)
                if cached is not None:
                    self._rgb_cache.move_to_end(raw_index)
                    return cached
            native_rgb = self.native[raw_index].rgb
            resized = np.stack([
                np.asarray(Image.fromarray(frame).resize(
                    (256, 256), Image.Resampling.BICUBIC,
                ), dtype=np.uint8)
                for frame in native_rgb
            ])
            if resized.shape != (21, 256, 256, 3):
                raise RuntimeError(f"unexpected MOVi-F RGB clip shape: {resized.shape}")
            with self._rgb_cache_lock:
                self._rgb_cache[raw_index] = resized
                self._rgb_cache.move_to_end(raw_index)
                while len(self._rgb_cache) > self._rgb_cache_size:
                    self._rgb_cache.popitem(last=False)
                return resized

    def source_rgb(self, index: int, source: int) -> np.ndarray:
        source = int(source)
        if not 0 <= source < 21:
            raise ValueError(f"source must be in [0,20], got {source}")
        if self.rgb_shards is not None:
            return self.rgb_shards.source_rgb(int(index), source)
        sample = self.sample(int(index))
        if sample.rgb.size:
            return sample.rgb[source]
        return self._rgb_clip(int(index))[source]

    def set_lazy_vae_sha256(self, value: str) -> None:
        self.lazy_latents.set_vae_sha256(value)

    def clean_latent(self, index: int) -> np.ndarray:
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
