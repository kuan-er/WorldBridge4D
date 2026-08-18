"""Deterministic three-dataset utilities for the 256px distributed route."""
from __future__ import annotations

from collections import OrderedDict
import fcntl
import json
import math
import mmap
import os
from pathlib import Path
import re
import threading
from typing import Any, Protocol

import numpy as np
import torch

from .data import MOViFDataset, MOViSample
from .dynamic_replica import DynamicReplicaDataset
from .geometry import GeometryBuilder
from .pointodyssey import PointOdysseyDataset

DATASET_NAMES = ("kubric", "pointodyssey", "dynamic_replica")
# Per-clip compact Kubric geometry archives produced by
# scripts/compact_kubric_geometry.py; the adapter reads these instead of
# re-parsing TFRecord PNGs when present.
KUBRIC_GEOMETRY_CACHE = Path("/tmp/worldbridge4d-cache/kubric_geometry")
KUBRIC_MMAP_FIELDS = ("depth", "depth_valid", "segmentation")
MIX_CYCLE = (
    "kubric", "pointodyssey", "dynamic_replica", "kubric", "dynamic_replica",
    "pointodyssey", "kubric", "dynamic_replica", "pointodyssey", "kubric",
    "dynamic_replica", "pointodyssey", "kubric", "dynamic_replica", "pointodyssey",
    "kubric", "dynamic_replica", "pointodyssey", "kubric", "dynamic_replica",
)


class TrainingDataset(Protocol):
    rows: list[dict[str, Any]]
    def __len__(self) -> int: ...
    def source_all_targets(self, index: int, source: int) -> tuple[np.ndarray, np.ndarray]: ...
    def rgb(self, index: int) -> np.ndarray: ...
    def clean_latent(self, index: int) -> np.ndarray: ...
    def latent_cached(self, index: int) -> bool: ...
    def cache_latent(self, index: int, value: np.ndarray, vae_sha256: str) -> bool: ...


def deterministic_dataset_schedule(seed: int) -> tuple[str, ...]:
    """One seeded 20-update cycle containing exactly 7/6/7 datasets."""
    values = list(MIX_CYCLE)
    np.random.default_rng(np.random.SeedSequence([int(seed), 20])).shuffle(values)
    assert values.count("kubric") == 7 and values.count("pointodyssey") == 6
    assert values.count("dynamic_replica") == 7
    return tuple(values)


def dataset_for_step(global_step: int, seed: int) -> str:
    if int(global_step) < 0:
        raise ValueError("global_step cannot be negative")
    # Every cycle has the same shuffled composition. This makes resume a pure
    # function of global_step while retaining the exact requested ratio.
    return deterministic_dataset_schedule(seed)[int(global_step) % 20]


def cosine_learning_rate_factor(update_number: int, warmup_steps: int,
                                horizon_steps: int) -> float:
    update_number = int(update_number)
    warmup_steps = int(warmup_steps)
    horizon_steps = int(horizon_steps)
    if update_number < 1 or warmup_steps < 0 or horizon_steps <= warmup_steps:
        raise ValueError("invalid cosine schedule arguments")
    if warmup_steps and update_number <= warmup_steps:
        return update_number / warmup_steps
    progress = min(1.0, (update_number - warmup_steps) / (horizon_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def apply_cosine_schedule(optimizer: torch.optim.Optimizer, update_number: int,
                          warmup_steps: int, horizon_steps: int) -> float:
    factor = cosine_learning_rate_factor(update_number, warmup_steps, horizon_steps)
    for group in optimizer.param_groups:
        base_lr = float(group.setdefault("_base_lr", group["lr"]))
        group["lr"] = base_lr * factor
    return factor


def source_with_eligible_targets(dataset: TrainingDataset, index: int,
                                 sources: np.ndarray, min_targets: int = 1
                                 ) -> tuple[int, np.ndarray, np.ndarray]:
    """Return the first source with at least ``min_targets`` supervised pairs.

    Dataset adapters may provide ``select_source_with_eligible_targets`` to
    perform a metadata-only capacity check before materializing dense XYZ.  The
    generic fallback preserves the original protocol for external adapters.
    """
    min_targets = int(min_targets)
    if min_targets < 1:
        raise ValueError("min_targets must be positive")
    sources = np.asarray(sources, dtype=np.int64).reshape(-1)
    selector = getattr(dataset, "select_source_with_eligible_targets", None)
    if callable(selector):
        return selector(int(index), sources, min_targets)
    for source in sources:
        xyz, valid = dataset.source_all_targets(int(index), int(source))
        eligible = np.asarray(valid, dtype=bool).reshape(21, -1).any(axis=1)
        if int(eligible.sum()) >= min_targets:
            return int(source), xyz, valid
    raise ValueError(
        f"clip index {index} has no source with {min_targets} eligible targets"
    )


def sample_eligible_targets(valid: np.ndarray, k: int,
                            rng: np.random.Generator) -> np.ndarray:
    """Uniform without-replacement targets among pairs with >=1 valid point."""
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 3:
        raise ValueError("valid must be [T,H,W]")
    eligible = np.flatnonzero(valid.reshape(valid.shape[0], -1).any(axis=1))
    k = int(k)
    if k < 1:
        raise ValueError("K must be positive")
    if len(eligible) < k:
        raise ValueError(
            f"selected source has {len(eligible)} eligible targets, fewer than K={k}"
        )
    return np.asarray(rng.choice(eligible, size=k, replace=False), dtype=np.int64)


def deterministic_sample_plan(dataset: TrainingDataset, dataset_name: str,
                              seed: int, global_step: int, microstep: int,
                              rank: int, microsteps_per_rank: int = 2
                              ) -> tuple[int, int, np.random.Generator]:
    """Plan one clip/source from only checkpointed counters and rank."""
    if not len(dataset):
        raise ValueError(f"empty dataset: {dataset_name}")
    dataset_id = DATASET_NAMES.index(dataset_name)
    sequence = np.random.SeedSequence([
        int(seed), int(global_step), int(microstep), int(rank), dataset_id,
    ])
    rng = np.random.default_rng(sequence)
    rows = getattr(dataset, "rows", [])
    if dataset_name != "kubric" and rows and all("parent_id" in row for row in rows):
        # The parent->members mapping is dataset-static; build it once and
        # reuse it across the ~400k per-rank plan calls instead of rebuilding
        # it (and re-scanning every row) on each call.
        cache = getattr(dataset, "_parent_index_cache", None)
        if cache is None:
            parents: dict[str, list[int]] = {}
            for index, row in enumerate(rows):
                parents.setdefault(str(row["parent_id"]), []).append(index)
            names = sorted(parents)
            cache = (names, parents)
            dataset._parent_index_cache = cache
        names, parents = cache
        # All ranks and accumulation microsteps stay in one parent/scene for
        # this update; rank-local RNG chooses different clips inside the block.
        parent_rng = np.random.default_rng(np.random.SeedSequence([
            int(seed), int(global_step), dataset_id, 991,
        ]))
        parent = names[int(parent_rng.integers(len(names)))]
        members = parents[parent]
        base = int(parent_rng.integers(len(members)))
        index = members[(base + rank * int(microsteps_per_rank) + int(microstep)) % len(members)]
    else:
        base_rng = np.random.default_rng(np.random.SeedSequence([
            int(seed), int(global_step), dataset_id, 557,
        ]))
        base = int(base_rng.integers(len(dataset)))
        index = (base + rank * int(microsteps_per_rank) + int(microstep)) % len(dataset)
    source = int(rng.integers(21))
    return index, source, rng


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


class CachedExternalDataset:
    """Attach canonical sharded and optional lazy 256 latents to PO/DR geometry."""
    def __init__(self, geometry: PointOdysseyDataset | DynamicReplicaDataset,
                 cache_root: str | Path, dataset_name: str,
                 allow_missing_latents: bool = False) -> None:
        self.geometry = geometry
        self.dataset_name = str(dataset_name)
        cache_root = Path(cache_root)
        index_path = cache_root / "splits" / "train.jsonl"
        if not index_path.is_file():
            raise FileNotFoundError(index_path)
        self.rows = [json.loads(line) for line in index_path.read_text().splitlines() if line]
        by_clip = {str(row["clip_id"]): index for index, row in enumerate(geometry.rows)}
        try:
            self.geometry_indices = [by_clip[str(row["clip_id"])] for row in self.rows]
        except KeyError as exc:
            raise ValueError(f"256 index is not a subset of its geometry cache: {exc}") from exc
        try:
            self.latents: LatentShardStore | None = LatentShardStore(
                cache_root / "latents" / "wan2.1_1.3b_fp32_256"
            )
        except FileNotFoundError:
            if not allow_missing_latents:
                raise
            self.latents = None
        self.lazy_latents = LazyLatentCache(
            cache_root / "latents" / "wan2.1_1.3b_fp32_256_lazy", self.dataset_name
        )

    def __len__(self) -> int:
        return len(self.rows)

    def source_all_targets(self, index: int, source: int) -> tuple[np.ndarray, np.ndarray]:
        return self.geometry.source_all_targets(self.geometry_indices[index], source)

    def source_all_targets_with_visibility(
        self, index: int, source: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.geometry.source_all_targets_with_visibility(self.geometry_indices[index], source)

    def rgb(self, index: int) -> np.ndarray:
        return self.geometry.rgb(self.geometry_indices[int(index)])

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
                for array in value:
                    mapping = getattr(array, "_mmap", None)
                    try:
                        if mapping is not None and hasattr(mapping, "madvise"):
                            mapping.madvise(mmap.MADV_RANDOM)
                    except (AttributeError, OSError, ValueError):
                        # MADV_RANDOM is an optional Linux optimization; the
                        # cache contract must remain portable and read-only.
                        pass
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


class MOViF256Dataset:
    """Read-only MOVi-F 512 source -> audited 256 geometry and latent cache."""
    def __init__(self, raw_root: str | Path, cache_root: str | Path,
                 split: str = "train", allow_missing_latents: bool = False,
                 geometry_mmap_root: str | Path | None = None,
                 geometry_sample_cache_size: int = 32,
                 geometry_mmap_max_open_shards: int | None = None) -> None:
        cache_root = Path(cache_root)
        self.cache_root = cache_root
        index_path = cache_root / "splits" / f"{split}.jsonl"
        if not index_path.is_file():
            raise FileNotFoundError(index_path)
        self.rows = [json.loads(line) for line in index_path.read_text().splitlines() if line]
        if not self.rows:
            raise RuntimeError(f"empty MOVi-F 256 index: {index_path}")
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
            compact = KUBRIC_GEOMETRY_CACHE / f"geom_{raw_index:06d}.npz"
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
    def _eligible_targets_for_sample(
        sample: MOViSample, sources: np.ndarray,
    ) -> np.ndarray:
        """Return exact target-capacity masks without constructing dense XYZ.

        Kubric validity is source-depth validity intersected with each rigid
        instance's finite target state; visibility is deliberately not part of
        the training-validity contract.  Background and out-of-range segment
        ids retain source validity at every target, matching GeometryBuilder.
        """
        sources = np.asarray(sources, dtype=np.int64).reshape(-1)
        if np.any((sources < 0) | (sources >= sample.num_frames)):
            raise ValueError("source index outside sample frame range")
        builder = GeometryBuilder(sample)
        state_valid = (
            np.isfinite(sample.instance_positions).all(axis=-1)
            & np.isfinite(builder._object_rot).all(axis=(2, 3))
        )
        result = np.zeros((len(sources), sample.num_frames), dtype=bool)
        for row, source in enumerate(sources):
            source_valid = np.asarray(sample.depth_valid[source], dtype=bool)
            instance = np.asarray(sample.segmentation[source])
            static = source_valid & (
                (instance <= 0) | (instance > sample.num_instances)
            )
            if np.any(static):
                result[row] = True
                continue
            present = np.unique(instance[source_valid])
            for value in present:
                object_index = int(value) - 1
                if 0 <= object_index < sample.num_instances:
                    result[row] |= state_valid[object_index]
        return result

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

    def select_source_with_eligible_targets(
        self, index: int, sources: np.ndarray, min_targets: int,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        """Select by metadata and materialize dense geometry exactly once."""
        sources = np.asarray(sources, dtype=np.int64).reshape(-1)
        sample = self.sample(int(index))
        capacity = self._eligible_targets_for_sample(sample, sources)
        for row, source in enumerate(sources):
            if int(capacity[row].sum()) < int(min_targets):
                continue
            xyz, valid, _ = self._geometry_from_sample(sample, int(source), False)
            actual = valid.reshape(valid.shape[0], -1).any(axis=1)
            if not np.array_equal(actual, capacity[row]):
                raise RuntimeError(
                    f"Kubric eligibility audit failed for clip={index}, source={int(source)}"
                )
            return int(source), xyz, valid
        raise ValueError(
            f"clip index {index} has no source with {min_targets} eligible targets"
        )

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
        return self.sample(index).rgb

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


def prepare_training_indexes(config: dict[str, Any]) -> None:
    """Create missing train indexes outside raw mounts before lazy VAE warmup."""
    for name in DATASET_NAMES:
        values = config["datasets"][name]
        destination = Path(values["cache_root"]) / "splits" / "train.jsonl"
        if destination.is_file():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if name == "kubric":
            native = MOViFDataset(values["raw_root"], split="train", clip_length=21, clip_start=0)
            rows = [{
                "index": index, "raw_index": index,
                "clip_id": f"movi-f/train/{index:06d}",
                "parent_id": f"movi-f-train-{index:06d}",
                "start": 0, "stride": 1,
                "timestamps": [frame / 12.0 for frame in range(21)],
            } for index in range(len(native))]
            text = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
        else:
            source = Path(values.get("geometry_cache_root", values["cache_root"])) / "splits" / "train.jsonl"
            if not source.is_file():
                raise FileNotFoundError(source)
            text = source.read_text()
        temporary = destination.with_suffix(f".{os.getpid()}.tmp.jsonl")
        temporary.write_text(text)
        temporary.replace(destination)


def load_training_dataset(config: dict[str, Any], name: str,
                          allow_missing_latents: bool = False) -> TrainingDataset:
    """Load only one requested dataset (important for standalone inference)."""
    roots = config["datasets"]
    image_size = int(config["image_size"])
    name = str(name).lower()
    if image_size != 256:
        raise ValueError("three-dataset route requires image_size=256")
    if name not in DATASET_NAMES:
        raise ValueError(f"dataset must be one of {DATASET_NAMES}, got {name!r}")
    values = roots[name]
    if name == "kubric":
        max_open_shards = values.get("geometry_mmap_max_open_shards")
        return MOViF256Dataset(
            values["raw_root"], values["cache_root"],
            allow_missing_latents=allow_missing_latents,
            geometry_mmap_root=values.get("geometry_mmap_root"),
            geometry_sample_cache_size=int(
                values.get("geometry_sample_cache_size", 32)
            ),
            geometry_mmap_max_open_shards=(
                None if max_open_shards is None else int(max_open_shards)
            ),
        )
    if name == "pointodyssey":
        geometry = PointOdysseyDataset(
            values.get("geometry_cache_root", values["cache_root"]),
            image_size=image_size, raw_root=values["raw_root"],
        )
    else:
        geometry = DynamicReplicaDataset(
            values.get("geometry_cache_root", values["cache_root"]),
            image_size=image_size, raw_root=values["raw_root"],
        )
    return CachedExternalDataset(
        geometry, values["cache_root"], name,
        allow_missing_latents=allow_missing_latents,
    )


def load_training_datasets(config: dict[str, Any],
                           allow_missing_latents: bool = False) -> dict[str, TrainingDataset]:
    return {
        name: load_training_dataset(config, name, allow_missing_latents=allow_missing_latents)
        for name in DATASET_NAMES
    }
