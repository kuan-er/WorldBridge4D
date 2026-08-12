"""Deterministic three-dataset utilities for the 256px distributed route."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Protocol

import numpy as np
import torch

from .data import MOViFDataset, MOViSample
from .dynamic_replica import DynamicReplicaDataset
from .geometry import GeometryBuilder
from .pointodyssey import PointOdysseyDataset

DATASET_NAMES = ("kubric", "pointodyssey", "dynamic_replica")
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
    def clean_latent(self, index: int) -> np.ndarray: ...


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
                                 sources: np.ndarray
                                 ) -> tuple[int, np.ndarray, np.ndarray]:
    """Try a deterministic source order and return the first supervised one."""
    for source in np.asarray(sources, dtype=np.int64).reshape(-1):
        xyz, valid = dataset.source_all_targets(int(index), int(source))
        if np.asarray(valid, dtype=bool).reshape(21, -1).any():
            return int(source), xyz, valid
    raise ValueError(f"clip index {index} has no eligible source/target pair")


def sample_eligible_targets(valid: np.ndarray, k: int,
                            rng: np.random.Generator) -> np.ndarray:
    """Uniform without-replacement targets among pairs with >=1 valid point."""
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 3:
        raise ValueError("valid must be [T,H,W]")
    eligible = np.flatnonzero(valid.reshape(valid.shape[0], -1).any(axis=1))
    if not len(eligible):
        raise ValueError("selected source has no eligible target")
    count = min(int(k), len(eligible))
    if count < 1:
        raise ValueError("K must be positive")
    return np.asarray(rng.choice(eligible, size=count, replace=False), dtype=np.int64)


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
        parents: dict[str, list[int]] = {}
        for index, row in enumerate(rows):
            parents.setdefault(str(row["parent_id"]), []).append(index)
        names = sorted(parents)
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


class CachedExternalDataset:
    """Attach one canonical 256 latent store to PO/DR geometry adapters."""
    def __init__(self, geometry: PointOdysseyDataset | DynamicReplicaDataset,
                 cache_root: str | Path) -> None:
        self.geometry = geometry
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
        self.latents = LatentShardStore(
            cache_root / "latents" / "wan2.1_1.3b_fp32_256"
        )

    def __len__(self) -> int:
        return len(self.rows)

    def source_all_targets(self, index: int, source: int) -> tuple[np.ndarray, np.ndarray]:
        return self.geometry.source_all_targets(self.geometry_indices[index], source)

    def source_all_targets_with_visibility(
        self, index: int, source: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.geometry.source_all_targets_with_visibility(self.geometry_indices[index], source)

    def clean_latent(self, index: int) -> np.ndarray:
        # The 256 producer writes each split in split-local row order; training
        # consumes the train split beginning at latent index zero.
        return self.latents[int(index)]


class MOViF256Dataset:
    """Read-only MOVi-F 512 source -> audited 256 geometry and latent cache."""
    def __init__(self, raw_root: str | Path, cache_root: str | Path,
                 split: str = "train") -> None:
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
        self._sample_cache: OrderedDict[int, MOViSample] = OrderedDict()

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _resize_sample(sample: MOViSample) -> MOViSample:
        from PIL import Image
        size = 256
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
        value = self._sample_cache.pop(raw_index, None)
        if value is None:
            value = self._resize_sample(self.native[raw_index])
        self._sample_cache[raw_index] = value
        while len(self._sample_cache) > 2:
            self._sample_cache.popitem(last=False)
        return value

    def _geometry(self, index: int, source: int, compute_visibility: bool
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        xyz, visible, valid, _ = GeometryBuilder(self.sample(index)).trajectory_block(
            int(source), coordinate_frame="source", compute_visibility=compute_visibility
        )
        xyz = xyz.reshape(256, 256, 21, 3).transpose(2, 3, 0, 1).astype(np.float32)
        valid = valid.reshape(256, 256, 21).transpose(2, 0, 1).astype(bool)
        visible_out = (
            visible.reshape(256, 256, 21).transpose(2, 0, 1).astype(bool)
            if visible is not None else None
        )
        return xyz, valid, visible_out

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

    def clean_latent(self, index: int) -> np.ndarray:
        if self.latents is None:
            raise FileNotFoundError("MOVi-F 256 latent cache has not been generated")
        return self.latents[int(index)]


def load_training_datasets(config: dict[str, Any]) -> dict[str, TrainingDataset]:
    roots = config["datasets"]
    image_size = int(config["image_size"])
    if image_size != 256:
        raise ValueError("three-dataset route requires image_size=256")
    kubric = MOViF256Dataset(roots["kubric"]["raw_root"], roots["kubric"]["cache_root"])
    po_geometry = PointOdysseyDataset(
        roots["pointodyssey"].get("geometry_cache_root", roots["pointodyssey"]["cache_root"]),
        image_size=image_size, raw_root=roots["pointodyssey"]["raw_root"],
    )
    dr_geometry = DynamicReplicaDataset(
        roots["dynamic_replica"].get("geometry_cache_root", roots["dynamic_replica"]["cache_root"]),
        image_size=image_size, raw_root=roots["dynamic_replica"]["raw_root"],
    )
    return {
        "kubric": kubric,
        "pointodyssey": CachedExternalDataset(po_geometry, roots["pointodyssey"]["cache_root"]),
        "dynamic_replica": CachedExternalDataset(dr_geometry, roots["dynamic_replica"]["cache_root"]),
    }
