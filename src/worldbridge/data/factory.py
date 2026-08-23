"""Construction and index validation for training datasets."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .constants import (
    DATASET_NAMES,
    DYNAMIC_REPLICA_DEPTH_CACHE,
    DYNAMIC_REPLICA_TRAJECTORY_CACHE,
    KUBRIC_GEOMETRY_CACHE,
    POINTODYSSEY_ANNO_CACHE,
    POINTODYSSEY_ANNO_NPY_CACHE,
    POINTODYSSEY_DEPTH_CACHE,
)
from .datasets.cached_external import CachedExternalDataset
from .datasets.dynamic_replica import DynamicReplicaDataset
from .datasets.movif256 import MOViF256Dataset
from .datasets.pointodyssey import PointOdysseyDataset
from .movif import MOViFDataset

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
    rgb_cache_root = config.get("source_rgb_cache_root")
    rgb_cache_max_open_shards = int(config.get("source_rgb_cache_max_open_shards", 16))
    if name == "kubric":
        max_open_shards = values.get("geometry_mmap_max_open_shards")
        return MOViF256Dataset(
            values["raw_root"], values["cache_root"],
            allow_missing_latents=allow_missing_latents,
            geometry_mmap_root=values.get("geometry_mmap_root"),
            geometry_compact_root=values.get(
                "geometry_compact_root", KUBRIC_GEOMETRY_CACHE
            ),
            geometry_sample_cache_size=int(
                values.get("geometry_sample_cache_size", 16)
            ),
            geometry_mmap_max_open_shards=(
                None if max_open_shards is None else int(max_open_shards)
            ),
            rgb_cache_root=rgb_cache_root,
            rgb_cache_max_open_shards=rgb_cache_max_open_shards,
        )
    if name == "pointodyssey":
        geometry = PointOdysseyDataset(
            values.get("geometry_cache_root", values["cache_root"]),
            image_size=image_size, raw_root=values["raw_root"],
            annotation_cache_root=values.get(
                "annotation_cache_root", POINTODYSSEY_ANNO_CACHE
            ),
            annotation_npy_cache_root=values.get(
                "annotation_npy_cache_root", POINTODYSSEY_ANNO_NPY_CACHE
            ),
            depth_cache_root=values.get(
                "depth_cache_root", POINTODYSSEY_DEPTH_CACHE
            ),
        )
    else:
        geometry = DynamicReplicaDataset(
            values.get("geometry_cache_root", values["cache_root"]),
            image_size=image_size, raw_root=values["raw_root"],
            trajectory_cache_root=values.get(
                "trajectory_cache_root", DYNAMIC_REPLICA_TRAJECTORY_CACHE
            ),
            depth_cache_root=values.get(
                "depth_cache_root", DYNAMIC_REPLICA_DEPTH_CACHE
            ),
        )
    return CachedExternalDataset(
        geometry, values["cache_root"], name,
        allow_missing_latents=allow_missing_latents,
        rgb_cache_root=rgb_cache_root,
        rgb_cache_max_open_shards=rgb_cache_max_open_shards,
    )


def load_training_datasets(config: dict[str, Any],
                           allow_missing_latents: bool = False) -> dict[str, TrainingDataset]:
    return {
        name: load_training_dataset(config, name, allow_missing_latents=allow_missing_latents)
        for name in DATASET_NAMES
    }
