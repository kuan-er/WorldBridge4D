"""Direct, read-only adapter for native MOVi-F 128x128 TFRecords.

The adapter intentionally parses the on-disk tf.train.Example schema instead of
assuming TFDS has registered MOVi-F. It never writes below ``data_root``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json
import os

import numpy as np


@dataclass
class MOViSample:
    video_name: str
    rgb: np.ndarray                 # [T,H,W,3], uint8
    depth: np.ndarray               # [T,H,W], float32 radial camera distance
    depth_valid: np.ndarray         # [T,H,W], bool
    segmentation: np.ndarray        # [T,H,W], int64, 0=background, 1..N=instance
    camera_positions: np.ndarray    # [T,3], float32, Kubric world
    camera_quaternions: np.ndarray  # [T,4], float32, wxyz camera-to-world
    focal_length: float
    sensor_width: float
    field_of_view: float
    instance_positions: np.ndarray  # [N,T,3], float32, Kubric world
    instance_quaternions: np.ndarray# [N,T,4], float32, wxyz local-to-world
    instance_dynamic: np.ndarray    # [N], bool
    instance_visibility: np.ndarray # [N,T], uint16 visible-pixel count
    depth_range: np.ndarray         # [2], float32
    clip_start: int

    @property
    def num_frames(self) -> int:
        return int(self.depth.shape[0])

    @property
    def height(self) -> int:
        return int(self.depth.shape[1])

    @property
    def width(self) -> int:
        return int(self.depth.shape[2])

    @property
    def num_instances(self) -> int:
        return int(self.instance_positions.shape[0])


class MOViFDataset:
    """Small random-access view over native TFRecord shards.

    Index construction scans record boundaries only until ``max_examples``.
    This is suitable for smoke/tiny/bounded research runs. A production-scale
    input pipeline can replace it without changing ``MOViSample`` or geometry.
    """

    def __init__(
        self,
        data_root: str | os.PathLike,
        split: str = "train",
        clip_length: int = 21,
        clip_start: int | None = 0,
        max_examples: int | None = None,
        seed: int = 0,
    ) -> None:
        self.root = Path(data_root)
        self.version_dir = self._find_version_dir(self.root)
        self.split = split
        self.clip_length = int(clip_length)
        self.clip_start = clip_start
        self.seed = int(seed)
        self.files = self._split_files(split)
        if not self.files:
            available = ", ".join(self.available_splits()) or "none"
            raise FileNotFoundError(f"No MOVi-F TFRecords for split={split!r}; available: {available}")
        self.records = self._index_records(max_examples)
        if not self.records:
            raise RuntimeError(f"No examples found in {self.files[0]}")

    @staticmethod
    def _find_version_dir(root: Path) -> Path:
        base = root / "128x128" if (root / "128x128").is_dir() else root
        versions = sorted(p for p in base.iterdir() if p.is_dir() and (p / "features.json").exists())
        if not versions:
            raise FileNotFoundError(f"Could not find 128x128/*/features.json below {root}")
        return versions[-1]

    def available_splits(self) -> list[str]:
        result = set()
        for p in self.version_dir.glob("movi_f-*.tfrecord-*"):
            name = p.name
            result.add(name.split("movi_f-", 1)[1].split(".tfrecord-", 1)[0])
        return sorted(result)

    def _split_files(self, split: str) -> list[Path]:
        aliases = {"val": "validation", "valid": "validation"}
        canonical = aliases.get(split, split)
        return sorted(self.version_dir.glob(f"movi_f-{canonical}.tfrecord-*"))

    @staticmethod
    def _tf():
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf
        return tf

    def _metadata_shard_lengths(self) -> list[int] | None:
        """Read TFDS shard counts without scanning/decode-opening every shard."""
        info_path = self.version_dir / "dataset_info.json"
        try:
            info = json.loads(info_path.read_text())
            canonical = {"val": "validation", "valid": "validation"}.get(self.split, self.split)
            split_info = next(x for x in info.get("splits", []) if x.get("name") == canonical)
            lengths = [int(x) for x in split_info["shardLengths"]]
            return lengths if len(lengths) == len(self.files) else None
        except (OSError, KeyError, StopIteration, TypeError, ValueError):
            return None

    def _index_records(self, max_examples: int | None) -> list[tuple[Path, int]]:
        limit = int(max_examples) if max_examples is not None else None
        records: list[tuple[Path, int]] = []
        lengths = self._metadata_shard_lengths()
        if lengths is not None:
            for path, length in zip(self.files, lengths):
                remaining = length if limit is None else max(0, limit - len(records))
                records.extend((path, local_index) for local_index in range(min(length, remaining)))
                if limit is not None and len(records) >= limit:
                    break
            return records
        tf = self._tf()
        for path in self.files:
            for local_index, _ in enumerate(tf.data.TFRecordDataset([str(path)])):
                records.append((path, local_index))
                if limit is not None and len(records) >= limit:
                    return records
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _raw_record(self, index: int) -> bytes:
        tf = self._tf()
        path, local_index = self.records[index]
        record = next(iter(tf.data.TFRecordDataset([str(path)]).skip(local_index).take(1)), None)
        if record is None:
            raise IndexError(index)
        return bytes(record.numpy())

    @staticmethod
    def _feature_array(example, key: str, kind: str, dtype) -> np.ndarray:
        feature = example.features.feature[key]
        return np.asarray(getattr(feature, kind).value, dtype=dtype)

    def _decode(self, raw: bytes, index: int) -> MOViSample:
        tf = self._tf()
        ex = tf.train.Example.FromString(raw)
        f32 = lambda key: self._feature_array(ex, key, "float_list", np.float32)
        i64 = lambda key: self._feature_array(ex, key, "int64_list", np.int64)
        bytes_values = lambda key: list(ex.features.feature[key].bytes_list.value)

        total_t = int(i64("metadata/num_frames")[0])
        height = int(i64("metadata/height")[0])
        width = int(i64("metadata/width")[0])
        n_instances = int(i64("metadata/num_instances")[0])
        if self.clip_length > total_t:
            raise ValueError(f"clip_length={self.clip_length} exceeds native {total_t}; padding is intentionally unsupported")
        max_start = total_t - self.clip_length
        if self.clip_start is None:
            # Stable per-index start; no global RNG state and no hidden temporal padding.
            start = int(np.random.default_rng(self.seed + index).integers(max_start + 1))
        else:
            start = int(self.clip_start)
        if not 0 <= start <= max_start:
            raise ValueError(f"clip_start={start} outside [0,{max_start}]")
        sl = slice(start, start + self.clip_length)

        def png_sequence(key: str, channels: int, dtype):
            values = bytes_values(key)
            if len(values) != total_t:
                raise ValueError(f"{key}: expected {total_t} PNGs, got {len(values)}")
            return np.stack([
                tf.io.decode_png(x, channels=channels, dtype=dtype).numpy() for x in values[sl]
            ], axis=0)

        depth_range = f32("metadata/depth_range")
        depth_raw = png_sequence("depth", 1, tf.uint16)[..., 0]
        depth = depth_range[0] + depth_raw.astype(np.float32) / np.float32(65535.0) * (depth_range[1] - depth_range[0])
        depth_valid = np.isfinite(depth) & (depth > 0.0) & (depth_raw > 0)
        segmentation = png_sequence("segmentations", 1, tf.uint8)[..., 0].astype(np.int64)
        rgb = png_sequence("video", 3, tf.uint8).astype(np.uint8)

        camera_positions = f32("camera/positions").reshape(total_t, 3)[sl]
        camera_quaternions = f32("camera/quaternions").reshape(total_t, 4)[sl]
        instance_positions = f32("instances/positions").reshape(n_instances, total_t, 3)[:, sl]
        instance_quaternions = f32("instances/quaternions").reshape(n_instances, total_t, 4)[:, sl]
        instance_dynamic = i64("instances/is_dynamic").astype(bool)
        instance_visibility = i64("instances/visibility").reshape(n_instances, total_t)[:, sl].astype(np.uint16)
        video_name = bytes_values("metadata/video_name")[0].decode("utf8")
        return MOViSample(
            video_name=video_name, rgb=rgb, depth=depth.astype(np.float32), depth_valid=depth_valid,
            segmentation=segmentation, camera_positions=camera_positions.astype(np.float32),
            camera_quaternions=camera_quaternions.astype(np.float32),
            focal_length=float(f32("camera/focal_length")[0]), sensor_width=float(f32("camera/sensor_width")[0]),
            field_of_view=float(f32("camera/field_of_view")[0]),
            instance_positions=instance_positions.astype(np.float32),
            instance_quaternions=instance_quaternions.astype(np.float32),
            instance_dynamic=instance_dynamic, instance_visibility=instance_visibility,
            depth_range=depth_range.astype(np.float32), clip_start=start,
        )

    def __getitem__(self, index: int) -> MOViSample:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self._decode(self._raw_record(index), index)

    def __iter__(self) -> Iterable[MOViSample]:
        for index in range(len(self)):
            yield self[index]
