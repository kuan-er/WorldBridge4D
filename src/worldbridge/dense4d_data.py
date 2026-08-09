"""Pair sampling and dense MOVi-F XYZ targets for H004."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from .data import MOViSample
from .pointmap import DynamicPointmap, build_dynamic_pointmap


@dataclass(frozen=True)
class CoordinateStats:
    mean: np.ndarray
    scale: np.ndarray
    examples: int = 0

    def __post_init__(self):
        mean = np.asarray(self.mean, dtype=np.float32).reshape(3)
        scale = np.asarray(self.scale, dtype=np.float32).reshape(3)
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("coordinate mean/scale must be finite and scale>0")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)

    @classmethod
    def from_npz(cls, path: str | Path) -> "CoordinateStats":
        with np.load(path) as values:
            return cls(values["mean"], values["scale"], int(values.get("examples", 0)))

    def normalize_numpy(self, xyz: np.ndarray) -> np.ndarray:
        return ((np.asarray(xyz, dtype=np.float32) - self.mean) / self.scale).astype(np.float32)

    def denormalize_numpy(self, normalized_xyz: np.ndarray) -> np.ndarray:
        return (np.asarray(normalized_xyz, dtype=np.float32) * self.scale + self.mean).astype(np.float32)

    def tensors(self, device: torch.device | str, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.as_tensor(self.mean, device=device, dtype=dtype).reshape(1, 1, 3, 1, 1)
        scale = torch.as_tensor(self.scale, device=device, dtype=dtype).reshape(1, 1, 3, 1, 1)
        return mean, scale


class DynamicPointmapCache:
    """Bounded CPU cache keyed by clip, source, and coordinate convention."""

    def __init__(self, max_entries: int = 32, depth_tolerance: float = 0.05,
                 depth_relative_tolerance: float = 0.01):
        self.max_entries = int(max_entries)
        self.depth_tolerance = float(depth_tolerance)
        self.depth_relative_tolerance = float(depth_relative_tolerance)
        self._values: OrderedDict[tuple[str, int, int, str], DynamicPointmap] = OrderedDict()

    def get(self, sample: MOViSample, source: int, coordinate_frame: str = "anchor") -> DynamicPointmap:
        coordinate_frame = str(coordinate_frame).lower()
        if coordinate_frame not in {"anchor", "source"}:
            raise ValueError(f"coordinate_frame must be 'anchor' or 'source', got {coordinate_frame!r}")
        key = (sample.video_name, sample.clip_start, int(source), coordinate_frame)
        value = self._values.pop(key, None)
        if value is None:
            value = build_dynamic_pointmap(
                sample, int(source), depth_tolerance=self.depth_tolerance,
                depth_relative_tolerance=self.depth_relative_tolerance,
                coordinate_frame=coordinate_frame,
            )
        self._values[key] = value
        while len(self._values) > self.max_entries:
            self._values.popitem(last=False)
        return value


def dense_pair_targets(sample: MOViSample, source: Sequence[int], target: Sequence[int],
                       stats: CoordinateStats, cache: DynamicPointmapCache | None = None,
                       coordinate_frame: str = "anchor"
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized/metric XYZ, visibility M, and validity A for K maps."""
    coordinate_frame = str(coordinate_frame).lower()
    if coordinate_frame not in {"anchor", "source"}:
        raise ValueError(f"coordinate_frame must be 'anchor' or 'source', got {coordinate_frame!r}")
    source = np.asarray(source, dtype=np.int64).reshape(-1)
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    if source.shape != target.shape:
        raise ValueError("source and target lengths differ")
    if np.any(source < 0) or np.any(source >= sample.num_frames) or np.any(target < 0) or np.any(target >= sample.num_frames):
        raise ValueError("pair time index outside clip")
    cache = cache or DynamicPointmapCache(max_entries=max(len(np.unique(source)), 1))
    metric, visible, valid = [], [], []
    for s, t in zip(source, target):
        pointmap = cache.get(sample, int(s), coordinate_frame=coordinate_frame)
        metric.append(pointmap.xyz[int(t)].transpose(2, 0, 1))
        visible.append(pointmap.visible[int(t)])
        valid.append(pointmap.valid[int(t)])
    metric_xyz = np.stack(metric).astype(np.float32)
    # Coordinate axis is channel-first here.
    normalized = ((metric_xyz - stats.mean[None, :, None, None]) /
                  stats.scale[None, :, None, None]).astype(np.float32)
    return normalized, metric_xyz, np.stack(visible).astype(bool), np.stack(valid).astype(bool)


def _off_diagonal_pair(num_frames: int, category: int, rng: np.random.Generator) -> tuple[int, int]:
    if num_frames < 2:
        raise ValueError("off-diagonal pairs require at least two frames")
    short_max = max(1, num_frames // 4)
    long_min = max(1, num_frames // 2)
    forward = category % 2 == 0
    long_gap = category // 2 == 1
    if long_gap:
        gap = int(rng.integers(long_min, num_frames))
    else:
        gap = int(rng.integers(1, min(short_max, num_frames - 1) + 1))
    if forward:
        source = int(rng.integers(0, num_frames - gap))
        target = source + gap
    else:
        target = int(rng.integers(0, num_frames - gap))
        source = target + gap
    return source, target


def sample_dense_pairs(num_frames: int, num_pairs: int, rng: np.random.Generator,
                       diagonal_fraction: float = 1.0 / 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Balance diagonal with forward/backward and short/long off-diagonal pairs."""
    num_frames, num_pairs = int(num_frames), int(num_pairs)
    if num_pairs < 1:
        raise ValueError("num_pairs must be positive")
    diagonal = max(1, int(round(num_pairs * float(diagonal_fraction)))) if num_pairs > 1 else 1
    diagonal = min(diagonal, num_pairs)
    source = [int(rng.integers(num_frames)) for _ in range(diagonal)]
    target = list(source)
    categories = np.arange(4, dtype=np.int64)
    rng.shuffle(categories)
    for index in range(num_pairs - diagonal):
        s, t = _off_diagonal_pair(num_frames, int(categories[index % 4]), rng)
        source.append(s); target.append(t)
    order = rng.permutation(num_pairs)
    return np.asarray(source, np.int64)[order], np.asarray(target, np.int64)[order]


def sample_h001_balanced_pairs(num_frames: int, num_pairs: int,
                               rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Match H001's three training families: reconstruction, s=0, arbitrary."""
    num_frames, num_pairs = int(num_frames), int(num_pairs)
    if num_frames < 2 or num_pairs < 3:
        raise ValueError("H001-balanced sampling requires >=2 frames and >=3 pairs")
    counts = [num_pairs // 3] * 3
    for index in range(num_pairs - sum(counts)):
        counts[index] += 1
    source, target = [], []
    for _ in range(counts[0]):
        frame = int(rng.integers(num_frames))
        source.append(frame); target.append(frame)
    for _ in range(counts[1]):
        source.append(0); target.append(int(rng.integers(1, num_frames)))
    categories = np.arange(4, dtype=np.int64)
    rng.shuffle(categories)
    for index in range(counts[2]):
        s, t = _off_diagonal_pair(num_frames, int(categories[index % 4]), rng)
        source.append(s); target.append(t)
    order = rng.permutation(num_pairs)
    return np.asarray(source, np.int64)[order], np.asarray(target, np.int64)[order]


def parse_fixed_pairs(values: Iterable[Sequence[int]], num_frames: int, expected_count: int | None = None
                      ) -> tuple[np.ndarray, np.ndarray]:
    pairs = np.asarray(list(values), dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("fixed_pairs must be a list of [source,target]")
    if expected_count is not None and len(pairs) != int(expected_count):
        raise ValueError(f"fixed pair count {len(pairs)} != num_query_pairs {expected_count}")
    if np.any(pairs < 0) or np.any(pairs >= int(num_frames)):
        raise ValueError("fixed pair outside clip")
    return pairs[:, 0], pairs[:, 1]
