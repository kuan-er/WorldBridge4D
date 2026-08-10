"""Source-anchored dynamic pointmap construction and metric XYZ codec.

For a source frame ``s`` and source pixel ``p=(u,v)``, every output location
``Y[s][t,v,u]`` is the *same source pixel's* physical 3D point at target time
``t``.  The output grid is never re-anchored to the target image grid.  XYZ is
normally in the clip-frame-0 camera anchor, or in the fixed camera coordinate
system of ``s`` when ``coordinate_frame='source'``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

from .data import MOViSample
from .geometry import GeometryBuilder


@dataclass
class DynamicPointmap:
    xyz: np.ndarray       # [T,H,W,3], source-grid anchored
    visible: np.ndarray | None  # [T,H,W], M; omitted on XYZ-only training paths
    valid: np.ndarray      # [T,H,W], A
    source: int
    coordinate_frame: str = "anchor"


def build_dynamic_pointmap(sample: MOViSample, source: int, *,
                           depth_tolerance: float = 0.05,
                           depth_relative_tolerance: float = 0.01,
                           coordinate_frame: str = "anchor",
                           compute_visibility: bool = True) -> DynamicPointmap:
    """Build one source-grid trajectory map with optional visibility masks."""
    source = int(source)
    if not 0 <= source < sample.num_frames:
        raise ValueError(f"source={source} outside [0,{sample.num_frames})")
    coordinate_frame = GeometryBuilder._check_coordinate_frame(coordinate_frame)
    geom = GeometryBuilder(sample, depth_tolerance, depth_relative_tolerance)
    xyz, visible, valid, _ = geom.trajectory_block(
        source, coordinate_frame=coordinate_frame, compute_visibility=compute_visibility,
    )
    # GeometryBuilder returns [source_pixel,target_time,3]. Reorder only the
    # tensor axes; the HxW locations remain the source-frame pixel grid.
    xyz = xyz.reshape(sample.height, sample.width, sample.num_frames, 3).transpose(2, 0, 1, 3)
    if visible is not None:
        visible = visible.reshape(sample.height, sample.width, sample.num_frames).transpose(2, 0, 1)
    valid = valid.reshape(sample.height, sample.width, sample.num_frames).transpose(2, 0, 1)
    # A=0 has no XYZ semantics. Zeroing only invalid values prevents accidental
    # NaNs entering the VAE, while M=0,A=1 remains untouched.
    xyz = np.where(valid[..., None], xyz, 0.0).astype(np.float32)
    return DynamicPointmap(
        xyz=xyz, visible=visible.astype(bool) if visible is not None else None,
        valid=valid.astype(bool), source=source,
        coordinate_frame=coordinate_frame,
    )


@dataclass
class WorldCoordCodec:
    """Fixed global per-axis affine map between metric XYZ and [0,1].

    ``lo`` and ``hi`` are computed once from the training split. They are not
    recomputed per clip/video. Encoding clips values for a reported saturation
    rate; decoding is the inverse affine map (without a hidden normalization).
    """
    lo: np.ndarray
    hi: np.ndarray
    margin: float = 0.02

    def __post_init__(self) -> None:
        self.lo = np.asarray(self.lo, dtype=np.float32).reshape(3)
        self.hi = np.asarray(self.hi, dtype=np.float32).reshape(3)
        if not np.isfinite(self.lo).all() or not np.isfinite(self.hi).all() or np.any(self.hi <= self.lo):
            raise ValueError("codec bounds must be finite and hi>lo")
        self.margin = float(self.margin)
        if self.margin < 0:
            raise ValueError("margin must be non-negative")
        span = self.hi - self.lo
        self.lo = self.lo - self.margin * span
        self.hi = self.hi + self.margin * span

    @classmethod
    def from_training_values(cls, values: np.ndarray, margin: float = 0.02) -> "WorldCoordCodec":
        x = np.asarray(values, dtype=np.float32).reshape(-1, 3)
        finite = np.isfinite(x).all(axis=1)
        if not finite.any():
            raise ValueError("no finite training XYZ values")
        return cls(x[finite].min(0), x[finite].max(0), margin)

    @classmethod
    def from_json(cls, path: str | Path) -> "WorldCoordCodec":
        p = json.loads(Path(path).read_text())
        return cls(np.asarray(p["lo"], np.float32), np.asarray(p["hi"], np.float32), margin=0.0)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"lo": self.lo.tolist(), "hi": self.hi.tolist(), "margin": self.margin}, indent=2))

    def encode_xyz_to_pseudorgb(self, xyz: np.ndarray, return_stats: bool = False):
        x = np.asarray(xyz, dtype=np.float32)
        if x.shape[-1] != 3:
            raise ValueError(f"XYZ last dimension must be 3, got {x.shape}")
        value = (x - self.lo) / (self.hi - self.lo)
        clipped = np.clip(value, 0.0, 1.0)
        saturation = float(np.mean((value < 0.0) | (value > 1.0)))
        if return_stats:
            return clipped.astype(np.float32), {"saturation_rate": saturation}
        return clipped.astype(np.float32)

    def decode_pseudorgb_to_xyz(self, value: np.ndarray) -> np.ndarray:
        y = np.asarray(value, dtype=np.float32)
        if y.shape[-1] != 3:
            raise ValueError(f"pseudo RGB last dimension must be 3, got {y.shape}")
        return (y * (self.hi - self.lo) + self.lo).astype(np.float32)

    def round_trip(self, xyz: np.ndarray) -> tuple[np.ndarray, float]:
        encoded, stats = self.encode_xyz_to_pseudorgb(xyz, return_stats=True)
        return self.decode_pseudorgb_to_xyz(encoded), stats["saturation_rate"]


def training_codec(samples, *, margin: float = 0.02,
                   depth_tolerance: float = 0.05,
                   depth_relative_tolerance: float = 0.01) -> WorldCoordCodec:
    chunks = []
    for sample in samples:
        p, valid = GeometryBuilder(sample, depth_tolerance, depth_relative_tolerance).pointmaps()
        chunks.append(p[valid])
    if not chunks:
        raise ValueError("no samples for codec statistics")
    return WorldCoordCodec.from_training_values(np.concatenate(chunks, axis=0), margin=margin)
