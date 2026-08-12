"""Read-only WorldBridge4D v1 adapter for Dynamic Replica caches.

Dynamic Replica publishes persistent sparse mesh-vertex trajectories.  The
adapter rasterizes source-visible tracks onto the 128x128 source grid and uses
``traj_3d_world`` directly for every target, including targets at which the
point is occluded.  Target visibility is returned separately and never masks
canonical XYZ validity.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
import re
import threading
from typing import Any

import numpy as np
from PIL import Image
import torch

T, H, W = 21, 128, 128
RAW_W, RAW_H = 1280, 720
CROP_X, CROP_Y, CROP_SIZE = 280, 0, 720
_SCALE = W / CROP_SIZE
# PyTorch3D view coordinates are +X left, +Y up, +Z forward.  The protocol is
# +X right, +Y up, -Z forward.
_P3D_TO_PROTOCOL = np.array([-1.0, 1.0, -1.0], dtype=np.float64)


def _spatial_affine() -> np.ndarray:
    """Original pixel centres to PIL crop/resize output pixel centres."""
    return np.array([
        [_SCALE, 0.0, (0.5 - CROP_X) * _SCALE - 0.5],
        [0.0, _SCALE, (0.5 - CROP_Y) * _SCALE - 0.5],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _transform_uv(uv: np.ndarray) -> np.ndarray:
    uv = np.asarray(uv, dtype=np.float64)
    out = np.empty_like(uv[..., :2], dtype=np.float64)
    out[..., 0] = (uv[..., 0] - CROP_X + 0.5) * _SCALE - 0.5
    out[..., 1] = (uv[..., 1] - CROP_Y + 0.5) * _SCALE - 0.5
    return out


def _rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (RAW_W, RAW_H):
            raise ValueError(f"unexpected Dynamic Replica RGB size {im.size}: {path}")
        im = im.crop((CROP_X, CROP_Y, CROP_X + CROP_SIZE, CROP_Y + CROP_SIZE))
        return np.asarray(im.resize((W, H), Image.Resampling.LANCZOS), dtype=np.uint8)


def _depth(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Decode Dynamic Replica's uint16-bit-pattern float16 z-depth."""
    with Image.open(path) as im:
        raw = np.asarray(im, dtype=np.uint16)
    if raw.shape != (RAW_H, RAW_W):
        raise ValueError(f"unexpected Dynamic Replica depth size {raw.shape}: {path}")
    raw = raw[CROP_Y:CROP_Y + CROP_SIZE, CROP_X:CROP_X + CROP_SIZE]
    # Nearest-neighbour preserves the float16 bit pattern and does not blend
    # across foreground/background depth discontinuities.
    raw = np.asarray(Image.fromarray(raw).resize((W, H), Image.Resampling.NEAREST), dtype=np.uint16)
    depth = raw.view(np.float16).astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    return depth, valid


def _pixel_intrinsics(viewpoint: dict[str, Any]) -> np.ndarray:
    if str(viewpoint["intrinsics_format"]).lower() != "ndc_isotropic":
        raise ValueError(f"unsupported Dynamic Replica intrinsics: {viewpoint['intrinsics_format']}")
    focal_ndc = np.asarray(viewpoint["focal_length"], dtype=np.float64)
    principal_ndc = np.asarray(viewpoint["principal_point"], dtype=np.float64)
    rescale = min(RAW_W / 2.0, RAW_H / 2.0)
    focal = focal_ndc * rescale
    principal = np.array([RAW_W / 2.0, RAW_H / 2.0]) - principal_ndc * rescale
    K = np.array([[focal[0], 0.0, principal[0]], [0.0, focal[1], principal[1]], [0.0, 0.0, 1.0]])
    return _spatial_affine() @ K


def _camera_to_world(viewpoint: dict[str, Any]) -> np.ndarray:
    # Annotation convention: row-vector world @ R + T gives PyTorch3D view.
    R = np.asarray(viewpoint["R"], dtype=np.float64)
    translation = np.asarray(viewpoint["T"], dtype=np.float64)
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = _P3D_TO_PROTOCOL[:, None] * R.T
    world_to_camera[:3, 3] = _P3D_TO_PROTOCOL * translation
    return np.linalg.inv(world_to_camera)


def _select_source_tracks(annotation: dict[str, np.ndarray], source: int,
                          viewpoint: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select one source-visible persistent track per output raster cell."""
    uv = annotation["trajs_2d"][source].astype(np.float64)
    world = annotation["trajs_3d_world"].astype(np.float64)
    visible = annotation["visible"]
    mapped = _transform_uv(uv)
    finite_uv = np.isfinite(mapped).all(1)
    iu = np.full(len(mapped), -1, dtype=np.int64)
    iv = np.full(len(mapped), -1, dtype=np.int64)
    iu[finite_uv] = np.rint(mapped[finite_uv, 0]).astype(np.int64)
    iv[finite_uv] = np.rint(mapped[finite_uv, 1]).astype(np.int64)
    finite_source = np.isfinite(world[source]).all(1)
    good = visible[source] & finite_uv & finite_source
    good &= (iu >= 0) & (iu < W) & (iv >= 0) & (iv < H)
    ids = np.flatnonzero(good)
    if not len(ids):
        return ids, np.empty(0, np.int64), np.empty(0, np.int64)
    linear = iv[ids] * W + iu[ids]
    distance = (mapped[ids, 0] - iu[ids]) ** 2 + (mapped[ids, 1] - iv[ids]) ** 2
    R = np.asarray(viewpoint["R"], dtype=np.float64)
    translation = np.asarray(viewpoint["T"], dtype=np.float64)
    view_depth = (world[source, ids] @ R + translation)[:, 2]
    order = np.lexsort((view_depth, distance, linear))
    sorted_linear = linear[order]
    first = np.r_[True, sorted_linear[1:] != sorted_linear[:-1]]
    chosen = ids[order[first]]
    return chosen, iv[chosen], iu[chosen]


class DynamicReplicaDataset:
    """Protocol-v1 consumer for left- or right-camera Dynamic Replica caches."""

    def __init__(self, cache_root: str | Path, split: str = "train") -> None:
        self.root = Path(cache_root)
        index = self.root / "splits" / f"{split}.jsonl"
        if not index.exists():
            raise FileNotFoundError(f"Dynamic Replica cache index is missing: {index}")
        self.rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line]
        state_path = self.root / "PREPROCESSING_STATE.json"
        if not state_path.exists():
            raise FileNotFoundError(f"Dynamic Replica preprocessing state is missing: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        self.raw_train_root = Path(state["raw_root"]) / "train"
        self._clips: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self._max_clip_cache = 2
        self._clip_lock = threading.RLock()
        # Different clips may decode concurrently.  A single lock around all 21
        # torch.load calls would silently serialize the geometry prefetch pool;
        # per-clip locks only coalesce duplicate requests for the same clip.
        self._clip_load_locks: dict[str, threading.Lock] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def clip_id(self, index: int) -> str:
        return str(self.rows[index]["clip_id"])

    def _load_clip(self, row: dict[str, Any]) -> dict[str, np.ndarray]:
        key = str(row["clip_id"])
        with self._clip_lock:
            cached = self._clips.get(key)
            if cached is not None:
                self._clips.move_to_end(key)
                return cached
            load_lock = self._clip_load_locks.setdefault(key, threading.Lock())

        with load_lock:
            # Another worker may have completed this clip while we waited.
            with self._clip_lock:
                cached = self._clips.get(key)
                if cached is not None:
                    self._clips.move_to_end(key)
                    return cached

            # Do expensive independent file reads outside the global LRU lock.
            uv, world, visible, instances = [], [], [], []
            expected_points: int | None = None
            for frame in row["frames"]:
                path = self.raw_train_root / frame["trajectory"]
                value = torch.load(path, map_location="cpu", weights_only=True)
                n = int(value["traj_3d_world"].shape[0])
                if expected_points is None:
                    expected_points = n
                if n != expected_points or int(value["traj_2d"].shape[0]) != expected_points:
                    raise ValueError(f"track count changed inside {key}: {path}")
                uv.append(value["traj_2d"][:, :2].numpy())
                world.append(value["traj_3d_world"].numpy())
                visible.append(value["verts_inds_vis"].numpy())
                instances.append(value["instances"].numpy())
            annotation = {
                "trajs_2d": np.stack(uv),
                "trajs_3d_world": np.stack(world),
                "visible": np.stack(visible).astype(bool),
                "instances": np.stack(instances),
            }
            # Point indices must retain object identity across the clip.
            if not np.all(annotation["instances"] == annotation["instances"][0:1]):
                raise ValueError(f"track instance identity changed inside {key}")
            for value in annotation.values():
                value.setflags(write=False)

            with self._clip_lock:
                self._clips[key] = annotation
                self._clips.move_to_end(key)
                while len(self._clips) > self._max_clip_cache:
                    self._clips.popitem(last=False)
                return annotation

    def rgb(self, index: int) -> np.ndarray:
        row = self.rows[index]
        return np.stack([_rgb(self.raw_train_root / frame["rgb"]) for frame in row["frames"]])

    def camera(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        row = self.rows[index]
        intrinsics = np.stack([_pixel_intrinsics(frame["viewpoint"]) for frame in row["frames"]])
        camera_to_world = np.stack([_camera_to_world(frame["viewpoint"]) for frame in row["frames"]])
        depth, depth_valid = zip(*[_depth(self.raw_train_root / frame["depth"]) for frame in row["frames"]])
        return intrinsics, camera_to_world, np.stack(depth), np.stack(depth_valid)

    def source_all_targets(self, index: int, source: int) -> tuple[np.ndarray, np.ndarray]:
        xyz, valid, _ = self.source_all_targets_with_visibility(index, source)
        return xyz, valid

    def source_all_targets_with_visibility(self, index: int, source: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not 0 <= int(source) < T:
            raise ValueError(f"source must be in [0,20], got {source}")
        source = int(source)
        row = self.rows[index]
        annotation = self._load_clip(row)
        world = annotation["trajs_3d_world"].astype(np.float64)
        visible = annotation["visible"]
        finite_world = np.isfinite(world).all(2)
        viewpoint = row["frames"][source]["viewpoint"]
        chosen, ys, xs = _select_source_tracks(annotation, source, viewpoint)

        xyz = np.zeros((T, 3, H, W), dtype=np.float32)
        out_valid = np.zeros((T, H, W), dtype=bool)
        out_visible = np.zeros_like(out_valid)
        if len(chosen):
            R = np.asarray(viewpoint["R"], dtype=np.float64)
            translation = np.asarray(viewpoint["T"], dtype=np.float64)
            points_p3d = np.einsum("tnc,cm->tnm", world[:, chosen, :], R) + translation
            points = points_p3d * _P3D_TO_PROTOCOL[None, None, :]
            target_valid = finite_world[:, chosen] & np.isfinite(points).all(2)
            xyz[:, :, ys, xs] = np.transpose(points.astype(np.float32), (0, 2, 1))
            out_valid[:, ys, xs] = target_valid
            out_visible[:, ys, xs] = visible[:, chosen] & target_valid

        # The protocol diagonal is the point observed at the output pixel
        # centre.  Replace only the source diagonal with canonical resized-depth
        # backprojection; off-diagonal identity remains the persistent track.
        depth, depth_valid = _depth(self.raw_train_root / row["frames"][source]["depth"])
        K = _pixel_intrinsics(row["frames"][source]["viewpoint"])
        track_ys, track_xs = np.where(out_valid[source])
        if len(track_ys):
            source_ok = depth_valid[track_ys, track_xs]
            if np.any(~source_ok):
                out_valid[:, track_ys[~source_ok], track_xs[~source_ok]] = False
                out_visible[:, track_ys[~source_ok], track_xs[~source_ok]] = False
        # Diagonal correspondence is known for every valid source-depth pixel,
        # not only pixels carrying an off-diagonal sparse trajectory.
        ys, xs = np.where(depth_valid)
        d = depth[ys, xs]
        xyz[source, 0, ys, xs] = (xs - K[0, 2]) * d / K[0, 0]
        xyz[source, 1, ys, xs] = -(ys - K[1, 2]) * d / K[1, 1]
        xyz[source, 2, ys, xs] = -d
        out_valid[source, ys, xs] = True
        out_visible[source, ys, xs] = True
        return xyz, out_valid, out_visible

    def clean_latent(self, index: int) -> np.ndarray:
        """Load a global-order latent while exposing split-local indexing."""
        from safetensors import safe_open
        global_index = int(self.rows[index]["index"])
        latent_root = self.root / "latents" / "wan2.1_1.3b_fp32"
        for path in sorted(latent_root.glob("*.safetensors")):
            match = re.fullmatch(r"shard_(\d+)_(\d+)", path.stem)
            if match is None:
                continue
            first, count = map(int, match.groups())
            if first <= global_index < first + count:
                with safe_open(str(path), framework="np") as f:
                    return f.get_tensor("latents")[global_index - first]
        raise IndexError(f"no Dynamic Replica latent shard for global index {global_index}")
