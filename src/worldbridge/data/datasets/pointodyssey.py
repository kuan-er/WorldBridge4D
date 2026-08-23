"""Read-only WorldBridge4D v1 adapter for PointOdyssey caches."""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
import threading
from typing import Any

import numpy as np
from PIL import Image

from ..constants import (
    POINTODYSSEY_ANNO_CACHE,
    POINTODYSSEY_ANNO_NPY_CACHE,
    POINTODYSSEY_DEPTH_CACHE,
)

T, H, W = 21, 128, 128
RAW_W, RAW_H, CROP_X, CROP_Y, CROP_SIZE = 960, 540, 210, 0, 540
D = np.diag([1.0, -1.0, -1.0, 1.0])
DEPTH_SCALE = np.float32(1000.0 / 65535.0)
# Optional compressed and mmap-friendly annotation caches. The defaults are
# durable; callers may override them through the dataset configuration.
ANNO_CACHE = POINTODYSSEY_ANNO_CACHE
ANNO_NPY_CACHE = POINTODYSSEY_ANNO_NPY_CACHE


def _rgb(path: Path, image_size: int = W) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.asarray(im.crop((CROP_X, CROP_Y, CROP_X + CROP_SIZE, CROP_SIZE)).resize(
        (image_size, image_size), Image.Resampling.BICUBIC), np.uint8)


DEPTH_CACHE_ROOT = POINTODYSSEY_DEPTH_CACHE


def _depth(path: Path, image_size: int = W,
           cache_root: str | Path = DEPTH_CACHE_ROOT) -> tuple[np.ndarray, np.ndarray]:
    p = Path(path)
    cached = Path(cache_root) / p.parent.parent.name / "depths" / p.name
    if cached.is_file():
        p = cached
    raw = np.asarray(Image.open(p), dtype=np.uint16)[CROP_Y:CROP_Y + CROP_SIZE, CROP_X:CROP_X + CROP_SIZE]
    raw = np.asarray(Image.fromarray(raw).resize((image_size, image_size), Image.Resampling.NEAREST), dtype=np.uint16)
    depth = raw.astype(np.float32) * DEPTH_SCALE
    return depth, (raw > 0) & np.isfinite(depth) & (depth > 0)


class PointOdysseyDataset:
    """Dataset API required by protocol v1.

    The source release contains sparse tracks.  The adapter rasterizes the
    source-coordinate track to the 128x128 grid and leaves all other cells
    invalid.  ``visibs`` is exposed only through ``source_all_targets_with_visibility``;
    it never masks canonical XYZ validity.
    """
    def __init__(self, cache_root: str | Path, split: str = "train", image_size: int = W,
                 raw_root: str | Path | None = None,
                 annotation_cache_root: str | Path = ANNO_CACHE,
                 annotation_npy_cache_root: str | Path = ANNO_NPY_CACHE,
                 depth_cache_root: str | Path = DEPTH_CACHE_ROOT) -> None:
        self.root = Path(cache_root)
        self.image_size = int(image_size)
        self.raw_root = Path(raw_root).resolve() if raw_root is not None else None
        self.annotation_cache_root = Path(annotation_cache_root)
        self.annotation_npy_cache_root = Path(annotation_npy_cache_root)
        self.depth_cache_root = Path(depth_cache_root)
        if self.image_size not in (128, 256):
            raise ValueError("PointOdyssey adapter supports only audited 128 or 256 grids")
        index = self.root / "splits" / f"{split}.jsonl"
        if not index.exists():
            raise FileNotFoundError(f"PointOdyssey cache index is missing: {index}")
        self.rows = [json.loads(x) for x in index.read_text().splitlines() if x]
        if self.raw_root is not None:
            for row in self.rows:
                row["source_scene"] = str(self.raw_root / split / Path(row["source_scene"]).name)
        # An annotation can be hundreds of MB.  Keep a small per-consumer LRU;
        # the DDP geometry workers each create their own dataset instance.
        self._anno: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self._max_scene_cache = 109
        self._anno_lock = threading.RLock()

    def __len__(self) -> int:
        return len(self.rows)

    def clip_id(self, index: int) -> str:
        return self.rows[index]["clip_id"]

    def _load(self, row: dict[str, Any]) -> dict[str, np.ndarray]:
        scene = row["source_scene"]
        # Share decompressed scene arrays across geometry workers.  Without
        # this lock, eight workers can independently read/decompress the same
        # multi-hundred-MiB NPZ when a batch is scene-local.
        with self._anno_lock:
            if scene not in self._anno:
                npy_dir = self.annotation_npy_cache_root / Path(scene).name
                if npy_dir.is_dir() and all((npy_dir / f"{k}.npy").is_file() for k in ("trajs_2d", "trajs_3d", "valids", "visibs", "intrinsics", "extrinsics")):
                    value = {k: np.load(npy_dir / f"{k}.npy", mmap_mode="r")
                             for k in ("trajs_2d", "trajs_3d", "valids", "visibs", "intrinsics", "extrinsics")}
                else:
                    cached = self.annotation_cache_root / f"{Path(scene).name}.npz"
                    if cached.is_file():
                        with np.load(cached) as z:
                            value = {k: z[k] for k in ("trajs_2d", "trajs_3d", "valids", "visibs", "intrinsics", "extrinsics")}
                    else:
                        with np.load(Path(scene) / "anno.npz") as z:
                            value = {k: z[k] for k in ("trajs_2d", "trajs_3d", "valids", "visibs", "intrinsics", "extrinsics")}
                self._anno[scene] = value
                self._anno.move_to_end(scene)
                while len(self._anno) > self._max_scene_cache:
                    self._anno.popitem(last=False)
            else:
                self._anno.move_to_end(scene)
            return self._anno[scene]

    def rgb(self, index: int) -> np.ndarray:
        row = self.rows[index]
        scene, start = Path(row["source_scene"]), int(row["start"])
        return np.stack([_rgb(scene / "rgbs" / f"rgb_{start + j:05d}.jpg", self.image_size) for j in range(T)])

    def source_rgb(self, index: int, source: int) -> np.ndarray:
        source = int(source)
        if not 0 <= source < T:
            raise ValueError(f"source must be in [0,20], got {source}")
        row = self.rows[int(index)]
        scene, start = Path(row["source_scene"]), int(row["start"])
        return _rgb(scene / "rgbs" / f"rgb_{start + source:05d}.jpg", self.image_size)

    def camera(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        row, a = self.rows[index], self._load(self.rows[index])
        start = int(row["start"])
        scene = Path(row["source_scene"])
        size = self.image_size
        depth, depth_valid = zip(*[
            _depth(
                scene / "depths" / f"depth_{start+j:05d}.png",
                size,
                getattr(self, "depth_cache_root", DEPTH_CACHE_ROOT),
            )
            for j in range(T)
        ])
        # PointOdyssey intrinsics are constant here, but preserve all 3x3 terms.
        K = a["intrinsics"][start:start + T].astype(np.float64).copy()
        S = np.array([[size / CROP_SIZE, 0, -CROP_X * size / CROP_SIZE], [0, size / CROP_SIZE, 0], [0, 0, 1]], np.float64)
        K = np.einsum("ij,tjk->tik", S, K)
        c2w = np.linalg.inv(np.einsum("ij,tjk->tik", D, a["extrinsics"][start:start + T].astype(np.float64)))
        return K, c2w, np.stack(depth), np.stack(depth_valid)

    def source_all_targets(self, index: int, source: int) -> tuple[np.ndarray, np.ndarray]:
        xyz, valid, _ = self.source_all_targets_with_visibility(index, source)
        return xyz, valid

    def source_all_targets_with_visibility(self, index: int, source: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not 0 <= int(source) < T:
            raise ValueError(f"source must be in [0,20], got {source}")
        row, a = self.rows[index], self._load(self.rows[index])
        start, f = int(row["start"]), int(row["start"]) + int(source)
        uv = a["trajs_2d"][f].astype(np.float64)
        world = a["trajs_3d"][start:start + T].astype(np.float64)
        valid = a["valids"][start:start + T].astype(bool)
        vis = a["visibs"][start:start + T].astype(bool)
        size = getattr(self, "image_size", W)
        pu = (uv[:, 0] - CROP_X + 0.5) * size / CROP_SIZE - 0.5
        pv = (uv[:, 1] - CROP_Y + 0.5) * size / CROP_SIZE - 0.5
        finite_uv = np.isfinite(uv).all(1)
        iu = np.full(len(uv), -1, dtype=np.int64)
        iv = np.full(len(uv), -1, dtype=np.int64)
        iu[finite_uv] = np.rint(pu[finite_uv]).astype(np.int64)
        iv[finite_uv] = np.rint(pv[finite_uv]).astype(np.int64)
        # A source-grid anchor must denote the surface actually observed at
        # that source pixel. Keep target-time occluded-but-valid supervision,
        # but never anchor a trajectory that is already occluded at source.
        src_vis = vis[source]
        if not src_vis.any():
            # Rare fully-occluded clips (~0.34% of PointOdyssey) have no visible
            # source-frame track.  Fall back to validity so they still yield
            # supervision instead of raising on an empty anchor set.
            src_vis = valid[source]
        good = valid[source] & src_vis & finite_uv & np.isfinite(world).all((0, 2))
        good &= (iu >= 0) & (iu < size) & (iv >= 0) & (iv < size)
        xyz = np.zeros((T, 3, size, size), np.float32); out_valid = np.zeros((T, size, size), bool); out_vis = np.zeros_like(out_valid)
        E = a["extrinsics"][f, :3].astype(np.float64)
        ids = np.flatnonzero(good)
        if len(ids):
            # Deterministic nearest-track-per-pixel selection without a Python
            # loop over tracks.  Lexicographic sort uses raster pixel first,
            # then subpixel distance, so the first item in each run wins.
            linear = iv[ids] * size + iu[ids]
            distance = (pu[ids] - iu[ids]) ** 2 + (pv[ids] - iv[ids]) ** 2
            order = np.lexsort((distance, linear))
            sorted_linear = linear[order]
            first = np.r_[True, sorted_linear[1:] != sorted_linear[:-1]]
            chosen = ids[order[first]]
            ys, xs = iv[chosen], iu[chosen]
            points = world[:, chosen, :]
            q = np.einsum("tnc,mc->tnm", points, E[:, :3]) + E[:, 3]
            q *= np.array([1.0, -1.0, -1.0], dtype=np.float64)[None, None, :]
            g = valid[:, chosen] & np.isfinite(q).all(2)
            xyz[:, :, ys, xs] = np.transpose(q.astype(np.float32), (0, 2, 1))
            out_valid[:, ys, xs] = g
            out_vis[:, ys, xs] = vis[:, chosen] & g
        # Force the diagonal to the canonical source-depth backprojection. This
        # makes the interchange identity exact while preserving occluded-valid
        # off-diagonal tracks from PointOdyssey.
        # Preserve the diagonal identity using one source depth/K load.  Do
        # not call camera() inside the pixel loop: that would decode all 21
        # depth images once per sparse track.
        scene = Path(row["source_scene"])
        depth_path = scene / "depths" / f"depth_{f:05d}.png"
        depth, depth_valid = _depth(
            depth_path, size, getattr(self, "depth_cache_root", DEPTH_CACHE_ROOT)
        )
        K = a["intrinsics"][f].astype(np.float64).copy()
        S = np.array([[size / CROP_SIZE, 0, -CROP_X * size / CROP_SIZE], [0, size / CROP_SIZE, 0], [0, 0, 1]], np.float64)
        K = S @ K
        ys, xs = np.where(out_valid[source])
        if len(ys):
            d = depth[ys, xs]
            keep = depth_valid[ys, xs] & np.isfinite(d)
            if np.any(~keep):
                out_valid[source, ys[~keep], xs[~keep]] = False
                out_vis[source, ys[~keep], xs[~keep]] = False
            ys, xs, d = ys[keep], xs[keep], d[keep]
            xyz[source, 0, ys, xs] = (xs - K[0, 2]) * d / K[0, 0]
            xyz[source, 1, ys, xs] = -(ys - K[1, 2]) * d / K[1, 1]
            xyz[source, 2, ys, xs] = -d
        return xyz, out_valid, out_vis

    def clean_latent(self, index: int) -> np.ndarray:
        """Load the tensor-only latent shard in the manifest's clip order."""
        from safetensors import safe_open
        for p in sorted((self.root / "latents" / "wan2.1_1.3b_fp32").glob("*.safetensors")):
            first, last = [int(x) for x in p.stem.split("-")[1:]]
            if first <= index <= last:
                with safe_open(str(p), framework="np") as f:
                    return f.get_tensor("latent")[index - first]
        raise IndexError(f"no latent shard for index {index}")
