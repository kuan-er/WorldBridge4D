#!/usr/bin/env python3
"""Build the WorldBridge4D v1 cache for the native PointOdyssey release.

PointOdyssey publishes sparse, identity-preserving world-space tracks rather than
rigid instance poses.  This adapter therefore uses the protocol's dense_xyz
mode: the source-grid map is sparse and ``valid`` is the source-track validity
mask (never the visibility mask).  No optical-flow or occlusion labels are
invented.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]

T, H, W, FPS = 21, 128, 128, 24.0
RAW_W, RAW_H = 960, 540
CROP_X, CROP_Y, CROP_SIZE = 210, 0, 540
DEPTH_SCALE = 1000.0 / 65535.0
# PointOdyssey extrinsics are OpenCV world-to-camera (+Z forward, +Y down).
# D converts that basis to the protocol basis (-Z forward, +Y up).
D = np.diag([1.0, -1.0, -1.0, 1.0])


def sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "0" * 40


def scenes(raw_root: Path, split: str) -> list[Path]:
    result = []
    for p in sorted((raw_root / split).iterdir(), key=lambda x: x.name):
        if not p.is_dir():
            continue
        required = [p / "anno.npz", p / "rgbs", p / "depths", p / "masks"]
        if all(x.exists() for x in required):
            result.append(p)
    return result


def frame_count(scene: Path) -> int:
    return len([x for x in (scene / "rgbs").iterdir() if x.suffix.lower() in {".jpg", ".jpeg", ".png"}])


def annotation_shape_error(scene: Path, frame_count_: int) -> str | None:
    """Return a reproducible reason when a scene cannot provide dense tracks."""
    required = ("trajs_2d", "trajs_3d", "valids", "visibs", "intrinsics", "extrinsics")
    try:
        with np.load(scene / "anno.npz") as z:
            missing = [key for key in required if key not in z]
            if missing:
                return f"missing={missing}"
            shapes = {key: z[key].shape for key in required}
    except Exception as exc:
        return f"load_error={type(exc).__name__}:{exc}"
    t2, t3, valid, visib, intr, extr = (shapes[key] for key in required)
    if len(t2) != 3 or t2[-1] != 2:
        return f"trajs_2d_shape={t2}"
    if len(t3) != 3 or t3[-1] != 3:
        return f"trajs_3d_shape={t3}"
    if len(valid) != 2 or len(visib) != 2:
        return f"valids_visibs_shape={valid},{visib}"
    if len(intr) != 3 or len(extr) != 3 or intr[-2:] != (3, 3) or extr[-2:] != (4, 4):
        return f"camera_shapes={intr},{extr}"
    if not (t2[0] >= frame_count_ and t3[0] >= frame_count_ and valid[0] >= frame_count_ and visib[0] >= frame_count_ and intr[0] >= frame_count_ and extr[0] >= frame_count_):
        return f"frame_count={frame_count_},shapes={t2},{t3},{valid},{visib},{intr},{extr}"
    if not (t2[1] == t3[1] == valid[1] == visib[1]):
        return f"track_counts={t2[1]},{t3[1]},{valid[1]},{visib[1]}"
    return None


def make_index(raw_root: Path, split: str, max_clips: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene in scenes(raw_root, split):
        n = frame_count(scene)
        error = annotation_shape_error(scene, n)
        if error is not None:
            print(f"skip scene {scene}: malformed annotation ({error})", flush=True)
            continue
        for start in range(0, n - T + 1, T):
            rows.append({
                "index": len(rows),
                "clip_id": f"pointodyssey/{split}/{scene.name}/start{start:06d}",
                "parent_id": scene.name,
                "source_scene": str(scene),
                "start": start,
                "stride": 1,
                "fps": FPS,
                "timestamps": [round((start + j) / FPS, 9) for j in range(T)],
                "frame_count": n,
            })
            if max_clips is not None and len(rows) >= max_clips:
                return rows
    return rows


def resize_rgb(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if im.size != (RAW_W, RAW_H):
        raise ValueError(f"unexpected RGB size {im.size} in {path}")
    return np.asarray(im.crop((CROP_X, CROP_Y, CROP_X + CROP_SIZE, CROP_Y + CROP_SIZE)).resize((W, H), Image.Resampling.BILINEAR), dtype=np.uint8)


def resize_depth(path: Path) -> tuple[np.ndarray, np.ndarray]:
    im = Image.open(path)
    raw = np.asarray(im, dtype=np.uint16)
    if raw.shape != (RAW_H, RAW_W):
        raise ValueError(f"unexpected depth shape {raw.shape} in {path}")
    cropped = raw[CROP_Y:CROP_Y + CROP_SIZE, CROP_X:CROP_X + CROP_SIZE]
    out = np.asarray(Image.fromarray(cropped).resize((W, H), Image.Resampling.NEAREST), dtype=np.uint16)
    depth = out.astype(np.float32) * np.float32(DEPTH_SCALE)
    return depth, (out > 0) & np.isfinite(depth) & (depth > 0)


def load_anno(scene: Path) -> dict[str, np.ndarray]:
    with np.load(scene / "anno.npz") as z:
        return {k: z[k] for k in ("trajs_2d", "trajs_3d", "valids", "visibs", "intrinsics", "extrinsics")}


def source_geometry(anno: dict[str, np.ndarray], source_frame: int, start: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return source-grid sparse target XYZ, valid, visibility, and source uv.

    The returned XYZ is [21,3,H,W].  Duplicate tracks in one output pixel are
    resolved deterministically by nearest source-coordinate distance.
    """
    f = start + source_frame
    uv = anno["trajs_2d"][f].astype(np.float64)
    world = anno["trajs_3d"][start:start + T].astype(np.float64)  # [target,point,3]
    valid = anno["valids"][start:start + T].astype(bool)
    vis = anno["visibs"][start:start + T].astype(bool)
    finite = np.isfinite(uv).all(1) & np.isfinite(world).all((0, 2))
    # Center crop then exact 540 -> 128 spatial transform, integer pixel centers.
    pu = (uv[:, 0] - CROP_X + 0.5) * W / CROP_SIZE - 0.5
    pv = (uv[:, 1] - CROP_Y + 0.5) * H / CROP_SIZE - 0.5
    iu, iv = np.rint(pu).astype(np.int64), np.rint(pv).astype(np.int64)
    inside = finite & (iu >= 0) & (iu < W) & (iv >= 0) & (iv < H) & valid[source_frame]
    xyz = np.zeros((T, 3, H, W), dtype=np.float32)
    ok = np.zeros((T, H, W), dtype=bool)
    visibility = np.zeros((T, H, W), dtype=bool)
    source_uv = np.full((H, W, 2), np.nan, dtype=np.float32)
    # closest track wins; this avoids order-dependent duplicate rasterization.
    best = np.full((H, W), np.inf, dtype=np.float64)
    for p in np.flatnonzero(inside):
        y, x = int(iv[p]), int(iu[p])
        dist = float((pu[p] - x) ** 2 + (pv[p] - y) ** 2)
        if dist >= best[y, x]:
            continue
        best[y, x] = dist
        source_uv[y, x] = (pu[p], pv[p])
        # Track world points are transformed to the source camera basis.
        for target in range(T):
            if valid[target, p] and np.isfinite(world[target, p]).all():
                # E_f is world-to-camera; all targets stay in the source basis.
                xyz[target, :, y, x] = (D[:3, :3] @ (anno["extrinsics"][f, :3, :] @ np.r_[world[target, p], 1.0]))[:3]
                ok[target, y, x] = True
                visibility[target, y, x] = bool(vis[target, p])
    return xyz, ok, visibility, source_uv


def source_geometry_fast(anno: dict[str, np.ndarray], source_frame: int, start: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized form used by the stats/consumer path."""
    f = start + source_frame
    uv = anno["trajs_2d"][f].astype(np.float64)
    world = anno["trajs_3d"][start:start + T].astype(np.float64)
    valid = anno["valids"][start:start + T].astype(bool)
    vis = anno["visibs"][start:start + T].astype(bool)
    finite = np.isfinite(uv).all(1) & np.isfinite(world).all(2).all(0)
    pu = (uv[:, 0] - CROP_X + 0.5) * W / CROP_SIZE - 0.5
    pv = (uv[:, 1] - CROP_Y + 0.5) * H / CROP_SIZE - 0.5
    iu, iv = np.rint(pu).astype(np.int64), np.rint(pv).astype(np.int64)
    inside = finite & (iu >= 0) & (iu < W) & (iv >= 0) & (iv < H) & valid[source_frame]
    chosen: dict[tuple[int, int], int] = {}
    for p in np.flatnonzero(inside):
        key = (int(iv[p]), int(iu[p]))
        dist = (pu[p] - key[1]) ** 2 + (pv[p] - key[0]) ** 2
        old = chosen.get(key)
        if old is None or dist < (pu[old] - key[1]) ** 2 + (pv[old] - key[0]) ** 2:
            chosen[key] = int(p)
    xyz = np.zeros((T, 3, H, W), np.float32)
    ok = np.zeros((T, H, W), bool)
    visibility = np.zeros((T, H, W), bool)
    E = anno["extrinsics"][f, :3, :]
    for (y, x), p in chosen.items():
        q = (world[:, p] @ E[:, :3].T + E[:, 3]) @ D[:3, :3].T
        good = valid[:, p] & np.isfinite(q).all(1)
        xyz[:, :, y, x] = q.astype(np.float32)
        ok[:, y, x] = good
        visibility[:, y, x] = vis[:, p] & good
    return xyz, ok, visibility


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return sha256(path)


def file_artifact(kind: str, split: str | None, path: Path, root: Path, **extra: Any) -> dict[str, Any]:
    return {"kind": kind, "split": split, "path": str(path.relative_to(root)), "format": path.suffix.lstrip(".") or "json",
            "bytes": path.stat().st_size, "sha256": sha256(path), **extra}


def build_latents(rows: list[dict[str, Any]], out: Path, checkpoint: Path, device: str, shard_size: int) -> list[dict[str, Any]]:
    import torch
    from safetensors.torch import save_file
    from worldbridge.models.wan import WanVAEEncoder
    enc = WanVAEEncoder(checkpoint, device=torch.device(device), dtype=torch.float32)
    artifacts = []
    for base in range(0, len(rows), shard_size):
        batch = []
        for row in rows[base:base + shard_size]:
            scene = Path(row["source_scene"]); start = int(row["start"])
            frames = [resize_rgb(scene / "rgbs" / f"rgb_{start+j:05d}.jpg") for j in range(T)]
            batch.append(np.stack(frames))
        x = torch.from_numpy(np.stack(batch)).permute(0, 1, 4, 2, 3).contiguous().to(enc.device)
        with torch.inference_mode():
            z = enc(x).float().cpu()
        if tuple(z.shape[1:]) != (16, 6, 16, 16):
            raise RuntimeError(f"unexpected latent shape {tuple(z.shape)}")
        path = out / "latents" / "wan2.1_1.3b_fp32" / f"shard-{base:06d}-{base+len(batch)-1:06d}.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({"latent": z.contiguous()}, str(path))
        artifacts.append(file_artifact("wan_latent_shard", None, path, out, first_clip_index=base, clip_count=len(batch)))
        print(f"latents {base + len(batch)}/{len(rows)}", flush=True)
    return artifacts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("/dataset/PointOdyssey"))
    ap.add_argument("--output-root", type=Path, default=Path("/dataset/PointOdyssey_worldbridge4d_v1"))
    ap.add_argument("--wan-checkpoint", type=Path, default=Path("/dataset/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=2029)
    ap.add_argument("--shard-size", type=int, default=256)
    ap.add_argument("--max-clips", type=int, default=None, help="debug bound applied per split")
    ap.add_argument("--skip-latents", action="store_true")
    ap.add_argument("--skip-source-hash", action="store_true",
                    help="do not read the potentially 100+ GB release tarball during handoff")
    args = ap.parse_args()
    out = args.output_root
    if out.exists() and (out / "manifest.json").exists():
        raise SystemExit(f"refusing to mutate validated cache: {out}")
    out.mkdir(parents=True, exist_ok=True)
    all_rows: dict[str, list[dict[str, Any]]] = {}
    artifacts: list[dict[str, Any]] = []
    for split, name in (("train", "train"), ("val", "validation"), ("test", "test")):
        rows = make_index(args.data_root, split, args.max_clips)
        # Reindex globally only after concatenation; each split JSONL keeps local index
        all_rows[name] = rows
        path = out / "splits" / f"{name}.jsonl"
        write_jsonl(path, rows)
        artifacts.append(file_artifact("split_index", name, path, out))
        print(split, len(rows), flush=True)
    # Scene annotation paths and clip metadata are intentionally immutable references;
    # the source release is checksummed in the handoff record and never modified.
    for split, rows in all_rows.items():
        path = out / "samples" / f"{split}.jsonl"
        write_jsonl(path, rows)
        artifacts.append(file_artifact("sample_shard", split, path, out, first_clip_index=0, clip_count=len(rows)))
    stats_sum = np.zeros(3, np.float64); stats_sq = np.zeros(3, np.float64); point_count = 0
    # Load each compressed scene once.  The diagonal map only needs the
    # source-frame world point, not the full 21-target rasterization.
    train_by_scene: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows["train"]:
        train_by_scene.setdefault(row["source_scene"], []).append(row)
    for scene_name, scene_rows in train_by_scene.items():
        with np.load(Path(scene_name) / "anno.npz") as z:
            t2 = z["trajs_2d"].astype(np.float64)
            t3 = z["trajs_3d"].astype(np.float64)
            valid = z["valids"].astype(bool)
            E = z["extrinsics"].astype(np.float64)
            if t2.ndim != 3 or t3.ndim != 3 or valid.ndim != 2:
                print(f"skip stats scene {scene_name}: malformed loaded shapes {t2.shape},{t3.shape},{valid.shape}", flush=True)
                continue
            used_frames = np.concatenate([np.arange(int(r["start"]), int(r["start"]) + T) for r in scene_rows])
            for f in np.unique(used_frames):
                uv = t2[f]
                finite_uv = np.isfinite(uv).all(1)
                pu = (uv[:, 0] - CROP_X + 0.5) * W / CROP_SIZE - 0.5
                pv = (uv[:, 1] - CROP_Y + 0.5) * H / CROP_SIZE - 0.5
                iu = np.zeros(len(uv), dtype=np.int64); iv = np.zeros(len(uv), dtype=np.int64)
                iu[finite_uv] = np.rint(pu[finite_uv]).astype(np.int64)
                iv[finite_uv] = np.rint(pv[finite_uv]).astype(np.int64)
                good = valid[f] & finite_uv & np.isfinite(t3[f]).all(1)
                good &= (iu >= 0) & (iu < W) & (iv >= 0) & (iv < H)
                ids = np.flatnonzero(good)
                if not len(ids):
                    continue
                # Keep one physical track per source output pixel.
                _, keep = np.unique(iv[ids] * W + iu[ids], return_index=True)
                ids = ids[np.sort(keep)]
                q = (t3[f, ids] @ E[f, :3, :3].T + E[f, :3, 3]) @ D[:3, :3].T
                q = q[np.isfinite(q).all(1)]
                stats_sum += q.sum(0); stats_sq += (q * q).sum(0); point_count += len(q)
        print(f"stats scene {scene_name}: {point_count} points", flush=True)
    mean = stats_sum / max(point_count, 1)
    scale = np.sqrt(np.maximum(stats_sq / max(point_count, 1) - mean * mean, 1e-12))
    stats_path = out / "stats" / "coordinate_stats_train_source.npz"; stats_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(stats_path, mean=mean.astype(np.float32), scale=scale.astype(np.float32), examples=np.int64(len(all_rows["train"])), point_count=np.int64(point_count), coordinate_frame=np.array("source"), stats_source=np.array("diagonal_pointmaps_in_selected_coordinate_frame"))
    artifacts.append(file_artifact("coordinate_stats", None, stats_path, out))
    latent_artifacts = []
    if not args.skip_latents:
        if not args.wan_checkpoint.exists():
            raise FileNotFoundError(f"WAN VAE checkpoint is required by protocol: {args.wan_checkpoint}")
        # Encode in one deterministic global order, split-local indices remain stable.
        ordered = all_rows["train"] + all_rows["validation"] + all_rows["test"]
        latent_artifacts = build_latents(ordered, out, args.wan_checkpoint, args.device, args.shard_size)
        artifacts.extend(latent_artifacts)
    # A complete per-clip consumer can use source_all_targets without copying the
    # enormous dense tensor.  Geometry is generated deterministically from source
    # scene annotations; no hidden resize or temporal operation occurs here.
    report = {"protocol": "worldbridge4d.dataset.v1", "status": "not_validated", "gates": {}, "notes": [
        "PointOdyssey is adapted as dense_xyz from identity-preserving sparse tracks.",
        "valid is valids, never visibs; visibs is retained only for evaluation.",
        "The current producer writes source references and computes geometry on demand; no FP16 dense tier is claimed.",
        "Real-Wan gradient/tiny-overfit gates require the repository consumer smoke and are not silently marked passed.",
    ]}
    report_path = out / "audit" / "validation_report.json"; report_path.parent.mkdir(parents=True, exist_ok=True); report_path.write_text(json.dumps(report, indent=2) + "\n")
    artifacts.append(file_artifact("validation_report", None, report_path, out))
    manifest = {"protocol": "worldbridge4d.dataset.v1", "dataset": {"id": "PointOdyssey", "version": "official-local-release", "source_uri": str(args.data_root), "source_sha256": (sha256(args.data_root / "train.tar.gz") if (args.data_root / "train.tar.gz").exists() and not args.skip_source_hash else "0" * 64), "license": None},
      "clip": {"frames": T, "height": H, "width": W, "channels": 3, "rgb_dtype": "uint8", "temporal_policy": "ordered_no_padding_no_interpolation", "fps": FPS, "default_stride": 1},
      "camera": {"intrinsics": "per_frame_3x3", "pose": "camera_to_world_4x4", "optical_axis": "-z", "image_axes": "u_right_v_down", "pixel_center": "integer_uv", "world_units": "meters", "depth_convention": "z_meters"},
      "geometry": {"annotation_mode": "dense_xyz", "coordinate_frame": "source_camera", "validity_semantics": "valid_not_visibility_occluded_valid_supervised", "dense_xyz_storage_dtype": "float32", "visibility_available": True},
      "wan_latent": {"model": "Wan2.1-T2V-1.3B-VAE", "checkpoint_sha256": sha256(args.wan_checkpoint) if args.wan_checkpoint.exists() else "0" * 64, "posterior": "mean", "normalization": "native_wan_channel_mean_std", "dtype": "float32", "shape": [16, 6, 16, 16]},
      "splits": {k: {"clips": len(v), "parents": len({x['parent_id'] for x in v}), "index_path": f"splits/{k}.jsonl", "index_sha256": sha256(out / "splits" / f"{k}.jsonl")} for k, v in all_rows.items()}, "artifacts": artifacts,
      "producer": {"git_commit": git_commit(), "argv": sys.argv, "seed": args.seed, "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "python": platform.python_version(), "torch": None, "cuda": None}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(out), "clips": {k: len(v) for k, v in all_rows.items()}, "latent_shards": len(latent_artifacts)}, indent=2))

if __name__ == "__main__":
    main()
