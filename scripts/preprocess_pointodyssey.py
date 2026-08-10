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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

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
        required = [p / "anno.npz", p / "info.npz", p / "rgbs", p / "depths", p / "masks"]
        if not all(x.exists() for x in required):
            continue
        with np.load(p / "info.npz") as info:
            shape = tuple(int(x) for x in np.asarray(info["trajs_3d"]).reshape(-1))
        # Some official train scenes contain a scalar placeholder instead of
        # 3D trajectories. Protocol v1 requires rejection, never invented XYZ.
        if len(shape) != 3 or shape[-1] != 3:
            continue
        result.append(p)
    return result


def rejected_scenes(raw_root: Path, split: str) -> list[dict[str, Any]]:
    accepted = {p.name for p in scenes(raw_root, split)}
    result = []
    for p in sorted((raw_root / split).iterdir(), key=lambda x: x.name):
        if p.is_dir() and (p / "rgbs").is_dir() and p.name not in accepted:
            result.append({"split": split, "parent_id": p.name, "clips": frame_count(p) // T,
                           "reason": "missing_or_invalid_dense_3d_trajectories"})
    return result


def frame_count(scene: Path) -> int:
    return len([x for x in (scene / "rgbs").iterdir() if x.suffix.lower() in {".jpg", ".jpeg", ".png"}])


def make_index(raw_root: Path, split: str, max_clips: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene in scenes(raw_root, split):
        n = frame_count(scene)
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
    iu = np.where(np.isfinite(pu), np.rint(pu), -1).astype(np.int64)
    iv = np.where(np.isfinite(pv), np.rint(pv), -1).astype(np.int64)
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
    iu = np.where(np.isfinite(pu), np.rint(pu), -1).astype(np.int64)
    iv = np.where(np.isfinite(pv), np.rint(pv), -1).astype(np.int64)
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


def _scene_rgb_clips(scene: Path, starts: list[int]) -> dict[int, np.ndarray]:
    """Decode each source video once instead of reopening 21 JPEGs per clip."""
    result: dict[int, np.ndarray] = {}
    mp4 = scene.parent / f"{scene.name}.mp4"
    if mp4.exists():
        import imageio_ffmpeg
        import subprocess
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        proc = subprocess.Popen([ffmpeg, "-loglevel", "error", "-i", str(mp4), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
        frame_bytes = RAW_W * RAW_H * 3
        wanted = set(starts)
        frame = 0
        try:
            while True:
                raw = proc.stdout.read(frame_bytes)
                if len(raw) != frame_bytes:
                    break
                start = (frame // T) * T
                if start in wanted:
                    im = Image.frombytes("RGB", (RAW_W, RAW_H), raw)
                    j = frame - start
                    result.setdefault(start, np.empty((T, H, W, 3), np.uint8))[j] = np.asarray(im.crop((CROP_X, CROP_Y, CROP_X + CROP_SIZE, CROP_SIZE)).resize((W, H), Image.Resampling.BILINEAR), np.uint8)
                frame += 1
        finally:
            proc.stdout.close(); proc.wait()
        if len(result) == len(wanted):
            return result
    # Fallback for a partial release: still preserve exact uint8 hot RGB.
    for start in starts:
        result[start] = np.stack([resize_rgb(scene / "rgbs" / f"rgb_{start+j:05d}.jpg") for j in range(T)])
    return result


def build_latents(rows: list[dict[str, Any]], out: Path, checkpoint: Path, device: str, shard_size: int, batch_size: int) -> list[dict[str, Any]]:
    import torch
    from safetensors.torch import save_file
    from worldbridge.wan import WanVAEEncoder
    enc = WanVAEEncoder(checkpoint, device=torch.device(device), dtype=torch.float32)
    artifacts: list[dict[str, Any]] = []
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_scene.setdefault(row["source_scene"], []).append(row)
    pending: list[torch.Tensor] = []
    pending_base: int | None = None
    next_index = 0

    def flush() -> None:
        nonlocal pending, pending_base
        if not pending:
            return
        z = torch.cat(pending).contiguous()
        base = int(pending_base)
        path = out / "latents" / "wan2.1_1.3b_fp32" / f"shard-{base:06d}-{base+len(z)-1:06d}.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True); save_file({"latent": z}, str(path))
        artifacts.append(file_artifact("wan_latent_shard", None, path, out, first_clip_index=base, clip_count=len(z)))
        pending = []; pending_base = None

    for scene_name, scene_rows in by_scene.items():
        clips = _scene_rgb_clips(Path(scene_name), [int(r["start"]) for r in scene_rows])
        for mini in range(0, len(scene_rows), batch_size):
            mini_rows = scene_rows[mini:mini + batch_size]
            x = torch.from_numpy(np.stack([clips[int(r["start"])] for r in mini_rows])).permute(0, 1, 4, 2, 3).contiguous().to(enc.device)
            with torch.inference_mode(): z = enc(x).float().cpu()
            if tuple(z.shape[1:]) != (16, 6, 16, 16): raise RuntimeError(f"unexpected latent shape {tuple(z.shape)}")
            for one in z.split(1):
                if pending_base is None: pending_base = next_index
                pending.append(one)
                next_index += 1
                if len(pending) >= shard_size: flush()
        print(f"latents {next_index}/{len(rows)}", flush=True)
    flush()
    return artifacts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("/dataset/PointOdyssey"))
    ap.add_argument("--output-root", type=Path, default=Path("/dataset/PointOdyssey_worldbridge4d_v1"))
    ap.add_argument("--wan-checkpoint", type=Path, default=Path("/dataset/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=2029)
    ap.add_argument("--shard-size", type=int, default=256)
    ap.add_argument("--vae-batch-size", type=int, default=8)
    ap.add_argument("--max-clips", type=int, default=None, help="debug bound applied per split")
    ap.add_argument("--skip-latents", action="store_true")
    ap.add_argument("--skip-source-hash", action="store_true", help="debug only; leaves the source checksum gate failed")
    args = ap.parse_args()
    out = args.output_root
    if out.exists() and (out / "manifest.json").exists():
        raise SystemExit(f"refusing to mutate validated cache: {out}")
    out.mkdir(parents=True, exist_ok=True)
    all_rows: dict[str, list[dict[str, Any]]] = {}
    artifacts: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    global_offset = 0
    for split, name in (("train", "train"), ("val", "validation"), ("test", "test")):
        rows = make_index(args.data_root, split, args.max_clips)
        for local, row in enumerate(rows):
            row["index"] = local
            row["latent_index"] = global_offset + local
        global_offset += len(rows)
        rejected.extend(rejected_scenes(args.data_root, split))
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
            used_frames = np.concatenate([np.arange(int(r["start"]), int(r["start"]) + T) for r in scene_rows])
            for f in np.unique(used_frames):
                uv = t2[f]
                pu = (uv[:, 0] - CROP_X + 0.5) * W / CROP_SIZE - 0.5
                pv = (uv[:, 1] - CROP_Y + 0.5) * H / CROP_SIZE - 0.5
                iu = np.where(np.isfinite(pu), np.rint(pu), -1).astype(np.int64)
                iv = np.where(np.isfinite(pv), np.rint(pv), -1).astype(np.int64)
                good = valid[f] & np.isfinite(uv).all(1) & np.isfinite(t3[f]).all(1)
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
        latent_artifacts = build_latents(ordered, out, args.wan_checkpoint, args.device, args.shard_size, args.vae_batch_size)
        artifacts.extend(latent_artifacts)
    # A complete per-clip consumer can use source_all_targets without copying the
    # enormous dense tensor.  Geometry is generated deterministically from source
    # scene annotations; no hidden resize or temporal operation occurs here.
    report = {"protocol": "worldbridge4d.dataset.v1", "status": "not_validated", "gates": {},
        "accepted_clips": {k: len(v) for k, v in all_rows.items()},
        "rejected_scenes": rejected, "rejected_clip_count": int(sum(x["clips"] for x in rejected)), "notes": [
        "PointOdyssey is adapted as dense_xyz from identity-preserving sparse tracks.",
        "valid is valids, never visibs; visibs is retained only for evaluation.",
        "The current producer writes source references and computes geometry on demand; no FP16 dense tier is claimed.",
        "Real-Wan gradient/tiny-overfit gates require the repository consumer smoke and are not silently marked passed.",
    ]}
    report_path = out / "audit" / "validation_report.json"; report_path.parent.mkdir(parents=True, exist_ok=True); report_path.write_text(json.dumps(report, indent=2) + "\n")
    artifacts.append(file_artifact("validation_report", None, report_path, out))
    source_archive = args.data_root / "train.tar.gz"
    source_hash = "0" * 64 if args.skip_source_hash else (sha256(source_archive) if source_archive.exists() else "0" * 64)
    manifest = {"protocol": "worldbridge4d.dataset.v1", "dataset": {"id": "PointOdyssey", "version": "official-local-release", "source_uri": str(args.data_root), "source_sha256": source_hash, "license": None},
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
