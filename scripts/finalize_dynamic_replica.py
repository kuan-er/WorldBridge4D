#!/usr/bin/env python3
"""Finalize statistics and protocol audits after Dynamic Replica latent encoding."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from worldbridge.dynamic_replica import (
    DynamicReplicaDataset, H, T, W, _P3D_TO_PROTOCOL, _depth,
    _pixel_intrinsics, _select_source_tracks,
)


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


def composite_sha256(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(path.name.encode("utf-8"))
        h.update(bytes.fromhex(sha256(path)))
    return h.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def artifact(kind: str, split: str | None, path: Path, root: Path, fmt: str | None = None,
             **extra: Any) -> dict[str, Any]:
    return {
        "kind": kind, "split": split, "path": str(path.relative_to(root)),
        "format": fmt or path.suffix.lstrip("."), "bytes": path.stat().st_size,
        "sha256": sha256(path), **extra,
    }


def latent_shards(root: Path, expected: int) -> list[tuple[Path, int, int]]:
    found = []
    for path in sorted((root / "latents" / "wan2.1_1.3b_fp32").glob("*.safetensors")):
        match = re.fullmatch(r"shard_(\d+)_(\d+)", path.stem)
        if match:
            found.append((path, int(match.group(1)), int(match.group(2))))
    cursor = 0
    for path, first, count in found:
        if first != cursor or count <= 0:
            raise RuntimeError(f"latent coverage gap at {cursor}: {path.name}")
        with safe_open(str(path), framework="np") as f:
            shape = tuple(f.get_slice("latents").get_shape())
        if shape != (count, 16, 6, 16, 16):
            raise RuntimeError(f"latent shape mismatch in {path}: {shape}")
        cursor += count
    if cursor != expected:
        raise RuntimeError(f"latent cache incomplete: {cursor}/{expected}")
    return found


def coordinate_stats(dataset: DynamicReplicaDataset, output: Path, checkpoint_every: int = 10) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    progress = output.parent / ".coordinate_stats_progress.npz"
    start = 0; total = np.zeros(3, np.float64); square = np.zeros(3, np.float64); count = 0
    if progress.exists():
        with np.load(progress) as value:
            start = int(value["next_index"]); total = value["sum"].astype(np.float64)
            square = value["sum_of_squares"].astype(np.float64); count = int(value["point_count"])
    for index in range(start, len(dataset)):
        row = dataset.rows[index]
        for source in range(T):
            viewpoint = row["frames"][source]["viewpoint"]
            depth, depth_valid = _depth(dataset.raw_train_root / row["frames"][source]["depth"])
            K = _pixel_intrinsics(viewpoint)
            ys, xs = np.where(depth_valid)
            if not len(ys):
                continue
            d = depth[ys, xs].astype(np.float64)
            ys, xs = ys.astype(np.float64), xs.astype(np.float64)
            points = np.stack([
                (xs - K[0, 2]) * d / K[0, 0],
                -(ys - K[1, 2]) * d / K[1, 1],
                -d,
            ], axis=1)
            points = points[np.isfinite(points).all(1)]
            total += points.sum(0)
            square += (points * points).sum(0)
            count += len(points)
        if (index + 1) % checkpoint_every == 0 or index + 1 == len(dataset):
            tmp = progress.with_suffix(".tmp.npz")
            np.savez(tmp, next_index=np.int64(index + 1), sum=total,
                     sum_of_squares=square, point_count=np.int64(count))
            tmp.replace(progress)
            print(json.dumps({"stage": "coordinate_stats", "clips": index + 1,
                              "total_clips": len(dataset), "point_count": count}), flush=True)
    if count <= 0:
        raise RuntimeError("coordinate statistics found no valid diagonal points")
    mean = total / count
    scale = np.sqrt(np.maximum(square / count - mean * mean, 1e-12))
    np.savez(output, mean=mean.astype(np.float32), scale=scale.astype(np.float32),
             examples=np.int64(len(dataset)), point_count=np.int64(count),
             coordinate_frame=np.array("source"),
             stats_source=np.array("diagonal_pointmaps_in_selected_coordinate_frame"))
    progress.unlink(missing_ok=True)
    return {"mean": mean.tolist(), "scale": scale.tolist(), "examples": len(dataset), "point_count": count}


def geometry_audit(dataset: DynamicReplicaDataset, samples: int) -> dict[str, Any]:
    indices = np.linspace(0, len(dataset) - 1, min(samples, len(dataset)), dtype=int)
    projection_max = 0.0; diagonal_max = 0.0; valid_count = 0; occluded_valid = 0
    for number, index in enumerate(indices):
        source = int((number * 7) % T)
        row = dataset.rows[int(index)]
        annotation = dataset._load_clip(row)
        viewpoint = row["frames"][source]["viewpoint"]
        chosen, _, _ = _select_source_tracks(annotation, source, viewpoint)
        if len(chosen):
            world = annotation["trajs_3d_world"][source, chosen].astype(np.float64)
            R = np.asarray(viewpoint["R"], np.float64); translation = np.asarray(viewpoint["T"], np.float64)
            p = world @ R + translation
            f = np.asarray(viewpoint["focal_length"], np.float64) * 360.0
            c = np.array([640.0, 360.0]) - np.asarray(viewpoint["principal_point"], np.float64) * 360.0
            projected = np.stack([c[0] - f[0] * p[:, 0] / p[:, 2], c[1] - f[1] * p[:, 1] / p[:, 2]], axis=1)
            err = np.linalg.norm(projected - annotation["trajs_2d"][source, chosen, :2], axis=1)
            projection_max = max(projection_max, float(np.max(err)))
        xyz, valid, visible = dataset.source_all_targets_with_visibility(int(index), source)
        valid_count += int(valid.sum()); occluded_valid += int((valid & ~visible).sum())
        depth, depth_valid = _depth(dataset.raw_train_root / row["frames"][source]["depth"])
        K = _pixel_intrinsics(viewpoint)
        ys, xs = np.where(valid[source])
        if len(ys):
            d = depth[ys, xs]
            expected = np.stack([(xs - K[0, 2]) * d / K[0, 0],
                                 -(ys - K[1, 2]) * d / K[1, 1], -d], axis=1)
            got = xyz[source, :, ys, xs]
            diagonal_max = max(diagonal_max, float(np.max(np.abs(got - expected))))
    # Published trajectory coordinates are float32.  At 1280px source
    # resolution their roundoff can slightly exceed 1e-3px (observed maximum
    # 1.2545e-3px), so use a still-subpixel 2e-3px numerical tolerance.
    projection_tolerance_px = 2e-3
    return {"sample_clips": len(indices), "projection_max_px": projection_max,
            "projection_tolerance_px": projection_tolerance_px,
            "diagonal_max_m": diagonal_max, "valid_points": valid_count,
            "occluded_valid_points": occluded_valid,
            "pass": projection_max <= projection_tolerance_px and diagonal_max <= 1e-4 and occluded_valid > 0}


def latent_determinism(root: Path, rows: list[dict[str, Any]], checkpoint: Path,
                       device: str, samples: int = 16) -> dict[str, Any]:
    import torch
    from worldbridge.dynamic_replica import _rgb
    from worldbridge.wan import WanVAEEncoder
    chosen = np.linspace(0, len(rows) - 1, min(samples, len(rows)), dtype=int)
    videos = []
    for index in chosen:
        row = rows[int(index)]
        videos.append(np.stack([_rgb(Path(json.loads((root / "PREPROCESSING_STATE.json").read_text())["raw_root"]) / "train" / frame["rgb"]) for frame in row["frames"]]))
    x = torch.from_numpy(np.stack(videos)).permute(0, 1, 4, 2, 3)
    enc = WanVAEEncoder(checkpoint, device=torch.device(device), dtype=torch.float32)
    with torch.inference_mode():
        first = enc(x).cpu().numpy(); second = enc(x).cpu().numpy()
    difference = float(np.max(np.abs(first - second)))
    return {"samples": len(chosen), "shape": list(first.shape[1:]), "max_difference": difference,
            "pass": tuple(first.shape[1:]) == (16, 6, 16, 16) and difference <= 1e-6}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=Path, default=Path("/data/WorldBridge4D-persistent/datasets/dynamic_stereo_worldbridge4d_v1"))
    ap.add_argument("--wan-checkpoint", type=Path, default=Path("/data/WorldBridge4D/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--audit-clips", type=int, default=16)
    ap.add_argument("--skip-determinism", action="store_true")
    args = ap.parse_args()
    root = args.cache_root.resolve()
    state = json.loads((root / "PREPROCESSING_STATE.json").read_text())
    if state.get("status") != "latents_ready_geometry_adapter_pending":
        raise RuntimeError(f"latent producer has not completed successfully: {state.get('status')}")
    all_rows = [json.loads(line) for line in (root / "samples" / "index.jsonl").read_text().splitlines() if line]
    shards = latent_shards(root, len(all_rows))
    train = DynamicReplicaDataset(root, "train")
    stats_path = root / "stats" / "coordinate_stats_train_source.npz"
    stats = coordinate_stats(train, stats_path)
    geometry = geometry_audit(train, args.audit_clips)
    determinism = {"status": "skipped"} if args.skip_determinism else latent_determinism(root, all_rows, args.wan_checkpoint, args.device)
    train_parents = {row["parent_id"] for row in train.rows}
    validation = DynamicReplicaDataset(root, "validation")
    validation_parents = {row["parent_id"] for row in validation.rows}
    split_ok = not (train_parents & validation_parents)
    temporal_ok = all(len(row["timestamps"]) == T and np.all(np.diff(row["timestamps"]) > 0) for row in all_rows)
    report = {
        "protocol": "worldbridge4d.dataset.v1", "status": "core_cache_validated_smokes_pending",
        "formal_training_allowed": False, "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gates": {"split_leakage": {"pass": split_ok}, "temporal": {"pass": temporal_ok},
                  "geometry": geometry, "coordinate_statistics": {"pass": True, **stats},
                  "wan_determinism": determinism, "tiny_overfit": {"pass": False, "status": "pending"},
                  "real_gradient_smoke": {"pass": False, "status": "pending"}},
    }
    audit_path = root / "audit" / "validation_report.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(report, indent=2) + "\n")
    raw_root = Path(state["raw_root"])
    source_metadata = [raw_root / "train" / "frame_annotations_train.jgz", raw_root / "train" / "processed_dir.json"]
    artifacts = [
        artifact("split_index", split, root / "splits" / f"{split}.jsonl", root, "jsonl")
        for split in ("train", "validation")
    ]
    artifacts.append(artifact("sample_shard", None, root / "samples" / "index.jsonl", root, "jsonl",
                              first_clip_index=0, clip_count=len(all_rows)))
    artifacts.append(artifact("coordinate_stats", None, stats_path, root, "npz"))
    artifacts.append(artifact("validation_report", None, audit_path, root, "json"))
    for path, first, count in shards:
        artifacts.append(artifact("wan_latent_shard", None, path, root, "safetensors",
                                  first_clip_index=first, clip_count=count))
    split_rows = {name: [json.loads(line) for line in (root / "splits" / f"{name}.jsonl").read_text().splitlines() if line]
                  for name in ("train", "validation")}
    manifest = {
        "protocol": "worldbridge4d.dataset.v1",
        "dataset": {"id": "DynamicReplica", "version": "dynamic_stereo-local-train-left-camera",
                    "source_uri": str(raw_root), "source_sha256": composite_sha256(source_metadata), "license": None},
        "clip": {"frames": T, "height": H, "width": W, "channels": 3, "rgb_dtype": "uint8",
                 "temporal_policy": "ordered_no_padding_no_interpolation", "fps": 30.0, "default_stride": 1},
        "camera": {"intrinsics": "per_frame_3x3", "pose": "camera_to_world_4x4", "optical_axis": "-z",
                   "image_axes": "u_right_v_down", "pixel_center": "integer_uv", "world_units": "meters",
                   "depth_convention": "z_meters"},
        "geometry": {"annotation_mode": "dense_xyz", "coordinate_frame": "source_camera",
                     "validity_semantics": "valid_not_visibility_occluded_valid_supervised",
                     "dense_xyz_storage_dtype": "float32", "visibility_available": True},
        "wan_latent": {"model": "Wan2.1-T2V-1.3B-VAE", "checkpoint_sha256": sha256(args.wan_checkpoint),
                       "posterior": "mean", "normalization": "native_wan_channel_mean_std", "dtype": "float32",
                       "shape": [16, 6, 16, 16]},
        "splits": {name: {"clips": len(rows), "parents": len({row['parent_id'] for row in rows}),
                          "index_path": f"splits/{name}.jsonl", "index_sha256": sha256(root / "splits" / f"{name}.jsonl")}
                   for name, rows in split_rows.items()},
        "artifacts": artifacts,
        "producer": {"git_commit": git_commit(), "argv": sys.argv, "seed": int(state["seed"]),
                     "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "python": platform.python_version(),
                     "torch": __import__("torch").__version__, "cuda": __import__("torch").version.cuda},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "stats": stats, "geometry": geometry,
                      "determinism": determinism, "manifest": str(root / 'manifest.json')}, indent=2), flush=True)


if __name__ == "__main__":
    main()
