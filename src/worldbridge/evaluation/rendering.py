"""Render diverse high-EPE Kubric validation trajectories for visual diagnosis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from ..data.datasets.movif256 import MOViF256Dataset
from ..data.factory import load_dataset
from ..data.geometry import GeometryBuilder
from ..data.text_conditions import load_inference_text_condition
from ..models.factory import build_real_model, precision_dtype
from ..utils.io import atomic_json


def select_worst_unique_clips(evaluation_root: Path, count: int) -> list[dict[str, Any]]:
    candidates = []
    for path in sorted((evaluation_root / "kubric" / "clips").glob("clip_*.json")):
        record = json.loads(path.read_text())
        for source, values in enumerate(record["raw"]["source_target_mean_epe_m"]):
            valid = [float(value) for value in values if value is not None]
            if valid:
                candidates.append({
                    "index": int(record["index"]),
                    "source": source,
                    "clip_id": str(record["clip_id"]),
                    "raw_epe_m": float(np.mean(valid)),
                })
    candidates.sort(key=lambda value: value["raw_epe_m"], reverse=True)
    selected, seen = [], set()
    for global_rank, value in enumerate(candidates, start=1):
        if value["index"] in seen:
            continue
        seen.add(value["index"])
        selected.append(dict(value, global_source_rank=global_rank, diverse_rank=len(selected) + 1))
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"only found {len(selected)}/{count} distinct evaluated Kubric clips")
    return selected


def predict_clip(model, dataset: MOViF256Dataset, condition: torch.Tensor,
                 index: int, source: int, device: torch.device, dtype: torch.dtype,
                 target_chunk: int, mean: torch.Tensor, scale: torch.Tensor) -> np.ndarray:
    latent = torch.from_numpy(dataset.clean_latent(index))[None].to(device, dtype=dtype)
    source_rgb = torch.from_numpy(dataset.source_rgb(index, source))[None].permute(0, 3, 1, 2)
    source_rgb = source_rgb.to(device, dtype=dtype) / 127.5 - 1.0
    outputs = []
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=dtype, enabled=device.type == "cuda",
    ):
        z4d = model.backbone(latent, condition)
        pyramid = model.decoder.encode_source_rgb(source_rgb)
        for start in range(0, 21, target_chunk):
            targets = list(range(start, min(start + target_chunk, 21)))
            outputs.append(model.decoder(
                z4d,
                torch.full((1, len(targets)), source, device=device, dtype=torch.long),
                torch.tensor(targets, device=device, dtype=torch.long)[None],
                source_pyramid=pyramid,
            ).normalized_xyz.float().cpu())
    normalized = torch.cat(outputs, dim=1)[0]
    return (normalized * scale + mean).numpy()


def select_error_tracks(error: np.ndarray, valid: np.ndarray, count: int,
                        minimum_frames: int, minimum_distance: float) -> list[tuple[int, int]]:
    valid_count = valid.sum(axis=0)
    score = (error * valid).sum(axis=0) / np.maximum(valid_count, 1)
    score[valid_count < minimum_frames] = -np.inf
    chosen: list[tuple[int, int]] = []
    height, width = score.shape
    for linear in np.argsort(score.reshape(-1))[::-1]:
        if not np.isfinite(score.reshape(-1)[linear]):
            break
        y, x = divmod(int(linear), width)
        if all((x - px) ** 2 + (y - py) ** 2 >= minimum_distance ** 2
               for py, px in chosen):
            chosen.append((y, x))
            if len(chosen) == count:
                break
    if len(chosen) < min(4, count):
        raise RuntimeError(f"only found {len(chosen)} spatially separated error tracks")
    return chosen


def project_kubric(points: np.ndarray, source: int, builder: GeometryBuilder
                   ) -> tuple[np.ndarray, np.ndarray]:
    frames, tracks, coordinates = points.shape
    if coordinates != 3:
        raise ValueError("trajectories must be [T,N,3]")
    uv = np.full((frames, tracks, 2), np.nan, dtype=np.float64)
    finite = np.zeros((frames, tracks), dtype=bool)
    for target in range(frames):
        world = builder.camera_frame_to_world(points[target], source)
        projected, view_depth, _ = builder.camera.project(
            world, builder.sample.camera_positions[target], builder.sample.camera_quaternions[target],
        )
        good = np.isfinite(projected).all(axis=1) & np.isfinite(view_depth) & (view_depth > 1e-6)
        uv[target, good] = projected[good]
        finite[target] = good
    return uv, finite


def inside(point: np.ndarray) -> bool:
    return bool(np.isfinite(point).all() and 0 <= point[0] < 256 and 0 <= point[1] < 256)


def draw_overlay(frame: np.ndarray, target: int, source: int,
                 gt_uv: np.ndarray, pred_uv: np.ndarray, valid: np.ndarray,
                 visible: np.ndarray, clip_epe: float, index: int, trail: int) -> np.ndarray:
    image = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), (512, 512),
                       interpolation=cv2.INTER_NEAREST)
    for track in range(gt_uv.shape[1]):
        begin = max(0, target - trail)
        for values, color in ((gt_uv, (40, 220, 40)), (pred_uv, (220, 40, 220))):
            points = [np.rint(values[t, track] * 2).astype(np.int32)
                      for t in range(begin, target + 1)
                      if valid[t, track] and inside(values[t, track])]
            if len(points) >= 2:
                cv2.polylines(image, [np.stack(points)], False, color, 2, cv2.LINE_AA)
        if not valid[target, track]:
            continue
        gt, pred = gt_uv[target, track], pred_uv[target, track]
        if inside(gt):
            centre = tuple(np.rint(gt * 2).astype(int))
            cv2.circle(image, centre, 6, (40, 220, 40),
                       -1 if visible[target, track] else 2, cv2.LINE_AA)
            cv2.putText(image, str(track), (centre[0] + 5, centre[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        if inside(pred):
            centre = tuple(np.rint(pred * 2).astype(int))
            cv2.drawMarker(image, centre, (220, 40, 220), cv2.MARKER_TILTED_CROSS,
                           13, 2, cv2.LINE_AA)
        if inside(gt) and inside(pred):
            cv2.line(image, tuple(np.rint(gt * 2).astype(int)),
                     tuple(np.rint(pred * 2).astype(int)), (20, 210, 240), 1, cv2.LINE_AA)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (512, 66), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, image, 0.38, 0.0, image)
    cv2.putText(image, f"kubric validation clip={index} source={source} frame={target}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, f"raw EPE={clip_epe:.3f}m  GT=green  Prediction=magenta",
                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "hollow GT = occluded but geometrically valid", (10, 61),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (210, 210, 210), 1, cv2.LINE_AA)
    return image


def robust_limits(gt: np.ndarray, pred: np.ndarray, valid: np.ndarray
                 ) -> list[tuple[float, float]]:
    merged = np.concatenate([coordinates[valid] for coordinates in (gt, pred)], axis=0)
    limits = []
    for axis in range(3):
        low, high = np.quantile(merged[:, axis], [0.01, 0.99])
        if not np.isfinite(low + high) or high - low < 1e-3:
            low, high = float(low) - 0.5, float(high) + 0.5
        margin = 0.08 * (high - low)
        limits.append((float(low - margin), float(high + margin)))
    return limits


def render_3d_panel(gt: np.ndarray, pred: np.ndarray, valid: np.ndarray,
                    target: int, limits: list[tuple[float, float]]) -> np.ndarray:
    figure = plt.figure(figsize=(5.12, 5.12), dpi=100)
    axis = figure.add_subplot(111, projection="3d")
    for track in range(gt.shape[1]):
        times = np.flatnonzero(valid[:target + 1, track])
        if not len(times):
            continue
        axis.plot(gt[times, track, 0], gt[times, track, 2], gt[times, track, 1],
                  color="#36d936", linewidth=1.2, alpha=0.8)
        axis.plot(pred[times, track, 0], pred[times, track, 2], pred[times, track, 1],
                  color="#dc28dc", linewidth=1.2, alpha=0.8)
        if valid[target, track]:
            axis.scatter(*gt[target, track, [0, 2, 1]], color="#36d936", s=16)
            axis.scatter(*pred[target, track, [0, 2, 1]], color="#dc28dc", s=16, marker="x")
    axis.set_xlim(*limits[0]); axis.set_ylim(*limits[2]); axis.set_zlim(*limits[1])
    axis.set_xlabel("X (m)"); axis.set_ylabel("Z (m)"); axis.set_zlabel("Y (m)")
    axis.set_title(f"3D trajectories in source-camera frame, t={target}")
    axis.view_init(elev=22, azim=-62)
    figure.tight_layout(); figure.canvas.draw()
    panel = cv2.cvtColor(np.asarray(figure.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
    plt.close(figure)
    return panel


def render_mp4v(path: Path, video: np.ndarray, builder: GeometryBuilder, source: int,
                gt: np.ndarray, pred: np.ndarray, valid: np.ndarray,
                visible: np.ndarray, clip_epe: float, index: int, fps: float,
                trail: int) -> None:
    gt_uv, gt_projected = project_kubric(gt, source, builder)
    pred_uv, _ = project_kubric(pred, source, builder)
    drawable = valid & gt_projected
    limits = robust_limits(gt, pred, valid)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1024, 512))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open {path}")
    try:
        for target, frame in enumerate(video):
            overlay = draw_overlay(frame, target, source, gt_uv, pred_uv, drawable,
                                   visible, clip_epe, index, trail)
            writer.write(np.concatenate((overlay, render_3d_panel(
                gt, pred, valid, target, limits,
            )), axis=1))
    finally:
        writer.release()


def encode_h264(ffmpeg: Path, source: Path, destination: Path, fps: float) -> None:
    temporary = destination.with_suffix(".tmp.mp4")
    subprocess.run([
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
    ], check=True)
    temporary.replace(destination)
    capture = cv2.VideoCapture(str(destination))
    frames, actual_fps = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), capture.get(cv2.CAP_PROP_FPS)
    first_ok, _ = capture.read(); capture.set(cv2.CAP_PROP_POS_FRAMES, max(frames - 1, 0))
    last_ok, _ = capture.read(); capture.release()
    if frames != 21 or abs(actual_fps - fps) > 1e-6 or not first_ok or not last_ok:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"invalid encoded video {destination}: frames={frames}, fps={actual_fps}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--tracks", type=int, default=16)
    parser.add_argument("--minimum-valid-frames", type=int, default=5)
    parser.add_argument("--minimum-track-distance", type=float, default=14.0)
    parser.add_argument("--target-chunk", type=int, default=7)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--trail", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.count < 1 or args.tracks < 1 or not 1 <= args.target_chunk <= 21:
        raise ValueError("count/tracks must be positive and target-chunk must be in [1,21]")

    config_path, checkpoint_path = Path(args.config).resolve(), Path(args.checkpoint).resolve()
    output_dir, ffmpeg = Path(args.output_dir).resolve(), Path(args.ffmpeg).resolve()
    if not ffmpeg.is_file():
        raise FileNotFoundError(ffmpeg)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(config_path.read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    step = int(checkpoint.get("training_state", {}).get("global_step", -1))
    if checkpoint.get("config", {}).get("prompts") != config.get("prompts"):
        raise ValueError("checkpoint/config prompt mismatch")
    plans = select_worst_unique_clips(Path(args.evaluation_root).resolve(), args.count)

    dataset = load_dataset(config, "kubric", split="validation")
    device, dtype = torch.device(args.device), precision_dtype(config["precision"])
    condition, prompt_metadata = load_inference_text_condition(config, "kubric")
    condition = condition.to(device, dtype=dtype)
    model = build_real_model(config, device)
    model.load_state_dict(checkpoint["model"], strict=True); model.eval()
    mean = torch.as_tensor(checkpoint["coordinate_mean"], dtype=torch.float32).reshape(1, 3, 1, 1)
    scale = torch.as_tensor(checkpoint["coordinate_scale"], dtype=torch.float32).reshape(1, 3, 1, 1)

    rendered = []
    for plan in plans:
        index, source = plan["index"], plan["source"]
        gt, valid, visible = dataset.source_all_targets_with_visibility(index, source)
        pred = predict_clip(model, dataset, condition, index, source, device, dtype,
                            args.target_chunk, mean, scale)
        error = np.linalg.norm(pred - gt, axis=1)
        measured = float(error[valid].mean())
        if not np.isclose(measured, plan["raw_epe_m"], atol=2e-5, rtol=0):
            raise RuntimeError(f"EPE audit changed for clip {index}: {measured} != {plan['raw_epe_m']}")
        tracks = select_error_tracks(error, valid, args.tracks, args.minimum_valid_frames,
                                     args.minimum_track_distance)
        ys, xs = (np.asarray([p[i] for p in tracks], dtype=np.int64) for i in (0, 1))
        gt_tracks = gt[:, :, ys, xs].transpose(0, 2, 1)
        pred_tracks = pred[:, :, ys, xs].transpose(0, 2, 1)
        valid_tracks, visible_tracks = valid[:, ys, xs], visible[:, ys, xs]
        builder = GeometryBuilder(dataset.sample(index))
        source_uv, source_ok = project_kubric(gt_tracks, source, builder)
        reprojection = np.linalg.norm(source_uv[source] - np.stack((xs, ys), axis=-1), axis=-1)
        reprojection = reprojection[source_ok[source] & valid_tracks[source]]
        if not len(reprojection) or float(np.median(reprojection)) > 1.5:
            raise RuntimeError(f"source reprojection audit failed for clip {index}")
        stem = (f"kubric_step{step:06d}_validation_rank{plan['diverse_rank']:02d}_"
                f"clip{index}_source{source}_gt_vs_prediction")
        intermediate, destination = output_dir / f".{stem}_mp4v.mp4", output_dir / f"{stem}_vscode_h264.mp4"
        render_mp4v(intermediate, dataset.rgb(index), builder, source, gt_tracks,
                    pred_tracks, valid_tracks, visible_tracks, measured, index,
                    args.fps, args.trail)
        try:
            encode_h264(ffmpeg, intermediate, destination, args.fps)
        finally:
            intermediate.unlink(missing_ok=True)
        record = {**plan, "raw_epe_m": measured,
                  "selected_tracks_yx": [[int(y), int(x)] for y, x in tracks],
                  "source_reprojection_median_px": float(np.median(reprojection)),
                  "video": str(destination), "bytes": destination.stat().st_size}
        rendered.append(record)
        print(json.dumps({"event": "kubric_worst_render", **record}), flush=True)

    report = {
        "checkpoint": str(checkpoint_path), "checkpoint_step": step,
        "config": str(config_path), "evaluation_root": str(Path(args.evaluation_root).resolve()),
        "selection": "highest raw all-target source EPE with unique validation clips",
        "prompt_metadata": prompt_metadata, "renders": rendered,
    }
    summary = output_dir / f"kubric_step{step:06d}_validation_worst_additional_summary.json"
    atomic_json(summary, report)
    print(json.dumps({"summary": str(summary), "videos": [r["video"] for r in rendered]}), flush=True)
    print("KUBRIC_WORST_TRAJECTORY_RENDER_OK", flush=True)


if __name__ == "__main__":
    main()
