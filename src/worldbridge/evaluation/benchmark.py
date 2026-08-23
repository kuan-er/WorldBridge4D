"""Exhaustive validation: every clip and every source-target pair."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import yaml

from ..data.constants import DATASET_NAMES
from ..data.factory import load_dataset
from ..data.text_conditions import load_inference_text_condition
from ..models.factory import build_real_model, precision_dtype
from ..utils.io import atomic_json
from .inference import PERSISTENT_RUN_ROOT
from .metrics import align_sim3_to_ground_truth

DEFAULT_OUTPUT_ROOT = PERSISTENT_RUN_ROOT / "evaluation-step100000"


def decode_all_targets(model, z4d: torch.Tensor, source: int,
                       source_pyramid: dict[int, torch.Tensor], target_chunk: int,
                       device: torch.device) -> torch.Tensor:
    outputs = []
    targets = list(range(21))
    for start in range(0, 21, target_chunk):
        chunk = targets[start:start + target_chunk]
        outputs.append(model.decoder(
            z4d,
            torch.full((1, len(chunk)), source, device=device, dtype=torch.long),
            torch.tensor(chunk, device=device, dtype=torch.long)[None],
            source_pyramid=source_pyramid,
        ).normalized_xyz)
    return torch.cat(outputs, dim=1)[0].float().cpu()


def rgb_clip_tensor(dataset, index: int, device: torch.device,
                    dtype: torch.dtype) -> torch.Tensor:
    value = dataset.rgb(index)
    if value.shape != (21, 256, 256, 3) or value.dtype != np.uint8:
        raise RuntimeError(f"RGB clip must be uint8 [21,256,256,3], got {value.shape}/{value.dtype}")
    tensor = torch.from_numpy(value).permute(0, 3, 1, 2).to(device, dtype=dtype)
    return tensor / 127.5 - 1.0


def pair_epe(prediction: torch.Tensor, target: torch.Tensor,
             valid: torch.Tensor) -> tuple[list[float | None], list[int]]:
    error = torch.linalg.vector_norm(prediction - target, dim=1)
    means: list[float | None] = []
    counts: list[int] = []
    for target_index in range(21):
        mask = valid[target_index] & torch.isfinite(error[target_index])
        count = int(mask.sum())
        counts.append(count)
        means.append(float(error[target_index][mask].mean()) if count else None)
    return means, counts


def align_if_possible(prediction: torch.Tensor, target: torch.Tensor,
                      valid: torch.Tensor, enabled: bool) -> tuple[torch.Tensor, dict[str, Any]]:
    if not enabled:
        return prediction, {"enabled": False, "method": "none"}
    count = int(valid.sum())
    if count < 3:
        return prediction, {
            "enabled": False,
            "method": "proper_umeyama_prediction_to_ground_truth",
            "reason": "insufficient_valid_points",
            "points": count,
        }
    return align_sim3_to_ground_truth(prediction, target, valid)


def protocol_id(config: dict[str, Any], dataset: str, checkpoint: Path,
                target_chunk: int, sim3: bool) -> str:
    payload = {
        "version": 1,
        "dataset": dataset,
        "split": "validation",
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(config["selected_checkpoint_step"]),
        "checkpoint_sha256": config["selected_checkpoint_sha256"],
        "targets": list(range(21)),
        "sources": list(range(21)),
        "target_chunk": int(target_chunk),
        "sim3": bool(sim3),
        "sim3_scope": "one_joint_transform_per_clip_source_over_all_targets",
        "prompts": config["prompts"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validated_output_root(path: str | Path, dataset: str) -> Path:
    path = (Path(path) / dataset).expanduser().resolve(strict=False)
    try:
        path.relative_to(PERSISTENT_RUN_ROOT.resolve(strict=False))
    except ValueError as exc:
        raise ValueError(f"evaluation output must be under {PERSISTENT_RUN_ROOT}, got {path}") from exc
    return path


def load_completed_clip(path: Path, expected_protocol: str, index: int) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
        if value["protocol_id"] != expected_protocol or int(value["index"]) != int(index):
            return None
        for mode in ("raw", "sim3"):
            matrix = value[mode]["source_target_mean_epe_m"]
            if len(matrix) != 21 or any(len(row) != 21 for row in matrix):
                return None
        return value
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def aggregate(records: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    pair_sum = np.zeros((21, 21), dtype=np.float64)
    pair_clips = np.zeros((21, 21), dtype=np.int64)
    point_error_sum = 0.0
    point_count = 0
    clip_means = []
    for record in records:
        matrix = record[mode]["source_target_mean_epe_m"]
        counts = record[mode]["source_target_valid_points"]
        values = []
        for source in range(21):
            for target in range(21):
                value = matrix[source][target]
                count = int(counts[source][target])
                if value is None or count == 0:
                    continue
                value = float(value)
                pair_sum[source, target] += value
                pair_clips[source, target] += 1
                point_error_sum += value * count
                point_count += count
                values.append(value)
        if values:
            clip_means.append(float(np.mean(values)))
    pair_matrix = [
        [float(pair_sum[s, t] / pair_clips[s, t]) if pair_clips[s, t] else None
         for t in range(21)]
        for s in range(21)
    ]
    return {
        "clips": len(records),
        "valid_source_target_pairs": int(pair_clips.sum()),
        "macro_clip_mean_epe_m": float(np.mean(clip_means)),
        "macro_source_target_mean_epe_m": float(
            pair_sum.sum() / pair_clips.sum()
        ),
        "point_weighted_mean_epe_m": float(point_error_sum / point_count),
        "valid_points": int(point_count),
        "source_target_mean_epe_m": pair_matrix,
        "source_target_clip_counts": pair_clips.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True, choices=DATASET_NAMES)
    parser.add_argument("--target-chunk", type=int, default=21)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--no-sim3", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.target_chunk <= 21:
        raise ValueError("target-chunk must be in [1,21]")
    if args.max_clips is not None and args.max_clips < 1:
        raise ValueError("max-clips must be positive")

    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    config = yaml.safe_load(config_path.read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    if int(checkpoint.get("training_state", {}).get("global_step", -1)) != int(
        config["selected_checkpoint_step"]
    ):
        raise ValueError("evaluation requires the selected production checkpoint step")
    if checkpoint.get("config", {}).get("prompts") != config.get("prompts"):
        raise ValueError("checkpoint/config prompt mismatch")

    dataset = load_dataset(config, args.dataset, split="validation")
    clip_count = len(dataset) if args.max_clips is None else min(len(dataset), args.max_clips)
    output_root = validated_output_root(args.output_root, args.dataset)
    clips_root = output_root / "clips"
    clips_root.mkdir(parents=True, exist_ok=True)
    protocol = protocol_id(config, args.dataset, checkpoint_path, args.target_chunk, not args.no_sim3)

    device = torch.device(args.device)
    dtype = precision_dtype(config["precision"])
    condition, prompt_metadata = load_inference_text_condition(config, args.dataset)
    condition = condition.to(device, dtype=dtype)
    model = build_real_model(config, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    mean = torch.as_tensor(checkpoint["coordinate_mean"], dtype=torch.float32).reshape(1, 3, 1, 1)
    scale = torch.as_tensor(checkpoint["coordinate_scale"], dtype=torch.float32).reshape(1, 3, 1, 1)
    started = time.perf_counter()
    records: list[dict[str, Any]] = []

    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=dtype, enabled=device.type == "cuda",
    ):
        for index in range(clip_count):
            record_path = clips_root / f"clip_{index:06d}.json"
            record = load_completed_clip(record_path, protocol, index)
            if record is not None:
                records.append(record)
                continue
            latent = torch.from_numpy(dataset.clean_latent(index))[None].to(device, dtype=dtype)
            z4d = model.backbone(latent, condition)
            rgb = rgb_clip_tensor(dataset, index, device, dtype)
            raw_matrix: list[list[float | None]] = []
            aligned_matrix: list[list[float | None]] = []
            count_matrix: list[list[int]] = []
            sim3_records = []
            for source in range(21):
                xyz_np, valid_np = dataset.source_all_targets(index, source)
                target = torch.from_numpy(xyz_np).float()
                valid = torch.from_numpy(valid_np).bool()
                source_pyramid = model.decoder.encode_source_rgb(rgb[source:source + 1])
                normalized = decode_all_targets(
                    model, z4d, source, source_pyramid, args.target_chunk, device,
                )
                raw = normalized * scale + mean
                raw_epe, counts = pair_epe(raw, target, valid)
                aligned, sim3 = align_if_possible(
                    raw, target, valid, enabled=not args.no_sim3,
                )
                aligned_epe, aligned_counts = pair_epe(aligned, target, valid)
                if aligned_counts != counts:
                    raise RuntimeError("raw/aligned valid counts differ")
                raw_matrix.append(raw_epe)
                aligned_matrix.append(aligned_epe)
                count_matrix.append(counts)
                sim3_records.append(sim3)
                del target, valid, normalized, raw, aligned, source_pyramid
            record = {
                "protocol_id": protocol,
                "dataset": args.dataset,
                "split": "validation",
                "index": index,
                "clip_id": str(dataset.rows[index]["clip_id"]),
                "raw": {
                    "source_target_mean_epe_m": raw_matrix,
                    "source_target_valid_points": count_matrix,
                },
                "sim3": {
                    "source_target_mean_epe_m": aligned_matrix,
                    "source_target_valid_points": count_matrix,
                    "source_transforms": sim3_records,
                },
            }
            atomic_json(record_path, record)
            records.append(record)
            print(json.dumps({
                "event": "exhaustive_evaluation_progress", "dataset": args.dataset,
                "clips": index + 1, "total": clip_count,
                "elapsed_seconds": time.perf_counter() - started,
            }), flush=True)
            del latent, z4d, rgb

    summary = {
        "status": "complete",
        "protocol_id": protocol,
        "dataset": args.dataset,
        "split": "validation",
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(config["selected_checkpoint_step"]),
        "checkpoint_sha256": config["selected_checkpoint_sha256"],
        "config": str(config_path),
        "clips": clip_count,
        "sources_per_clip": 21,
        "targets_per_source": 21,
        "target_chunk": args.target_chunk,
        "sim3_enabled": not args.no_sim3,
        "sim3_scope": "one_joint_transform_per_clip_source_over_all_targets",
        "prompt_metadata": prompt_metadata,
        "raw": aggregate(records, "raw"),
        "sim3": aggregate(records, "sim3"),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(output_root / "summary.json", summary)
    print(json.dumps({
        "dataset": args.dataset, "clips": clip_count,
        "raw_macro_epe_m": summary["raw"]["macro_source_target_mean_epe_m"],
        "sim3_macro_epe_m": summary["sim3"]["macro_source_target_mean_epe_m"],
        "output": str(output_root / "summary.json"),
    }), flush=True)
    print("EXHAUSTIVE_VALIDATION_EVALUATION_OK", flush=True)


if __name__ == "__main__":
    main()
