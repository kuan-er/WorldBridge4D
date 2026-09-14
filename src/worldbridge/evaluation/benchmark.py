"""Fixed-budget validation: the canonical 122-query protocol per clip."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable

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
FRAME_COUNT = 21
POINTMAP_SOURCES = tuple(range(FRAME_COUNT))
FIRST_FRAME_SOURCE = 0
ARBITRARY_SOURCES = (5, 10, 15, 20)

# The protocol counts category membership, so (0, 0) is intentionally present
# in both pointmap and first-frame tracking.  It is inferred only once and is
# retained once in the canonical matrix below.
QUERY_GROUPS: dict[str, tuple[tuple[int, int], ...]] = {
    "pointmap": tuple((source, source) for source in POINTMAP_SOURCES),
    "first_frame_tracking": tuple(
        (FIRST_FRAME_SOURCE, target) for target in range(FRAME_COUNT)
    ),
    "arbitrary_tracking": tuple(
        (source, target)
        for source in ARBITRARY_SOURCES
        for target in range(FRAME_COUNT)
        if target != source
    ),
}
LOGICAL_QUERIES = tuple(
    query for group in QUERY_GROUPS.values() for query in group
)
EVALUATED_QUERIES = tuple(dict.fromkeys(LOGICAL_QUERIES))
LOGICAL_QUERY_COUNT = len(LOGICAL_QUERIES)  # 122, with one category overlap
UNIQUE_QUERY_COUNT = len(EVALUATED_QUERIES)  # 121 actual source-target pairs
TRACKING_SOURCES = (FIRST_FRAME_SOURCE,) + ARBITRARY_SOURCES


def decode_targets(model, z4d: torch.Tensor, source: int,
                   source_pyramid: dict[int, torch.Tensor] | None,
                   targets: list[int], target_chunk: int,
                   device: torch.device) -> torch.Tensor:
    outputs = []
    for start in range(0, len(targets), target_chunk):
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
    if value.shape != (FRAME_COUNT, 256, 256, 3) or value.dtype != np.uint8:
        raise RuntimeError(
            f"RGB clip must be uint8 [21,256,256,3], got {value.shape}/{value.dtype}"
        )
    tensor = torch.from_numpy(value).permute(0, 3, 1, 2).to(device, dtype=dtype)
    return tensor / 127.5 - 1.0


def pair_epe(prediction: torch.Tensor, target: torch.Tensor,
             valid: torch.Tensor) -> tuple[list[float | None], list[int]]:
    error = torch.linalg.vector_norm(prediction - target, dim=1)
    means: list[float | None] = []
    counts: list[int] = []
    for target_index in range(prediction.shape[0]):
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
        "version": 2,
        "dataset": dataset,
        "split": "validation",
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(config["selected_checkpoint_step"]),
        "checkpoint_sha256": config["selected_checkpoint_sha256"],
        "query_groups": {name: list(queries) for name, queries in QUERY_GROUPS.items()},
        "logical_query_count": LOGICAL_QUERY_COUNT,
        "unique_query_count": UNIQUE_QUERY_COUNT,
        "target_chunk": int(target_chunk),
        "sim3": bool(sim3),
        "sim3_tracking_scope": "one_joint_transform_per_clip_source_over_all_21_targets",
        "sim3_pointmap_scope": "one_joint_transform_per_clip_over_21_diagonal_pointmaps",
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
            if len(matrix) != FRAME_COUNT or any(len(row) != FRAME_COUNT for row in matrix):
                return None
        if len(value.get("query_metrics", [])) != LOGICAL_QUERY_COUNT:
            return None
        return value
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _iter_matrix_queries(
    records: list[dict[str, Any]], mode: str,
    queries: Iterable[tuple[int, int]] | None,
):
    query_list = (
        list(queries)
        if queries is not None
        else [(source, target) for source in range(FRAME_COUNT)
              for target in range(FRAME_COUNT)]
    )
    for record in records:
        matrix = record[mode]["source_target_mean_epe_m"]
        counts = record[mode]["source_target_valid_points"]
        yield record, matrix, counts, query_list


def aggregate(records: list[dict[str, Any]], mode: str,
              queries: Iterable[tuple[int, int]] | None = None) -> dict[str, Any]:
    """Aggregate matrix-backed metrics over the requested unique queries.

    ``queries`` is explicit for the fixed-budget evaluator.  Omitting it keeps
    this helper useful for legacy matrix-shaped diagnostics and unit tests.
    """
    pair_sum = np.zeros((FRAME_COUNT, FRAME_COUNT), dtype=np.float64)
    pair_clips = np.zeros((FRAME_COUNT, FRAME_COUNT), dtype=np.int64)
    point_error_sum = 0.0
    point_count = 0
    clip_means = []
    for _record, matrix, counts, query_list in _iter_matrix_queries(records, mode, queries):
        values = []
        for source, target in query_list:
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
    evaluated_pair_count = int(pair_clips.sum())
    if not clip_means or point_count == 0:
        raise RuntimeError(f"no valid {mode} evaluation queries")
    pair_matrix = [
        [float(pair_sum[s, t] / pair_clips[s, t]) if pair_clips[s, t] else None
         for t in range(FRAME_COUNT)]
        for s in range(FRAME_COUNT)
    ]
    return {
        "clips": len(records),
        "valid_source_target_pairs": evaluated_pair_count,
        "macro_clip_mean_epe_m": float(np.mean(clip_means)),
        "macro_source_target_mean_epe_m": float(
            pair_sum.sum() / max(evaluated_pair_count, 1)
        ),
        "point_weighted_mean_epe_m": float(point_error_sum / point_count),
        "valid_points": int(point_count),
        "source_target_mean_epe_m": pair_matrix,
        "source_target_clip_counts": pair_clips.tolist(),
    }


def aggregate_query_metrics(records: list[dict[str, Any]], mode: str,
                            group: str | None = None) -> dict[str, Any]:
    """Aggregate logical fixed-budget query entries, including category overlap."""
    clip_means = []
    query_sum = 0.0
    query_count = 0
    point_error_sum = 0.0
    point_count = 0
    valid_queries = 0
    for record in records:
        values = []
        for entry in record["query_metrics"]:
            if group is not None and entry["group"] != group:
                continue
            value = entry[f"{mode}_epe_m"]
            count = int(entry["valid_points"])
            if value is None or count == 0:
                continue
            value = float(value)
            values.append(value)
            query_sum += value
            query_count += 1
            point_error_sum += value * count
            point_count += count
            valid_queries += 1
        if values:
            clip_means.append(float(np.mean(values)))
    if not clip_means or point_count == 0:
        raise RuntimeError(f"no valid fixed-budget {mode} queries")
    return {
        "clips": len(records),
        "logical_queries_per_clip": (
            len(QUERY_GROUPS[group]) if group is not None else LOGICAL_QUERY_COUNT
        ),
        "valid_queries": valid_queries,
        "macro_clip_mean_epe_m": float(np.mean(clip_means)),
        "macro_query_mean_epe_m": float(query_sum / query_count),
        "point_weighted_mean_epe_m": float(point_error_sum / point_count),
        "valid_points": int(point_count),
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
    if not 1 <= args.target_chunk <= FRAME_COUNT:
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
            targets_by_source = {
                source: list(range(FRAME_COUNT)) if source in TRACKING_SOURCES else [source]
                for source in POINTMAP_SOURCES
            }
            raw_by_source: dict[int, torch.Tensor] = {}
            target_by_source: dict[int, torch.Tensor] = {}
            valid_by_source: dict[int, torch.Tensor] = {}
            raw_matrix: list[list[float | None]] = [[None] * FRAME_COUNT for _ in range(FRAME_COUNT)]
            count_matrix: list[list[int]] = [[0] * FRAME_COUNT for _ in range(FRAME_COUNT)]

            for source in POINTMAP_SOURCES:
                selected_targets = targets_by_source[source]
                xyz_np, valid_np = dataset.source_all_targets(index, source)
                target = torch.from_numpy(xyz_np[selected_targets]).float()
                valid = torch.from_numpy(valid_np[selected_targets]).bool()
                source_pyramid = model.decoder.encode_source_rgb(rgb[source:source + 1])
                normalized = decode_targets(
                    model, z4d, source, source_pyramid, selected_targets,
                    args.target_chunk, device,
                )
                raw = normalized * scale + mean
                raw_by_source[source] = raw
                target_by_source[source] = target
                valid_by_source[source] = valid
                values, counts = pair_epe(raw, target, valid)
                for local, target_index in enumerate(selected_targets):
                    raw_matrix[source][target_index] = values[local]
                    count_matrix[source][target_index] = counts[local]
                del target, valid, normalized, raw, source_pyramid

            # Pointmap alignment is one transform for all 21 diagonal maps.
            point_pred = torch.stack([
                raw_by_source[source][targets_by_source[source].index(source)]
                for source in POINTMAP_SOURCES
            ])
            point_target = torch.stack([
                target_by_source[source][targets_by_source[source].index(source)]
                for source in POINTMAP_SOURCES
            ])
            point_valid = torch.stack([
                valid_by_source[source][targets_by_source[source].index(source)]
                for source in POINTMAP_SOURCES
            ])
            point_aligned, point_sim3 = align_if_possible(
                point_pred, point_target, point_valid, enabled=not args.no_sim3,
            )

            # Tracking alignment is one transform per clip/source over all 21 targets.
            tracking_aligned: dict[int, torch.Tensor] = {}
            tracking_sim3: dict[str, Any] = {}
            for source in TRACKING_SOURCES:
                aligned, sim3 = align_if_possible(
                    raw_by_source[source], target_by_source[source],
                    valid_by_source[source], enabled=not args.no_sim3,
                )
                tracking_aligned[source] = aligned
                tracking_sim3[str(source)] = sim3

            sim3_matrix: list[list[float | None]] = [[None] * FRAME_COUNT for _ in range(FRAME_COUNT)]
            query_metrics: list[dict[str, Any]] = []
            group_aligned: dict[str, dict[tuple[int, int], tuple[float | None, int]]] = {}
            for group_name, queries in QUERY_GROUPS.items():
                group_aligned[group_name] = {}
                for source, target in queries:
                    local = targets_by_source[source].index(target)
                    raw_value = raw_matrix[source][target]
                    count = count_matrix[source][target]
                    target_value = target_by_source[source][local:local + 1]
                    valid_value = valid_by_source[source][local:local + 1]
                    if group_name == "pointmap":
                        aligned_prediction = point_aligned[source:source + 1]
                    else:
                        aligned_prediction = tracking_aligned[source][local:local + 1]
                    aligned_values, aligned_counts = pair_epe(
                        aligned_prediction, target_value, valid_value,
                    )
                    sim3_value = aligned_values[0]
                    if group_name == "pointmap" or (source, target) not in group_aligned["pointmap"]:
                        sim3_matrix[source][target] = sim3_value
                    group_aligned[group_name][(source, target)] = (sim3_value, aligned_counts[0])
                    query_metrics.append({
                        "group": group_name,
                        "source": source,
                        "target": target,
                        "raw_epe_m": raw_value,
                        "sim3_epe_m": sim3_value,
                        "valid_points": count,
                    })

            record = {
                "protocol_id": protocol,
                "dataset": args.dataset,
                "split": "validation",
                "index": index,
                "clip_id": str(dataset.rows[index]["clip_id"]),
                "query_protocol": {
                    "logical_queries": LOGICAL_QUERY_COUNT,
                    "unique_queries": UNIQUE_QUERY_COUNT,
                    "groups": {name: list(queries) for name, queries in QUERY_GROUPS.items()},
                },
                "query_metrics": query_metrics,
                "raw": {
                    "source_target_mean_epe_m": raw_matrix,
                    "source_target_valid_points": count_matrix,
                },
                "sim3": {
                    "source_target_mean_epe_m": sim3_matrix,
                    "source_target_valid_points": count_matrix,
                    "tracking_transforms": tracking_sim3,
                    "pointmap_transform": point_sim3,
                },
            }
            atomic_json(record_path, record)
            records.append(record)
            print(json.dumps({
                "event": "fixed_budget_evaluation_progress", "dataset": args.dataset,
                "clips": index + 1, "total": clip_count,
                "logical_queries_per_clip": LOGICAL_QUERY_COUNT,
                "unique_queries_per_clip": UNIQUE_QUERY_COUNT,
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
        "logical_queries_per_clip": LOGICAL_QUERY_COUNT,
        "unique_queries_per_clip": UNIQUE_QUERY_COUNT,
        "query_groups": {name: list(queries) for name, queries in QUERY_GROUPS.items()},
        "target_chunk": args.target_chunk,
        "sim3_enabled": not args.no_sim3,
        "sim3_tracking_scope": "one_joint_transform_per_clip_source_over_all_21_targets",
        "sim3_pointmap_scope": "one_joint_transform_per_clip_over_21_diagonal_pointmaps",
        "prompt_metadata": prompt_metadata,
        "raw": aggregate_query_metrics(records, "raw"),
        "sim3": aggregate_query_metrics(records, "sim3"),
        "groups": {
            group: {
                "raw": aggregate_query_metrics(records, "raw", group),
                "sim3": aggregate_query_metrics(records, "sim3", group),
            }
            for group in QUERY_GROUPS
        },
        # Matrix summaries retain the familiar source-target layout, but only
        # contain the 121 unique fixed-budget pairs; null means unqueried.
        "raw_matrix": aggregate(records, "raw", EVALUATED_QUERIES),
        "sim3_matrix": aggregate(records, "sim3", EVALUATED_QUERIES),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(output_root / "summary.json", summary)
    print(json.dumps({
        "dataset": args.dataset, "clips": clip_count,
        "raw_macro_epe_m": summary["raw"]["macro_query_mean_epe_m"],
        "sim3_macro_epe_m": summary["sim3"]["macro_query_mean_epe_m"],
        "logical_queries_per_clip": LOGICAL_QUERY_COUNT,
        "unique_queries_per_clip": UNIQUE_QUERY_COUNT,
        "output": str(output_root / "summary.json"),
    }), flush=True)
    print("FIXED_BUDGET_VALIDATION_EVALUATION_OK", flush=True)


if __name__ == "__main__":
    main()
