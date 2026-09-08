"""Read-only matched checkpoint diagnostics: prepare, run (or gate), aggregate.

All generated assets/results are outside Git. No optimizer is constructed and
no model weights are saved. Run with ``python -m worldbridge.evaluation.diagnostics``.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
import yaml

from ..data.constants import DATASET_NAMES
from ..data.factory import load_dataset
from ..models.factory import build_real_model
from ..data.text_conditions import load_inference_text_condition
from ..trainer.cycle import camera_batch, pixel_cycle_loss
from ..utils.io import atomic_json
from .diagnostic_metrics import (
    cross_instance_neighbor_proxy, depth_discontinuity, edge_distance,
    parent_balanced_indices, segmentation_discontinuity, stratified_epe,
)

ROOT = Path(__file__).resolve().parents[3]


def emit(event, **values):
    print(json.dumps({"event": event, **values}, allow_nan=False), flush=True)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_identity(path):
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    return digest(path)


def environment():
    return {"python": sys.version, "executable": sys.executable, "torch": torch.__version__,
            "cuda_build": torch.version.cuda, "numpy": np.__version__,
            "platform": platform.platform(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "command": sys.argv, "git_snapshot": str(ROOT)}


def protocol_config(path):
    path = Path(path).resolve()
    spec = yaml.safe_load(path.read_text())
    config = yaml.safe_load((ROOT / spec["data_config"]).read_text())
    return spec, config


def output_root(path):
    path = Path(path).resolve()
    if not path.is_relative_to("/data/WorldBridge4D-runs"):
        raise ValueError("diagnostics require persistent output under /data/WorldBridge4D-runs")
    path.mkdir(parents=True, exist_ok=True)
    return path


def planned_clips(spec, config):
    historical = json.loads(Path(spec["historical_selection"]).read_text())
    plans = []
    for name in DATASET_NAMES:
        for item in historical["datasets"][name]["candidates"]:
            plans.append({"dataset": name, "split": "train", "cohort": "historical_train_replay",
                          "index": int(item["index"]), "clip_id": item["clip_id"],
                          "sources": [int(item["source"])], "historical_95k_epe_m": item["raw_epe_m"]})
        index_path = Path(config["datasets"][name]["cache_root"]) / "splits" / "validation.jsonl"
        rows = [json.loads(line) for line in index_path.read_text().splitlines() if line]
        selected = parent_balanced_indices(rows, int(spec["validation_clips_per_dataset"]),
                                           int(spec["seed"]) + DATASET_NAMES.index(name))
        for index, parent in selected:
            plans.append({"dataset": name, "split": "validation", "cohort": "validation_screen",
                          "index": index, "clip_id": rows[index]["clip_id"], "parent_id": parent,
                          "sources": list(spec["validation_sources"])})
    return plans


def geometry_context(dataset, index):
    if hasattr(dataset, "sample"):
        sample = dataset.sample(index)
        depth, depth_valid, segmentation = sample.depth, sample.depth_valid, sample.segmentation
    else:
        gi = dataset.geometry_indices[index]
        _, _, depth, depth_valid = dataset.geometry.camera(gi)
        segmentation = None
    return depth, depth_valid, segmentation, dataset.cycle_camera(index)


def prepare_clip(item, dataset, spec, root, protocol):
    index = item["index"]
    if dataset.rows[index]["clip_id"] != item["clip_id"]:
        raise ValueError("index/clip identity drift")
    base = root / "prepared" / item["dataset"] / item["split"] / f"clip-{index:06d}"
    marker = base / "complete.json"
    if marker.is_file():
        meta = json.loads(marker.read_text())
        if meta["protocol"] != protocol or meta["sources"] != item["sources"]:
            raise ValueError("prepared input protocol changed")
        for key, info in meta["files"].items():
            if digest(info["path"]) != info["sha256"]:
                raise ValueError(f"prepared input checksum mismatch: {key}")
        return {**item, "parent_id": str(item.get("parent_id", dataset.rows[index].get("parent_id", item["clip_id"]))),
                "prepared": str(marker), "prepared_sha256": digest(marker)}
    # Never replace an unavailable validation clip with a convenient cached one.
    latent = dataset.clean_latent(index)
    rgb = dataset.rgb(index)
    if latent.shape != (16, 6, 32, 32) or rgb.shape != (21, 256, 256, 3):
        raise ValueError("native 256 input shape mismatch")
    depth, depth_valid, segmentation, cameras = geometry_context(dataset, index)
    clip_path = base / "clip.npz"
    checksum = save_npz(clip_path, latent=latent, rgb=rgb,
                        **{f"camera_{key}": np.asarray(value) for key, value in cameras.items()})
    files = {"clip": {"path": str(clip_path), "sha256": checksum}}
    for source in item["sources"]:
        xyz, valid, visible = dataset.source_all_targets_with_visibility(index, source)
        if xyz.shape != (21, 3, 256, 256) or valid.shape != (21, 256, 256):
            raise ValueError("geometry shape mismatch")
        if not np.isfinite(xyz.transpose(0, 2, 3, 1)[valid]).all():
            raise ValueError("non-finite valid geometry")
        reverse_source = int(np.argmax(np.abs(np.arange(21) - source)))
        reverse_xyz, reverse_valid, reverse_visible = dataset.source_all_targets_with_visibility(index, reverse_source)
        distance = edge_distance(depth_discontinuity(depth[source], depth_valid[source], spec["depth_relative_jump"]))
        arrays = {"xyz": xyz, "valid": valid, "visible": visible, "depth_distance": distance,
                  "reverse_source": np.asarray(reverse_source), "reverse_gt": reverse_xyz[source],
                  "reverse_gt_valid": reverse_valid[source], "reverse_gt_visible": reverse_visible[source]}
        if segmentation is not None:
            arrays["segmentation"] = segmentation[source]
            arrays["segmentation_distance"] = edge_distance(segmentation_discontinuity(segmentation[source]))
        path = base / f"source-{source:02d}.npz"
        files[str(source)] = {"path": str(path), "sha256": save_npz(path, **arrays)}
    atomic_json(marker, {"protocol": protocol, "sources": item["sources"], "files": files})
    return {**item, "parent_id": str(item.get("parent_id", dataset.rows[index].get("parent_id", item["clip_id"]))),
            "prepared": str(marker), "prepared_sha256": digest(marker)}


def prepare(args):
    root = output_root(args.output_root)
    spec, config = protocol_config(args.protocol)
    protocol = json_digest({"spec": spec, "data_config": config})
    plan = [item for item in planned_clips(spec, config)
            if (not args.datasets or item["dataset"] in args.datasets)
            and (not args.cohorts or item["cohort"] in args.cohorts)]
    if args.limit_clips:
        plan = plan[:args.limit_clips]
    atomic_json(root / (args.manifest_name + ".plan.json"), {"protocol": protocol, "planned": plan})
    models = {}
    for label, value in spec["checkpoints"].items():
        checksum = digest(value["path"])
        if checksum != value["sha256"]:
            raise ValueError(f"checkpoint checksum mismatch: {label}")
        models[label] = {**value, "identity": file_identity(value["path"])}
        emit("diagnostic_checkpoint_verified", label=label, sha256=checksum)
    ready, blocked, datasets = [], [], {}
    for item in plan:
        started = time.monotonic()
        try:
            key = (item["dataset"], item["split"])
            if key not in datasets:
                datasets[key] = load_dataset(config, key[0], split=key[1], allow_missing_latents=True)
            prepared = prepare_clip(item, datasets[key], spec, root, protocol)
            ready.append(prepared)
            emit("diagnostic_input_ready", dataset=item["dataset"], cohort=item["cohort"],
                 index=item["index"], sources=item["sources"], seconds=time.monotonic() - started)
        except Exception as exc:
            blocked.append({**item, "reason": f"{type(exc).__name__}: {exc}"})
            emit("diagnostic_input_blocked", dataset=item["dataset"], cohort=item["cohort"],
                 index=item["index"], reason=str(exc))
    if not ready:
        raise RuntimeError("no diagnostic inputs ready")
    manifest = {"protocol": protocol, "spec": spec, "data_config": config, "checkpoints": models,
                "planned": plan, "ready": ready, "blocked": blocked, "environment": environment(),
                "historical_selection_sha256": digest(spec["historical_selection"]),
                "limited_gate_manifest": bool(args.limit_clips)}
    atomic_json(root / args.manifest_name, manifest)
    emit("DIAGNOSTIC_PREPARE_OK", ready_clips=len(ready), blocked_clips=len(blocked),
         manifest=str(root / args.manifest_name))


def load_arrays(info):
    if digest(info["path"]) != info["sha256"]:
        raise ValueError(f"input checksum changed: {info['path']}")
    with np.load(info["path"]) as data:
        return {key: data[key] for key in data.files}


def decode(model, z4d, rgb, source, targets, device):
    image = torch.from_numpy(rgb[source])[None].permute(0, 3, 1, 2).to(device, dtype=torch.bfloat16)
    pyramid = model.decoder.encode_source_rgb(image / 127.5 - 1.0)
    predictions = []
    # Target chunk is intentionally fixed to 1 for shared-card diagnostics.
    for target in targets:
        output = model.decoder(z4d, torch.tensor([[source]], device=device),
                               torch.tensor([[target]], device=device), source_pyramid=pyramid)
        predictions.append(output.normalized_xyz[0, 0].float().cpu())
        del output
    del pyramid, image
    return torch.stack(predictions).numpy()


def cycle_diagnostic(pred, reverse, data, camera, source):
    target = int(data["reverse_source"])
    camera_tensors = camera_batch([camera], torch.device("cpu"), torch.float32)
    source_valid = data["valid"][source] & data["visible"][source]
    args = (torch.tensor([source]), torch.tensor([target]),
            torch.from_numpy(source_valid[None]), torch.from_numpy(data["valid"][target:target + 1]),
            torch.from_numpy(data["visible"][target:target + 1]), *camera_tensors)
    loss, count, error = pixel_cycle_loss(torch.from_numpy(pred[target:target + 1]),
                                         torch.from_numpy(reverse[None]), *args, pixel_stride=4)
    ref_loss, ref_count, ref_error = pixel_cycle_loss(torch.from_numpy(data["xyz"][target:target + 1]),
                                                    torch.from_numpy(data["reverse_gt"][None]), *args,
                                                    pixel_stride=4)
    eligible = source_valid & data["valid"][target] & data["visible"][target]
    return {"target": target, "pixel_error": float(error), "loss": float(loss), "valid_points": int(count),
            "gt_eligible_stride4": int(eligible[::4, ::4].sum()),
            "gt_reference_pixel_error": float(ref_error), "gt_reference_loss": float(ref_loss),
            "gt_reference_valid_points": int(ref_count),
            "caveat": "training-compatible prediction-dependent in-bounds mask; report coverage; sparse raster GT cycle need not be zero"}


def diagnostic_device():
    if torch.cuda.device_count() != 1:
        raise RuntimeError("each diagnostic worker requires exactly one explicitly leased physical GPU")
    device = torch.device("cuda:0")
    # device_count can use NVML without initializing CUDA. Peak-stat reset cannot.
    torch.cuda.set_device(device)
    return device


def run(args):
    root = output_root(args.output_root)
    manifest = json.loads(Path(args.manifest).read_text())
    spec = manifest["spec"]
    if manifest["protocol"] != json_digest({"spec": spec, "data_config": manifest["data_config"]}):
        raise ValueError("manifest protocol checksum mismatch")
    if "cuda_allocator_budget_gib" in spec:
        raise ValueError("diagnostics must use native caching without a memory cap")
    if spec["targets"] != list(range(21)) or spec["target_chunk"] != 1 or spec["pixel_stride"] != 1:
        raise ValueError("diagnostics require all21 targets, chunk1 and native pixel stride1")
    np.random.seed(int(spec["seed"]))
    torch.manual_seed(int(spec["seed"]))
    if not args.gate and manifest["limited_gate_manifest"]:
        raise ValueError("a gate-only input manifest cannot be used as a full diagnostic")
    device = diagnostic_device()
    labels = args.labels or list(manifest["checkpoints"])
    items = [item for item in manifest["ready"] if not args.datasets or item["dataset"] in args.datasets]
    if args.gate:
        items = items[:1]
    if not items:
        raise ValueError("no inputs for this worker")
    result_dir = root / ("gate" if args.gate else "results")
    for label in labels:
        checkpoint_spec = manifest["checkpoints"][label]
        if file_identity(checkpoint_spec["path"]) != checkpoint_spec["identity"]:
            raise ValueError(f"immutable checkpoint identity changed: {label}; reverify SHA-256")
        # Local full training checkpoints include trusted, SHA-verified RNG metadata.
        checkpoint = torch.load(checkpoint_spec["path"], map_location="cpu", mmap=True, weights_only=False)
        if checkpoint["training_state"]["global_step"] != checkpoint_spec["step"]:
            raise ValueError("checkpoint step mismatch")
        config = dict(checkpoint["config"])
        # Architecture comes from THIS checkpoint, not the historical production
        # selection fields embedded in the continuation configs. Runtime input
        # paths come from the current audited config; strict loading is mandatory.
        runtime = manifest["data_config"]
        for key in ("wan_root", "vae_checkpoint", "text_conditions", "prompts"):
            config[key] = runtime[key]
        if checkpoint["config"]["prompts"] != runtime["prompts"]:
            raise ValueError("checkpoint prompt mismatch")
        for key in ("wan_checkpoint", "wan_dit_root", "empty_text_condition"):
            config.pop(key, None)
        config["gradient_checkpointing"] = False
        config["precision"] = "bf16"
        torch.cuda.reset_peak_memory_stats(device)
        model = build_real_model(config, device, load_wan_pretrained=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.requires_grad_(False).eval()
        mean = np.asarray(checkpoint["coordinate_mean"], np.float32).reshape(1, 3, 1, 1)
        scale = np.asarray(checkpoint["coordinate_scale"], np.float32).reshape(1, 3, 1, 1)
        # Release mmap-backed optimizer state; inference never transfers it to GPU.
        del checkpoint
        conditions = {}
        emit("diagnostic_model_loaded", label=label, parameters=sum(p.numel() for p in model.parameters()),
             trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
             environment=environment())
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for item in items:
                source_paths = [result_dir / label / item["cohort"] / item["dataset"] /
                                f"clip-{item['index']:06d}-source-{source:02d}.json" for source in item["sources"]]
                if all(path.is_file() for path in source_paths):
                    for path in source_paths:
                        record = json.loads(path.read_text())
                        if (record["protocol"] != manifest["protocol"] or
                                record["checkpoint_sha256"] != checkpoint_spec["sha256"]):
                            raise ValueError("result resume protocol mismatch")
                    continue
                started = time.monotonic()
                if digest(item["prepared"]) != item["prepared_sha256"]:
                    raise ValueError("prepared marker changed")
                metadata = json.loads(Path(item["prepared"]).read_text())
                if metadata["protocol"] != item.get("prepared_protocol", manifest["protocol"]):
                    raise ValueError("prepared input protocol mismatch")
                clip = load_arrays(metadata["files"]["clip"])
                camera = {key.removeprefix("camera_"): value for key, value in clip.items() if key.startswith("camera_")}
                name = item["dataset"]
                if name not in conditions:
                    condition, prompt_meta = load_inference_text_condition(config, name)
                    conditions[name] = condition.to(device, dtype=torch.bfloat16)
                latent = torch.from_numpy(clip["latent"])[None].to(device, dtype=torch.bfloat16)
                z4d = model.backbone(latent, conditions[name])
                for source, path in zip(item["sources"], source_paths):
                    data = load_arrays(metadata["files"][str(source)])
                    normalized = decode(model, z4d, clip["rgb"], source, list(range(21)), device)
                    prediction = normalized * scale + mean
                    reverse_source = int(data["reverse_source"])
                    reverse = decode(model, z4d, clip["rgb"], reverse_source, [source], device) * scale + mean
                    metrics = stratified_epe(prediction, data["xyz"], data["valid"], data["visible"], source,
                                             data["depth_distance"], boundary_px=spec["boundary_distance_px"],
                                             interior_px=spec["interior_distance_px"],
                                             minimum_frames=spec["minimum_track_valid_frames"],
                                             motion_static_m=spec.get("motion_static_m", 0.01),
                                             motion_large_m=spec.get("motion_large_m", 0.1))
                    metrics["cycle"] = cycle_diagnostic(prediction, reverse[0], data, camera, source)
                    if "segmentation" in data:
                        metrics["cross_instance"] = cross_instance_neighbor_proxy(
                            prediction, data["xyz"], data["valid"], data["segmentation"], source,
                            radius=spec["neighbor_radius_px"], min_separation=spec["neighbor_min_separation_m"])
                    torch.cuda.synchronize(device)
                    peak = torch.cuda.max_memory_allocated(device) / 2 ** 30
                    prediction_path = path.with_suffix(".npz")
                    checksum = save_npz(prediction_path, xyz=prediction)
                    record = {"protocol": manifest["protocol"], "checkpoint_sha256": checkpoint_spec["sha256"],
                              "label": label, "item": item, "source": source, "metrics": metrics,
                              "prediction": {"path": str(prediction_path), "sha256": checksum},
                              "peak_allocated_gib": peak,
                              "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2 ** 30,
                              "elapsed_clip_seconds": time.monotonic() - started}
                    path.parent.mkdir(parents=True, exist_ok=True)
                    atomic_json(path, record)
                    emit("diagnostic_source_complete", label=label, dataset=name, cohort=item["cohort"],
                         index=item["index"], source=source, raw_epe_m=metrics["groups"]["all/all/all"]["mean"],
                         boundary_epe_m=metrics["groups"]["all/boundary/all"]["mean"], peak_allocated_gib=peak,
                         elapsed_clip_seconds=time.monotonic() - started)
                    del data, normalized, prediction, reverse, metrics
                del clip, camera, latent, z4d
        del model, conditions
        gc.collect()  # Keep the native allocator cache for the next checkpoint.
    emit("DIAGNOSTIC_GATE_OK" if args.gate else "DIAGNOSTIC_WORKER_OK", labels=labels, clips=len(items))


def metric_values(record):
    metrics = record["metrics"]
    values = {key: item for key, item in metrics["groups"].items()}
    for prefix in ("track_mean_epe", "displacement_epe"):
        values.update({f"{prefix}/{key}": item for key, item in metrics[prefix].items()})
    values["sim3/all"] = metrics["sim3_epe"]
    if "cross_instance" in metrics:
        for key in ("neighbor_closer", "toward_neighbor", "between_surfaces"):
            values[f"cross_instance/{key}"] = metrics["cross_instance"][key]
    cycle = metrics["cycle"]
    values["cycle/pixel_error"] = {"mean": cycle["pixel_error"] if cycle["valid_points"] else None,
                                    "count": cycle["valid_points"], "sum": cycle["pixel_error"] * cycle["valid_points"]}
    return values


def aggregate(args):
    root = output_root(args.output_root)
    manifest = json.loads(Path(args.manifest).read_text())
    labels = list(manifest["checkpoints"])
    records, missing = {}, []
    for label in labels:
        for item in manifest["ready"]:
            for source in item["sources"]:
                path = root / "results" / label / item["cohort"] / item["dataset"] / f"clip-{item['index']:06d}-source-{source:02d}.json"
                key = (item["cohort"], item["dataset"], item["index"], source)
                if not path.is_file():
                    missing.append({"label": label, "key": key})
                    continue
                record = json.loads(path.read_text())
                if (record["protocol"] != manifest["protocol"] or
                        record["checkpoint_sha256"] != manifest["checkpoints"][label]["sha256"]):
                    raise ValueError("mixed diagnostic protocols")
                records[(label, key)] = record
    summary = {"protocol": manifest["protocol"], "complete": not missing,
               "blocked_inputs": manifest["blocked"], "missing_results": missing, "groups": {}}
    comparisons = manifest["spec"].get("comparisons", [
        ["h027_130k", "h023_100k"], ["h030_150k_w03", "h027_130k"],
        ["h030_150k_norm30", "h030_150k_w03"], ["h030_150k_norm30", "h023_100k"]])
    lines = ["# Matched checkpoint diagnostics", "", f"Complete ready-input matrix: {not missing}",
             f"Blocked planned clips: {len(manifest['blocked'])}; missing results: {len(missing)}", "",
             "Historical replay is TRAIN data, not held-out generalization. Validation is a small screen.",
             "Primary metrics use raw source-camera XYZ; Sim(3) is separately GT-assisted.",
             "Intervals resample parent groups, not pixels. Neighbor identity is only a GT-neighbor proxy.", ""]
    primary = ["all/all/all", "all/boundary/all", "all/interior/all", "pointmap/boundary/all",
               "tracking/all/all", "gap8plus/boundary/all", "tracking/all/occluded",
               "track_mean_epe/boundary", "displacement_epe/boundary", "sim3/all",
               "cross_instance/neighbor_closer", "cycle/pixel_error",
               "tracking/motion_le_1cm/all", "tracking/motion_1to10cm/all",
               "tracking/motion_gt_10cm/all"]
    for cohort in ("historical_train_replay", "validation_screen"):
        for name in DATASET_NAMES:
            keys = sorted({key for label, key in records if key[:2] == (cohort, name)})
            if not keys:
                continue
            group = {"checkpoints": {}, "comparisons": {}}
            for label in labels:
                selected = [records[(label, key)] for key in keys if (label, key) in records]
                if not selected:
                    continue
                accum = {}
                for record in selected:
                    for metric, value in metric_values(record).items():
                        entry = accum.setdefault(metric, {"sum": 0.0, "count": 0, "means": []})
                        entry["sum"] += value["sum"]
                        entry["count"] += value["count"]
                        if value["mean"] is not None:
                            entry["means"].append(value["mean"])
                group["checkpoints"][label] = {
                    metric: {"count": value["count"],
                             "point_weighted_mean": value["sum"] / value["count"] if value["count"] else None,
                             "source_macro_mean": float(np.mean(value["means"])) if value["means"] else None}
                    for metric, value in accum.items()}
            for candidate, baseline in comparisons:
                common = [key for key in keys if (candidate, key) in records and (baseline, key) in records]
                metrics_out = {}
                for metric in primary:
                    parent_deltas, base_values, candidate_values = {}, [], []
                    for key in common:
                        a, b = records[(candidate, key)], records[(baseline, key)]
                        ma, mb = metric_values(a).get(metric), metric_values(b).get(metric)
                        if ma is None or mb is None:
                            continue
                        if not metric.startswith("cycle/") and ma["count"] != mb["count"]:
                            raise ValueError(f"GT metric population changed: {metric}")
                        if ma["mean"] is None or mb["mean"] is None:
                            continue
                        parent = a["item"]["parent_id"]
                        parent_deltas.setdefault(parent, []).append(ma["mean"] - mb["mean"])
                        candidate_values.append(ma["mean"])
                        base_values.append(mb["mean"])
                    if not base_values:
                        continue
                    deltas = np.asarray([np.mean(v) for v in parent_deltas.values()])
                    rng = np.random.default_rng(manifest["spec"]["seed"])
                    interval = None
                    if len(deltas) >= 2:
                        sampled = deltas[rng.integers(0, len(deltas), (5000, len(deltas)))].mean(axis=1)
                        interval = np.quantile(sampled, [0.025, 0.975]).tolist()
                    base_mean, cand_mean = float(np.mean(base_values)), float(np.mean(candidate_values))
                    metrics_out[metric] = {"baseline_source_macro": base_mean, "candidate_source_macro": cand_mean,
                                           "relative_percent": 100 * (cand_mean / base_mean - 1) if base_mean else None,
                                           "parent_mean_delta": float(deltas.mean()), "parent_bootstrap_ci95": interval,
                                           "parents": len(deltas), "source_pairs": len(base_values)}
                group["comparisons"][f"{candidate}_vs_{baseline}"] = metrics_out
            summary["groups"][f"{cohort}/{name}"] = group
            lines += [f"## {cohort} / {name}", "", "| checkpoint | raw EPE | boundary | interior | occluded tracking |", "|---|---:|---:|---:|---:|"]
            for label, metrics in group["checkpoints"].items():
                def fmt(metric):
                    value = metrics.get(metric, {}).get("point_weighted_mean")
                    return "n/a" if value is None else f"{value:.6f}"
                lines.append(f"| {label} | {fmt('all/all/all')} | {fmt('all/boundary/all')} | {fmt('all/interior/all')} | {fmt('tracking/all/occluded')} |")
            lines.append("")
    atomic_json(root / "summary.json", summary)
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    emit("DIAGNOSTIC_AGGREGATE_OK", complete=not missing, blocked_clips=len(manifest["blocked"]),
         missing=len(missing), report=str(root / "REPORT.md"))
    if missing and not args.allow_partial:
        raise RuntimeError("diagnostic matrix incomplete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "aggregate"))
    parser.add_argument("--protocol", default=str(ROOT / "configs/h030_matched_diagnostics.yaml"))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--manifest-name", default="manifest.json")
    parser.add_argument("--limit-clips", type=int)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--datasets", nargs="+", choices=DATASET_NAMES)
    parser.add_argument("--cohorts", nargs="+", choices=("historical_train_replay", "validation_screen"))
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == "prepare":
        prepare(args)
    else:
        if not args.manifest:
            args.manifest = str(Path(args.output_root) / "manifest.json")
        (run if args.command == "run" else aggregate)(args)


if __name__ == "__main__":
    main()
