#!/usr/bin/env python3
"""Measure the largest safe physical batch for one H017 decoder arm."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from worldbridge.dense4d import DenseQueryDecoder


def decoder_parameter_count(config: dict) -> int:
    decoder = DenseQueryDecoder(
        num_frames=int(config["clip_length"]),
        latent_shape=(
            int(config.get("geometry_dim", 128)),
            int(config["clip_length"]),
            int(config.get("geometry_spatial_size", 16)),
            int(config.get("geometry_spatial_size", 16)),
        ),
        query_dim=int(config["query_dim"]),
        embedding_dim=int(config.get("embedding_dim", 128)),
        num_layers=int(config["num_cross_attn_layers"]),
        num_heads=int(config["num_heads"]),
        upsample_channels=tuple(int(value) for value in config["upsample_channels"]),
        output_size=(int(config["image_size"]), int(config["image_size"])),
        structured_motion_slots=int(config.get("motion_slots", 0)),
        structured_local_queries=bool(config.get("structured_local_queries", True)),
    )
    return sum(parameter.numel() for parameter in decoder.parameters())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--batch-sizes", nargs="+", type=int, required=True)
    parser.add_argument("--safe-gib", type=float, default=78.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    candidates = sorted(set(int(value) for value in args.batch_sizes))
    if not candidates or candidates[0] < 1:
        raise ValueError("batch sizes must be positive")
    base_path = Path(args.base_config).resolve()
    base = yaml.safe_load(base_path.read_text())
    decoder_parameters = decoder_parameter_count(base)
    work_root = Path(args.work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    cache_root = work_root / "screen_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    selected_batch = None
    stop_reason = "candidate_range_exhausted"
    for batch_size in candidates:
        config = copy.deepcopy(base)
        config.update({
            "max_clips": max(candidates),
            "batch_size": batch_size,
            "steps": 1,
            "matched_epoch_sampling": True,
            "geometry_cache_entries": max(candidates),
            "sample_cache": str(cache_root / "samples.pkl"),
            "clean_latent_cache": str(cache_root / "clean_latents.pt"),
            "save_checkpoint": False,
            "keep_checkpoint": False,
            "checkpoint_every_steps": 0,
            "tracking": {"enabled": False},
        })
        config_path = work_root / f"batch{batch_size}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        run_dir = work_root / f"batch{batch_size}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        command = [
            sys.executable, str(ROOT / "train_dense4d.py"),
            "--config", str(config_path), "--output-dir", str(run_dir),
            "--device", args.device,
        ]
        print(json.dumps({
            "event": "capacity_candidate_start", "batch_size": batch_size,
            "decoder_parameters": decoder_parameters, "command": command,
        }), flush=True)
        environment = os.environ.copy()
        environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        completed = subprocess.run(
            command, cwd=ROOT, env=environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        (work_root / f"batch{batch_size}.log").write_text(completed.stdout)
        metric_path = run_dir / "train_metrics.json"
        if completed.returncode == 0 and metric_path.exists():
            metrics = json.loads(metric_path.read_text())
            peak_gib = float(metrics["peak_cuda_memory_gib"])
            safe = peak_gib <= args.safe_gib
            result = {
                "batch_size": batch_size,
                "status": "succeeded" if safe else "succeeded_above_safe_ceiling",
                "returncode": completed.returncode,
                "peak_cuda_memory_gib": peak_gib,
                "elapsed_seconds": float(metrics["elapsed_seconds"]),
                "trainable_parameters": int(metrics["trainable_parameters"]),
            }
            results.append(result)
            print(json.dumps({"event": "capacity_candidate_result", **result}), flush=True)
            if safe:
                selected_batch = batch_size
                continue
            stop_reason = "safe_memory_ceiling"
            break
        lower_output = completed.stdout.lower()
        oom_markers = (
            "out of memory",
            "cublas_status_alloc_failed",
            "cudnn_status_alloc_failed",
            "cuda_error_out_of_memory",
        )
        result = {
            "batch_size": batch_size,
            "status": "failed",
            "returncode": completed.returncode,
            "oom": any(marker in lower_output for marker in oom_markers),
            "log_tail": completed.stdout[-2000:],
        }
        results.append(result)
        print(json.dumps({"event": "capacity_candidate_result", **result}), flush=True)
        stop_reason = "oom" if result["oom"] else "non_oom_failure"
        break

    payload = {
        "protocol": "h017_decoder_capacity_screen",
        "base_config": str(base_path),
        "decoder_parameters": decoder_parameters,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "safe_memory_ceiling_gib": args.safe_gib,
        "candidate_batches": candidates,
        "results": results,
        "selected_batch_size": selected_batch,
        "stop_reason": stop_reason,
        "bounded_above": stop_reason in {"oom", "safe_memory_ceiling"},
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)
    if selected_batch is None or stop_reason == "non_oom_failure":
        raise SystemExit(1)
    print("H017_CAPACITY_SCREEN_OK", flush=True)


if __name__ == "__main__":
    main()
