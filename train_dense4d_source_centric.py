#!/usr/bin/env python3
"""H004 source-centric ablation training with bounded asynchronous geometry."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import fcntl
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from worldbridge.data import MOViFDataset
from worldbridge.dense4d import masked_visibility_bce
from worldbridge.dense4d_data import CoordinateStats, DynamicPointmapCache, dense_pair_targets
from worldbridge.dense4d_prefetch import (
    SourceCentricBatch,
    SourceCentricPlan,
    SourceCentricPrefetcher,
    build_source_centric_batch,
    make_source_centric_plan,
    source_centric_loss_weights,
    weighted_masked_pair_smooth_l1,
)
from worldbridge.dense4d_runtime import (
    build_real_model,
    encode_clean_video_latents,
    optimizer_trainable_count,
    parameter_groups,
    precision_dtype,
    save_checkpoint,
)
from train_dense4d import init_wandb, load_or_create_clean_latents, load_or_create_samples


REQUIRED = {
    "data_root", "wan_root", "empty_text_condition", "coordinate_stats", "sample_cache",
    "clean_latent_cache", "clip_length", "image_size", "batch_size", "steps", "query_dim",
    "num_cross_attn_layers", "num_heads", "upsample_channels", "learning_rate",
    "backbone_learning_rate", "precision", "gradient_accumulation", "trainable_mode", "seed",
}


def _split_cache_metadata(config: dict[str, Any], count: int, split: str) -> dict[str, Any]:
    return {
        "format": 1,
        "data_root": str(Path(config["data_root"]).resolve()),
        "split": split,
        "clip_length": int(config["clip_length"]),
        "clip_start": int(config.get("clip_start", 0)),
        "seed": int(config["seed"]),
        "count": int(count),
    }


def _locked_load_or_create_samples(dataset: MOViFDataset, config: dict[str, Any],
                                    split: str, path: str | Path) -> list[Any]:
    path = Path(path)
    metadata = _split_cache_metadata(config, len(dataset), split)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        if path.exists():
            with path.open("rb") as handle:
                payload = __import__("pickle").load(handle)
            if payload.get("metadata") != metadata:
                raise RuntimeError(f"{split} sample cache metadata mismatch: {path}")
            samples = payload["samples"]
            if len(samples) != len(dataset):
                raise RuntimeError(f"{split} sample cache length mismatch: {path}")
            print(f"{split.upper()}_SAMPLE_CACHE_HIT: {path} ({len(samples)} clips)", flush=True)
            return samples
        samples = [dataset[index] for index in range(len(dataset))]
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            __import__("pickle").dump({"metadata": metadata, "samples": samples}, handle,
                                      protocol=__import__("pickle").HIGHEST_PROTOCOL)
        temporary.replace(path)
        print(f"{split.upper()}_SAMPLE_CACHE_CREATED: {path} ({len(samples)} clips)", flush=True)
        return samples


def _locked_load_or_create_latents(samples: list[Any], config: dict[str, Any],
                                   split: str, path: str | Path, device: torch.device) -> list[torch.Tensor]:
    path = Path(path)
    metadata = {
        **_split_cache_metadata(config, len(samples), split),
        "wan_root": str(Path(config["wan_root"]).resolve()),
        "dtype": "float32",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        if path.exists():
            payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
            if payload.get("metadata") != metadata:
                raise RuntimeError(f"{split} latent cache metadata mismatch: {path}")
            latents = payload["latents"]
            if tuple(latents.shape) != (len(samples), 16, 6, 16, 16):
                raise RuntimeError(f"{split} latent cache shape mismatch: {tuple(latents.shape)}")
            print(f"{split.upper()}_CLEAN_LATENT_CACHE_HIT: {path} ({len(samples)} clips)", flush=True)
            return [latents[index:index + 1] for index in range(len(samples))]
        encoded = encode_clean_video_latents(samples, config["wan_root"], device)
        stacked = torch.cat(encoded, dim=0).contiguous().float().cpu()
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save({"metadata": metadata, "latents": stacked}, temporary)
        temporary.replace(path)
        print(f"{split.upper()}_CLEAN_LATENT_CACHE_CREATED: {path} ({len(samples)} clips)", flush=True)
        return [stacked[index:index + 1] for index in range(len(samples))]


def _pin_cpu(array: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    return tensor.pin_memory() if torch.cuda.is_available() else tensor


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _estimate_latent_feature_stats(model: torch.nn.Module, clean_latents: list[torch.Tensor],
                                   config: dict[str, Any], device: torch.device,
                                   dtype: torch.dtype) -> tuple[np.ndarray, np.ndarray]:
    """Estimate fixed D1 channel statistics before any optimization updates.

    The first deterministic ``latent_stats_clips`` train-cache entries are used;
    this is a representation-only statistic, not a geometry target or a learned
    per-batch normalization.  Statistics are frozen for the complete run.
    """
    count = min(int(config.get("latent_stats_clips", 64)), len(clean_latents))
    if count < 1:
        raise ValueError("latent_stats_clips must select at least one cached train clip")
    was_training = model.backbone.training
    model.backbone.eval()
    channels = int(clean_latents[0].shape[1])
    total = 0
    channel_sum = torch.zeros(channels, dtype=torch.float64)
    channel_sq_sum = torch.zeros(channels, dtype=torch.float64)
    with torch.inference_mode():
        for latent_cpu in clean_latents[:count]:
            latent = latent_cpu.to(device=device, dtype=dtype, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda" and dtype == torch.bfloat16):
                feature = model.backbone(latent)
            feature = feature.float()
            values = feature.permute(1, 0, 2, 3, 4).reshape(channels, -1).double().cpu()
            channel_sum += values.sum(dim=1)
            channel_sq_sum += (values * values).sum(dim=1)
            total += values.shape[1]
    if was_training:
        model.backbone.train()
    mean = channel_sum / total
    variance = (channel_sq_sum / total - mean * mean).clamp_min(1e-8)
    scale = variance.sqrt().clamp_min(1e-4)
    mean_np = mean.numpy().astype(np.float32)
    scale_np = scale.numpy().astype(np.float32)
    print(json.dumps({
        "event": "latent_feature_stats",
        "adapter": "fixed_whiten",
        "clips": count,
        "elements_per_channel": total,
        "mean": mean_np.tolist(),
        "scale": scale_np.tolist(),
    }), flush=True)
    return mean_np, scale_np


class GPUUtilizationSampler:
    """Sample NVML utilization during each full step instead of at an idle boundary."""

    def __init__(self, interval_seconds: float = 0.2):
        self.interval_seconds = float(interval_seconds)
        self._values: list[float] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pynvml = None
        self._handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            visible = os.getenv("CUDA_VISIBLE_DEVICES", "")
            physical = int(visible.split(",")[0]) if visible else 0
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(physical)
            self._pynvml = pynvml
        except Exception:
            self._pynvml = None
            self._handle = None

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                value = float(self._pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu)
            except Exception:
                continue
            with self._lock:
                self._values.append(value)

    def reset(self) -> None:
        with self._lock:
            self._values.clear()

    def mean_and_reset(self) -> float | None:
        with self._lock:
            result = float(np.mean(self._values)) if self._values else None
            self._values.clear()
        return result

    def __enter__(self) -> "GPUUtilizationSampler":
        if self._pynvml is not None:
            self._thread = threading.Thread(target=self._sample, name="h004-nvml", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass


def _fixed_train_plan(config: dict[str, Any], samples_count: int, step: int) -> SourceCentricPlan:
    return make_source_centric_plan(
        samples_count, int(config["batch_size"]), int(config["clip_length"]), int(step)
    )


def _batch_loss(model: torch.nn.Module, clean: torch.Tensor, batch: SourceCentricBatch,
                device: torch.device, dtype: torch.dtype, config: dict[str, Any],
                pair_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Any]:
    tensors = batch.to_device(device)
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda" and dtype == torch.bfloat16):
        prediction, z4d, output = model(clean, tensors["source"], tensors["target"])
        xyz_loss = weighted_masked_pair_smooth_l1(
            prediction.float(), tensors["target_xyz"].float(), tensors["valid"], pair_weights,
            beta=float(config.get("smooth_l1_beta", 0.05)),
        )
        visibility_loss = prediction.sum() * 0.0
        if bool(config.get("visibility_head", False)):
            if output.visibility_logits is None:
                raise RuntimeError("visibility_head=true but decoder produced no visibility logits")
            visibility_loss = masked_visibility_bce(
                output.visibility_logits.float(), tensors["visible"], tensors["valid"],
                tensors["source"], tensors["target"], float(config["visibility_pos_weight"]),
            )
        loss = xyz_loss + float(config.get("lambda_visibility", 0.0)) * visibility_loss
    return loss, prediction, output


def _train_metric(prediction: torch.Tensor, batch: SourceCentricBatch,
                  stats: CoordinateStats) -> float:
    prediction_metric = prediction.detach().float().cpu().numpy()
    prediction_metric = prediction_metric * stats.scale[None, None, :, None, None] \
        + stats.mean[None, None, :, None, None]
    errors = np.linalg.norm(prediction_metric - batch.metric_xyz, axis=2)
    return float(errors[batch.valid].mean())


def _accumulator() -> dict[str, list[float]]:
    return {}


def _add_metric(acc: dict[str, list[float]], name: str, errors: np.ndarray,
                xyz_abs: np.ndarray, mask: np.ndarray) -> None:
    points = int(mask.sum())
    if points == 0:
        return
    values = acc.setdefault(name, [0.0, 0.0, 0.0])
    values[0] += float(errors[mask].sum())
    xyz_error = xyz_abs.sum(axis=1) if xyz_abs.ndim == errors.ndim + 1 else xyz_abs
    values[1] += float(xyz_error[mask].sum())
    values[2] += points


def _finish_metrics(acc: dict[str, list[float]]) -> dict[str, Any]:
    result = {}
    for name, (epe_sum, xyz_sum, points) in acc.items():
        result[name] = {
            "points": int(points),
            "epe": epe_sum / points,
            "xyz_mae": xyz_sum / (points * 3),
        }
    return result


def evaluate_fixed_validation(model: torch.nn.Module, samples: list[Any], latents: list[torch.Tensor],
                              stats: CoordinateStats, config: dict[str, Any], device: torch.device,
                              dtype: torch.dtype) -> dict[str, Any]:
    """Evaluate 32 fixed validation clips over all 441 legal source-target pairs."""
    accumulator = _accumulator()
    visibility_sum = 0.0
    visibility_points = 0
    was_training = model.training
    model.eval()
    pair_chunk = int(config.get("validation_pair_chunk", 7))
    with torch.inference_mode():
        for clip_index, (sample, latent) in enumerate(zip(samples, latents)):
            cache = DynamicPointmapCache(max_entries=21)
            source_all = np.repeat(np.arange(21, dtype=np.int64), 21)
            target_all = np.tile(np.arange(21, dtype=np.int64), 21)
            for start in range(0, len(source_all), pair_chunk):
                source = source_all[start:start + pair_chunk]
                target = target_all[start:start + pair_chunk]
                normalized, metric, visible, valid = dense_pair_targets(sample, source, target, stats, cache)
                source_tensor = torch.from_numpy(source[None]).to(device)
                target_tensor = torch.from_numpy(target[None]).to(device)
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda" and dtype == torch.bfloat16):
                    prediction, _, output = model(latent.to(device=device, dtype=dtype), source_tensor, target_tensor)
                prediction_metric = prediction.float().cpu().numpy()[0] * stats.scale[None, :, None, None] \
                    + stats.mean[None, :, None, None]
                errors = np.linalg.norm(prediction_metric - metric, axis=1)
                xyz_abs = np.abs(prediction_metric - metric)
                pair_valid = valid
                off = source != target
                gap = np.abs(target - source)
                group_pairs = {
                    "all_valid": np.ones(len(source), dtype=bool),
                    "diagonal": ~off,
                    "off_diagonal_all_valid": off,
                    "source_zero": source == 0,
                    "source_positive": source > 0,
                    "forward": target > source,
                    "backward": target < source,
                    "short_gap": (gap >= 1) & (gap <= 5),
                    "long_gap": gap >= 10,
                }
                for name, pair_select in group_pairs.items():
                    _add_metric(accumulator, name, errors, xyz_abs,
                                pair_valid & pair_select[:, None, None])
                    _add_metric(accumulator, f"{name}_visible", errors, xyz_abs,
                                pair_valid & visible & pair_select[:, None, None])
                    _add_metric(accumulator, f"{name}_occluded_valid", errors, xyz_abs,
                                pair_valid & ~visible & pair_select[:, None, None])
                if output.visibility_logits is not None and np.any(off):
                    logits = output.visibility_logits.float().cpu().numpy()[0, :, 0]
                    target_m = visible.astype(np.float32)
                    mask = valid & off[:, None, None]
                    if np.any(mask):
                        pos_weight = float(config["visibility_pos_weight"])
                        bce = np.maximum(logits, 0.0) - logits * target_m \
                            + np.log1p(np.exp(-np.abs(logits)))
                        bce += (pos_weight - 1.0) * target_m * np.logaddexp(0.0, -logits)
                        visibility_sum += float(bce[mask].sum())
                        visibility_points += int(mask.sum())
            print(json.dumps({"event": "validation_clip", "clip": clip_index + 1, "clips": len(samples)}), flush=True)
    if was_training:
        model.train()
    result = {"clips": len(samples), "pairs": 441, "metrics": _finish_metrics(accumulator)}
    if visibility_points:
        result["visibility_bce_off_diagonal"] = visibility_sum / visibility_points
    return result


def _load_validation(config: dict[str, Any], device: torch.device) -> tuple[list[Any], list[torch.Tensor]]:
    dataset = MOViFDataset(
        config["data_root"], split="validation", clip_length=int(config["clip_length"]),
        clip_start=int(config.get("clip_start", 0)), max_examples=32, seed=int(config["seed"]),
    )
    samples = _locked_load_or_create_samples(
        dataset, config, "validation", config["validation_sample_cache"],
    )
    latents = _locked_load_or_create_latents(
        samples, config, "validation", config["validation_clean_latent_cache"], device,
    )
    return samples, latents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=("sync", "async"), default="async")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    missing = sorted(REQUIRED - set(config))
    if missing:
        raise ValueError(f"missing required config keys: {missing}")
    if int(config["clip_length"]) != 21 or int(config["image_size"]) != 128:
        raise ValueError("source-centric H004 protocol requires clip_length=21 and image_size=128")
    if int(config["batch_size"]) != 16 or int(config["steps"]) != 1000:
        raise ValueError("formal source-centric screen requires batch_size=16 and steps=1000")
    if int(config.get("gradient_accumulation", 1)) != 1 or str(config["precision"]).lower() not in {"bf16", "bfloat16"}:
        raise ValueError("formal source-centric screen fixes accumulation=1 and BF16")
    if str(config["trainable_mode"]) != "full" or int(config["seed"]) != 2029:
        raise ValueError("formal source-centric screen fixes trainable_mode=full and seed=2029")
    if int(config.get("num_query_pairs", 21)) != 21:
        raise ValueError("source-centric training enumerates exactly all 21 targets")
    if int(config.get("geometry_workers", 8)) != 8 or int(config.get("prefetch_queue_depth", 2)) != 2:
        raise ValueError("source-centric protocol fixes 8 geometry workers and queue depth 2")
    if not np.isclose(float(config.get("xyz_pair_weight_diagonal", 1 / 3)), 1 / 3) or not np.isclose(
        float(config.get("xyz_pair_weight_off_diagonal", 2 / 3)), 2 / 3
    ):
        raise ValueError("XYZ pair weights must be exactly 1/3 diagonal and 2/3 off-diagonal")
    for excluded in ("depth_loss", "depth_loss_weight", "reprojection_loss", "reprojection_loss_weight",
                     "iterative_dit", "multi_step_dit"):
        if bool(config.get(excluded, False)):
            raise ValueError(f"{excluded} is excluded from the H004 source-centric screen")
    arm = str(config.get("ablation_arm"))
    actual = (str(config.get("rope_mode", "2d")), int(config["num_cross_attn_layers"]),
              bool(config.get("visibility_head", False)))
    expected = {
        "B0": ("2d", 2, False), "E3": ("3d", 2, False),
        "E5": ("2d", 2, True), "E6": ("2d", 4, False),
        "D1": ("2d", 2, False), "D2": ("2d", 2, False), "D3": ("2d", 2, False),
    }
    expected_adapters = {
        "B0": "none", "E3": "none", "E5": "none", "E6": "none",
        "D1": "fixed_whiten", "D2": "channel_affine", "D3": "conv1x1",
    }
    adapter = str(config.get("latent_adapter", "none")).lower()
    if arm not in expected or actual != expected[arm] or adapter != expected_adapters.get(arm):
        raise ValueError(
            f"strict arm mismatch: {arm=} has architecture={actual}, adapter={adapter!r}; "
            f"expected architecture={expected.get(arm)}, adapter={expected_adapters.get(arm)!r}"
        )
    if arm in {"D1", "D2", "D3"} and str(config.get("backbone_readout", "wan_velocity")) != "wan_velocity":
        raise ValueError(f"{arm} must adapt the Wan velocity readout")
    if arm == "D1" and int(config.get("latent_stats_clips", 0)) < 1:
        raise ValueError("D1 requires a positive deterministic latent_stats_clips count")
    if actual[0] == "3d" and int(config["query_dim"]) // int(config["num_heads"]) != 32:
        raise ValueError("E3 requires head_dim=32")
    if arm == "E5":
        if config.get("visibility_target", "M") != "M" or config.get("visibility_mask", "A") != "A" \
                or not bool(config.get("visibility_off_diagonal_only", True)):
            raise ValueError("E5 fixes visibility target=M, mask=A, and off-diagonal-only BCE")
        if "visibility_pos_weight" not in config or not np.isfinite(float(config["visibility_pos_weight"])):
            raise ValueError("E5 requires a finite precomputed visibility_pos_weight")
        if not np.isclose(float(config.get("lambda_visibility", -1)), 0.1):
            raise ValueError("E5 fixes lambda_visibility=0.1")
    elif not np.isclose(float(config.get("lambda_visibility", 0.0)), 0.0):
        raise ValueError(f"{arm} must remain XYZ-only")

    seed = int(config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("formal Wan ablations require CUDA")
    torch.cuda.set_device(0 if device.index is None else device.index)
    torch.cuda.reset_peak_memory_stats(device)
    dtype = precision_dtype(config["precision"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    wandb_run = init_wandb(config)

    # Reserve the selected GPU with Wan before the 30 GB CPU sample-cache load;
    # this minimizes the post-handoff interval during which the card looks free.
    model = build_real_model(config, device)
    groups = parameter_groups(model, config)
    optimizer = torch.optim.AdamW(groups, weight_decay=float(config.get("weight_decay", 0.0)))
    dataset = MOViFDataset(
        config["data_root"], split="train", clip_length=21, clip_start=int(config.get("clip_start", 0)),
        max_examples=None, seed=seed,
    )
    samples = load_or_create_samples(dataset, config)
    stats = CoordinateStats.from_npz(config["coordinate_stats"])
    clean_latents = load_or_create_clean_latents(samples, config, device)
    if adapter == "fixed_whiten":
        latent_mean, latent_scale = _estimate_latent_feature_stats(
            model, clean_latents, config, device, dtype,
        )
        model.decoder.set_latent_stats(
            torch.from_numpy(latent_mean).to(device), torch.from_numpy(latent_scale).to(device),
        )
        config["latent_feature_mean"] = latent_mean.tolist()
        config["latent_feature_scale"] = latent_scale.tolist()
        config["latent_stats_source"] = "first_deterministic_train_cache_entries_pre_update"
        (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        if wandb_run is not None:
            wandb_run.config.update({
                "latent_feature_mean": config["latent_feature_mean"],
                "latent_feature_scale": config["latent_feature_scale"],
                "latent_stats_clips": int(config["latent_stats_clips"]),
            }, allow_val_change=True)
    validation_samples, validation_latents = _load_validation(config, device)

    losses: list[float] = []
    train_epes: list[float] = []
    timings: list[dict[str, float | None]] = []
    backbone_gradient = False
    decoder_gradient = False
    adapter_gradient = adapter in {"none", "fixed_whiten"}
    pair_weights_cpu = None
    start_time = time.perf_counter()
    model.train()

    def plan_for(step: int) -> SourceCentricPlan:
        # This function is called only by the main training thread.
        return _fixed_train_plan(config, len(samples), step)

    optimizer.zero_grad(set_to_none=True)
    with GPUUtilizationSampler() as gpu_sampler, SourceCentricPrefetcher(
        samples, stats, workers=8, queue_depth=2,
        depth_tolerance=float(config.get("depth_tolerance", 0.05)),
        depth_relative_tolerance=float(config.get("depth_relative_tolerance", 0.01)),
    ) as prefetcher:
        if args.mode == "async":
            for prefill_step in range(min(2, int(config["steps"]))):
                prefetcher.submit(plan_for(prefill_step))
        for step in range(int(config["steps"])):
            iteration_start = time.perf_counter()
            gpu_sampler.reset()
            if args.mode == "async":
                batch, prefetch_wait = prefetcher.next()
                next_step = step + 2
                if next_step < int(config["steps"]):
                    prefetch_submit = prefetcher.submit(plan_for(next_step))
                else:
                    prefetch_submit = 0.0
            else:
                plan = plan_for(step)
                geometry_start = time.perf_counter()
                batch = build_source_centric_batch(
                    samples, stats, plan,
                    depth_tolerance=float(config.get("depth_tolerance", 0.05)),
                    depth_relative_tolerance=float(config.get("depth_relative_tolerance", 0.01)),
                )
                prefetch_wait = time.perf_counter() - geometry_start
                prefetch_submit = 0.0

            plan = batch.plan
            if pair_weights_cpu is None or pair_weights_cpu.shape != plan.source.shape:
                pair_weights_cpu = _pin_cpu(source_centric_loss_weights(plan.source, plan.target))
            pair_weights = pair_weights_cpu.to(device=device, non_blocking=True)
            clean_cpu = torch.cat([clean_latents[int(index)] for index in plan.sample_indices]).contiguous()
            if torch.cuda.is_available() and not clean_cpu.is_pinned():
                clean_cpu = clean_cpu.pin_memory()

            _cuda_sync(device)
            h2d_start = time.perf_counter()
            tensors = batch.to_device(device)
            clean = clean_cpu.to(device=device, dtype=dtype, non_blocking=True)
            _cuda_sync(device)
            h2d_seconds = time.perf_counter() - h2d_start

            logical_batch = int(plan.source.shape[0])
            microbatch_size = int(config.get("microbatch_size", logical_batch))
            if microbatch_size < 1 or logical_batch % microbatch_size:
                raise ValueError(f"microbatch_size={microbatch_size} must divide logical batch={logical_batch}")
            _cuda_sync(device)
            forward_start = time.perf_counter()
            predictions = []
            xyz_loss_value = 0.0
            visibility_loss_value = 0.0
            z4d_shape = None
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype == torch.bfloat16):
                for micro_start in range(0, logical_batch, microbatch_size):
                    micro_end = min(logical_batch, micro_start + microbatch_size)
                    fraction = (micro_end - micro_start) / logical_batch
                    prediction_micro, z4d_micro, output_micro = model(
                        clean[micro_start:micro_end],
                        tensors["source"][micro_start:micro_end],
                        tensors["target"][micro_start:micro_end],
                    )
                    xyz_loss_micro = weighted_masked_pair_smooth_l1(
                        prediction_micro.float(), tensors["target_xyz"][micro_start:micro_end].float(),
                        tensors["valid"][micro_start:micro_end], pair_weights[micro_start:micro_end],
                        beta=float(config.get("smooth_l1_beta", 0.05)),
                    )
                    visibility_loss_micro = prediction_micro.sum() * 0.0
                    if bool(config.get("visibility_head", False)):
                        visibility_loss_micro = masked_visibility_bce(
                            output_micro.visibility_logits.float(), tensors["visible"][micro_start:micro_end],
                            tensors["valid"][micro_start:micro_end], tensors["source"][micro_start:micro_end],
                            tensors["target"][micro_start:micro_end], float(config["visibility_pos_weight"]),
                        )
                    loss_micro = xyz_loss_micro + float(config.get("lambda_visibility", 0.0)) * visibility_loss_micro
                    (loss_micro * fraction / int(config.get("gradient_accumulation", 1))).backward()
                    predictions.append(prediction_micro.detach())
                    xyz_loss_value += float(xyz_loss_micro.detach()) * fraction
                    visibility_loss_value += float(visibility_loss_micro.detach()) * fraction
                    z4d_shape = list(z4d_micro.shape[1:])
            _cuda_sync(device)
            forward_backward_seconds = time.perf_counter() - forward_start
            prediction = torch.cat(predictions, dim=0)

            if step == 0:
                backbone_gradient = any(
                    parameter.grad is not None and torch.isfinite(parameter.grad).all()
                    for parameter in model.backbone.parameters() if parameter.requires_grad
                )
                decoder_gradient = any(
                    parameter.grad is not None and torch.isfinite(parameter.grad).all()
                    for parameter in model.decoder.parameters() if parameter.requires_grad
                )
                adapter_parameters = [
                    parameter for parameter in model.decoder.latent_adapter.parameters() if parameter.requires_grad
                ]
                if adapter_parameters:
                    adapter_gradient = all(
                        parameter.grad is not None and torch.isfinite(parameter.grad).all()
                        for parameter in adapter_parameters
                    )
                if not backbone_gradient or not decoder_gradient or not adapter_gradient:
                    raise RuntimeError(
                        f"gradient gate failed: backbone={backbone_gradient}, decoder={decoder_gradient}, "
                        f"adapter={adapter_gradient}"
                    )

            _cuda_sync(device)
            step_start = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(
                [parameter for group in groups for parameter in group["params"]],
                float(config.get("gradient_clip", 1.0)),
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            _cuda_sync(device)
            optimizer_step_seconds = time.perf_counter() - step_start

            raw_loss = xyz_loss_value + float(config.get("lambda_visibility", 0.0)) * visibility_loss_value
            mean_epe = _train_metric(prediction, batch, stats)
            timing = {
                "geometry_worker_sum_seconds": batch.geometry_seconds_sum,
                "geometry_worker_max_seconds": batch.geometry_seconds_max,
                "prefetch_wait_seconds": float(prefetch_wait),
                "prefetch_submit_seconds": float(prefetch_submit),
                "h2d_seconds": float(h2d_seconds),
                "forward_backward_seconds": float(forward_backward_seconds),
                "optimizer_step_seconds": float(optimizer_step_seconds),
                "step_seconds": float(time.perf_counter() - iteration_start),
                "cuda_allocated_gib": float(torch.cuda.memory_allocated(device) / (1024 ** 3)),
                "cuda_reserved_gib": float(torch.cuda.memory_reserved(device) / (1024 ** 3)),
                "gpu_utilization_pct": gpu_sampler.mean_and_reset(),
            }
            timings.append(timing)
            losses.append(raw_loss)
            train_epes.append(mean_epe)
            event = {
                "step": step + 1, "global_step": step + 1, "loss": raw_loss,
                "xyz_loss": xyz_loss_value,
                "visibility_loss": visibility_loss_value,
                "train_epe": mean_epe,
                "sample_indices": plan.sample_indices.tolist(),
                "source": plan.source[:, 0].tolist(),
                "target_shape": list(plan.target.shape),
                "z4d_shape": [logical_batch, *z4d_shape],
                "latent_adapter": adapter,
                "timing": timing,
            }
            if step == 0 or (step + 1) % int(config.get("log_every", 10)) == 0:
                print(json.dumps(event), flush=True)
                if wandb_run is not None:
                    wandb_run.log({
                        "global_step": step + 1,
                        "train/loss": raw_loss, "train/xyz_loss": xyz_loss_value,
                        "train/visibility_loss": visibility_loss_value, "train/epe_m": mean_epe,
                        "train/passes": (step + 1) * int(config["batch_size"]) / len(samples),
                        "timing/geometry_worker_sum_s": timing["geometry_worker_sum_seconds"],
                        "timing/geometry_worker_max_s": timing["geometry_worker_max_seconds"],
                        "timing/prefetch_wait_s": timing["prefetch_wait_seconds"],
                        "timing/h2d_s": timing["h2d_seconds"],
                        "timing/forward_backward_s": timing["forward_backward_seconds"],
                        "timing/optimizer_step_s": timing["optimizer_step_seconds"],
                        "timing/cuda_allocated_gib": timing["cuda_allocated_gib"],
                        "timing/cuda_reserved_gib": timing["cuda_reserved_gib"],
                        "system/gpu_utilization_pct": timing["gpu_utilization_pct"],
                    }, step=step + 1)

    # Persist the trained weights before any evaluation code runs.  If a
    # diagnostics bug occurs, the completed 1,000-update screen is recoverable
    # without repeating optimization.
    prevalidation_checkpoint = save_checkpoint(
        output_dir / "checkpoint.pt", model, config, stats.mean, stats.scale,
        extra={
            "steps": int(config["steps"]), "total_steps": int(config["steps"]), "seed": seed,
            "protocol": "h004_source_centric_ablation_screen", "validation_pending": True,
            "ablation_arm": arm, "latent_adapter": adapter,
        },
    )
    print(f"TRAIN_WEIGHTS_SAVED: {prevalidation_checkpoint}", flush=True)
    validation = evaluate_fixed_validation(
        model, validation_samples, validation_latents, stats, config, device, dtype,
    )
    print(json.dumps({"event": "source_centric_validation", **validation}), flush=True)
    checkpoint = save_checkpoint(
        output_dir / "checkpoint.pt", model, config, stats.mean, stats.scale,
        extra={
            "steps": int(config["steps"]), "total_steps": int(config["steps"]), "seed": seed,
            "protocol": "h004_source_centric_ablation_screen",
            "mode": args.mode, "visibility_pos_weight": config.get("visibility_pos_weight"),
            "ablation_arm": arm, "latent_adapter": adapter,
            "validation": validation,
            "timing_summary": {
                key: float(np.mean([row[key] for row in timings if row[key] is not None]))
                for key in timings[0] if timings and all(row[key] is not None for row in timings)
            },
        },
    )
    print(f"CHECKPOINT_CREATED: {checkpoint}", flush=True)
    loaded = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    checkpoint_load_ok = "decoder.source_embedding.weight" in loaded["model"] and loaded["extra"]["protocol"].startswith("h004_source_centric")
    del loaded
    result = {
        "protocol": "h004_source_centric_ablation_screen", "mode": args.mode,
        "ablation_arm": arm, "latent_adapter": adapter,
        "seed": seed, "steps": int(config["steps"]), "clips": len(samples),
        "batch_size": int(config["batch_size"]),
        "microbatch_size": int(config.get("microbatch_size", config["batch_size"])),
        "trainable_parameters": optimizer_trainable_count(groups),
        "initial_train_loss": losses[0], "final_train_loss": losses[-1],
        "initial_train_epe": train_epes[0], "final_train_epe": train_epes[-1],
        "backbone_gradient": backbone_gradient, "decoder_gradient": decoder_gradient,
        "adapter_gradient": adapter_gradient,
        "validation": validation, "checkpoint": str(checkpoint), "checkpoint_load_ok": checkpoint_load_ok,
        "visibility_pos_weight": config.get("visibility_pos_weight"),
        "timing_mean": {
            key: float(np.mean([row[key] for row in timings if row[key] is not None]))
            for key in timings[0] if timings and all(row[key] is not None for row in timings)
        },
        "timing_last": timings[-1],
        "peak_cuda_allocated_gib": float(torch.cuda.max_memory_allocated(device) / (1024 ** 3)),
        "peak_cuda_reserved_gib": float(torch.cuda.max_memory_reserved(device) / (1024 ** 3)),
        "environment": {"torch": torch.__version__, "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES")},
    }
    (output_dir / "train_metrics.json").write_text(json.dumps(result, indent=2))
    if wandb_run is not None:
        wandb_run.log({
            "global_step": int(config["steps"]),
            "final/off_diagonal_all_valid_epe": validation["metrics"]["off_diagonal_all_valid"]["epe"],
            "final/diagonal_epe": validation["metrics"]["diagonal"]["epe"],
            "final/source_zero_epe": validation["metrics"]["source_zero"]["epe"],
            "final/source_positive_epe": validation["metrics"]["source_positive"]["epe"],
            "final/visible_epe": validation["metrics"]["off_diagonal_all_valid_visible"]["epe"],
            "final/occluded_valid_epe": validation["metrics"]["off_diagonal_all_valid_occluded_valid"]["epe"],
            "final/peak_cuda_allocated_gib": result["peak_cuda_allocated_gib"],
            "final/peak_cuda_reserved_gib": result["peak_cuda_reserved_gib"],
        }, step=int(config["steps"]))
        wandb_run.summary.update({
            "checkpoint": str(checkpoint), "checkpoint_load_ok": checkpoint_load_ok,
            "ablation_arm": arm, "latent_adapter": adapter,
        })
        wandb_run.finish()
    print(json.dumps(result, indent=2), flush=True)
    print("DENSE4D_SOURCE_CENTRIC_TRAIN_OK", flush=True)


if __name__ == "__main__":
    main()
