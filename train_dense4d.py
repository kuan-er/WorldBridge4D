#!/usr/bin/env python3
"""Tiny/bounded end-to-end dense 4D training with masked XYZ loss."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import pickle
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import yaml

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from worldbridge.data import MOViFDataset
from worldbridge.dense4d import masked_pair_smooth_l1
from worldbridge.dense4d_data import (
    CoordinateStats, DynamicPointmapCache, dense_pair_targets, parse_fixed_pairs,
    sample_dense_pairs, sample_h001_balanced_pairs, sample_source_all_targets_pairs,
)
from worldbridge.dense4d_runtime import (
    build_real_model, capture_rng_state, encode_clean_video_latents, optimizer_trainable_count,
    apply_linear_warmup, parameter_groups, precision_dtype, restore_rng_state, save_checkpoint,
)


REQUIRED_CONFIG = {
    "data_root", "wan_root", "clip_length", "image_size", "num_query_pairs", "query_dim",
    "num_cross_attn_layers", "num_heads", "upsample_channels", "batch_size", "learning_rate",
    "precision", "gradient_checkpointing", "gradient_accumulation", "trainable_mode", "seed",
}


def init_wandb(config: dict[str, Any]):
    """Initialize explicit scalar tracking; PRL only injects W&B metadata."""
    tracking = config.get("tracking", {})
    mode = os.getenv("WANDB_MODE", tracking.get("mode", "online"))
    if not bool(tracking.get("enabled", False)) or mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("tracking.enabled=true but wandb is not installed") from exc
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", tracking.get("project", "worldbridge4d")),
        entity=os.getenv("WANDB_ENTITY", tracking.get("entity")),
        group=os.getenv("WANDB_GROUP", tracking.get("group", "h004-dense4d-long-train")),
        job_type="train",
        name=os.getenv("WANDB_NAME", f"h004-dense4d-{os.getenv('PRL_RUN_ID', 'local')}"),
        tags=list(tracking.get("tags", [])) + ["h004", "dense4d", "movi-f"],
        config=config,
        mode=mode,
        reinit="return_previous",
    )
    run.define_metric("global_step")
    for namespace in ("train/*", "timing/*", "system/*", "final/*"):
        run.define_metric(namespace, step_metric="global_step")
    print(f"WANDB_RUN_URL: {run.url}", flush=True)
    return run


def _cache_metadata(config: dict[str, Any], count: int) -> dict[str, Any]:
    return {
        "format": 1,
        "data_root": str(pathlib.Path(config["data_root"]).resolve()),
        "split": "train",
        "clip_length": int(config["clip_length"]),
        "clip_start": config.get("clip_start", 0),
        "seed": int(config["seed"]),
        "count": int(count),
    }


def load_or_create_samples(dataset: MOViFDataset, config: dict[str, Any]) -> list[Any]:
    cache_name = config.get("sample_cache")
    if not cache_name:
        return [dataset[index] for index in range(len(dataset))]
    path = pathlib.Path(cache_name)
    metadata = _cache_metadata(config, len(dataset))
    if path.exists():
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("metadata") != metadata:
            raise RuntimeError(f"sample cache metadata mismatch: {path}; remove it to rebuild")
        samples = payload["samples"]
        if len(samples) != len(dataset):
            raise RuntimeError(f"sample cache length mismatch: {path}")
        print(f"SAMPLE_CACHE_HIT: {path} ({len(samples)} clips)", flush=True)
        return samples
    samples = [dataset[index] for index in range(len(dataset))]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump({"metadata": metadata, "samples": samples}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    print(f"SAMPLE_CACHE_CREATED: {path} ({len(samples)} clips)", flush=True)
    return samples


def load_or_create_clean_latents(samples: list[Any], config: dict[str, Any], device: torch.device) -> list[torch.Tensor]:
    cache_name = config.get("clean_latent_cache")
    if not cache_name:
        return encode_clean_video_latents(samples, config["wan_root"], device)
    path = pathlib.Path(cache_name)
    metadata = {
        **_cache_metadata(config, len(samples)),
        "wan_root": str(pathlib.Path(config["wan_root"]).resolve()),
        "dtype": "float32",
    }
    if path.exists():
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        if payload.get("metadata") != metadata:
            raise RuntimeError(f"clean latent cache metadata mismatch: {path}; remove it to rebuild")
        latents = payload["latents"]
        if tuple(latents.shape) != (len(samples), 16, 6, 16, 16):
            raise RuntimeError(f"clean latent cache shape mismatch: {tuple(latents.shape)}")
        print(f"CLEAN_LATENT_CACHE_HIT: {path} ({len(samples)} clips)", flush=True)
        return [latents[index:index + 1] for index in range(len(samples))]
    encoded = encode_clean_video_latents(samples, config["wan_root"], device)
    stacked = torch.cat(encoded, dim=0).contiguous().float().cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"metadata": metadata, "latents": stacked}, temporary)
    temporary.replace(path)
    print(f"CLEAN_LATENT_CACHE_CREATED: {path} ({len(samples)} clips)", flush=True)
    return [stacked[index:index + 1] for index in range(len(samples))]


def pair_suite(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    values = config.get("evaluation_pairs") or [
        [0, 0], [7, 7], [20, 20], [0, 1], [0, 20], [7, 14], [20, 0], [14, 7], [20, 19], [3, 18],
    ]
    return parse_fixed_pairs(values, int(config["clip_length"]))


def grouped_eval(model, latent, sample, source, target, stats, cache, device, dtype,
                 coordinate_frame="anchor"):
    normalized, metric, visible, valid = dense_pair_targets(
        sample, source, target, stats, cache, coordinate_frame=coordinate_frame,
    )
    source_tensor = torch.from_numpy(source)[None].to(device)
    target_tensor = torch.from_numpy(target)[None].to(device)
    target_tensor_xyz = torch.from_numpy(normalized)[None].to(device)
    valid_tensor = torch.from_numpy(valid)[None].to(device)
    was_training = model.training
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype == torch.bfloat16):
        prediction, z4d, _ = model(latent.to(device=device, dtype=dtype), source_tensor, target_tensor)
    prediction_float = prediction.float()
    prediction_metric = (
        prediction_float.cpu().numpy()[0] * stats.scale[None, :, None, None]
        + stats.mean[None, :, None, None]
    )
    errors = np.linalg.norm(prediction_metric - metric, axis=1)
    groups = {
        "all": np.ones(len(source), dtype=bool),
        "diagonal": source == target,
        "off_diagonal": source != target,
        "forward": target > source,
        "backward": target < source,
        "source_zero": source == 0,
        "source_gt_zero": source > 0,
    }
    result = {
        "normalized_smooth_l1": float(masked_pair_smooth_l1(
            prediction_float, target_tensor_xyz.float(), valid_tensor
        )),
        "z4d_shape": list(z4d.shape),
        "z4d_motion_shape": list(z4d.motion.shape) if hasattr(z4d, "motion") else None,
    }
    for name, select in groups.items():
        mask = valid[select]
        result[f"{name}_epe"] = float(errors[select][mask].mean()) if np.any(mask) else None
    if was_training:
        model.train()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", help="resume from an exact Dense4D training checkpoint")
    parser.add_argument("--resume-weights-only", action="store_true",
                        help="initialize from model weights without optimizer/RNG resume state")
    args = parser.parse_args()
    if args.resume_weights_only and not args.resume:
        parser.error("--resume-weights-only requires --resume")
    config = yaml.safe_load(pathlib.Path(args.config).read_text())
    missing = sorted(REQUIRED_CONFIG - set(config))
    if missing:
        raise ValueError(f"missing required config keys: {missing}")
    if int(config["image_size"]) != 128 or int(config["clip_length"]) != 21:
        raise ValueError("dense 4D training is intentionally fixed to 21 frames and 128x128")
    coordinate_frame = str(config.get("coordinate_frame", "anchor")).lower()
    if coordinate_frame not in {"anchor", "source"}:
        raise ValueError("coordinate_frame must be 'anchor' or 'source'")
    config["coordinate_frame"] = coordinate_frame

    seed = int(config["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("real Wan training requires CUDA")
    torch.cuda.set_device(0 if device.index is None else device.index)
    torch.cuda.reset_peak_memory_stats(device)
    dtype = precision_dtype(config["precision"])
    resume_step_offset = 0
    resume_optimizer_updates = 0
    resume_payload = None
    if args.resume:
        resume_payload = torch.load(args.resume, map_location="cpu", mmap=True, weights_only=True)
        resume_extra = resume_payload.get("extra", {})
        resume_step_offset = int(resume_extra.get("total_steps", resume_extra.get("steps", 0)))
        if not args.resume_weights_only and (
            "optimizer" not in resume_payload or "training_state" not in resume_payload
        ):
            raise RuntimeError(
                "exact resume requires optimizer and training_state; use --resume-weights-only "
                "for a legacy model-only checkpoint"
            )
        if not args.resume_weights_only:
            resume_optimizer_updates = int(resume_payload["training_state"]["optimizer_updates"])
            if int(resume_payload["training_state"]["global_step"]) != resume_step_offset:
                raise RuntimeError("checkpoint extra/training_state global_step mismatch")
            mutable_resume_keys = {
                "steps", "checkpoint_every_steps", "save_checkpoint", "keep_checkpoint", "tracking",
            }
            checkpoint_config = resume_payload.get("config", {})
            mismatched = sorted(
                key for key in set(checkpoint_config) | set(config)
                if key not in mutable_resume_keys and checkpoint_config.get(key) != config.get(key)
            )
            if mismatched:
                raise RuntimeError(f"exact resume config mismatch for keys: {mismatched}")
    rng = np.random.default_rng(seed)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = init_wandb(config)

    max_examples = config.get("max_clips", 1)
    max_examples = None if max_examples is None else int(max_examples)
    dataset = MOViFDataset(
        config["data_root"], split="train", clip_length=int(config["clip_length"]),
        clip_start=int(config.get("clip_start", 0)), max_examples=max_examples, seed=seed,
    )
    samples = load_or_create_samples(dataset, config)
    stats = CoordinateStats.from_npz(config["coordinate_stats"])
    if args.resume and not args.resume_weights_only:
        checkpoint_mean = np.asarray(resume_payload["coordinate_mean"], dtype=np.float32)
        checkpoint_scale = np.asarray(resume_payload["coordinate_scale"], dtype=np.float32)
        if not np.array_equal(checkpoint_mean, stats.mean) or not np.array_equal(checkpoint_scale, stats.scale):
            raise RuntimeError("exact resume coordinate statistics differ from the checkpoint")
    clean_latents = load_or_create_clean_latents(samples, config, device)
    model = build_real_model(config, device)
    if args.resume:
        model.load_state_dict(resume_payload["model"], strict=True)
    groups = parameter_groups(model, config)
    optimizer = torch.optim.AdamW(groups, weight_decay=float(config.get("weight_decay", 0.0)))
    if args.resume and not args.resume_weights_only:
        optimizer.load_state_dict(resume_payload["optimizer"])
    if args.resume and not args.resume_weights_only:
        restore_rng_state(resume_payload["training_state"]["rng_state"], rng)
        print(f"RESUME_CHECKPOINT_LOADED: {args.resume} (global_step={resume_step_offset})", flush=True)
    elif args.resume:
        resume_step_offset = 0
        print(f"RESUME_WEIGHTS_ONLY_LOADED: {args.resume} (starting global_step=0)", flush=True)
    del resume_payload
    cache = DynamicPointmapCache(
        max_entries=int(config.get("geometry_cache_entries", 32)),
        depth_tolerance=float(config.get("depth_tolerance", 0.05)),
        depth_relative_tolerance=float(config.get("depth_relative_tolerance", 0.01)),
        compute_visibility=False,
    )

    evaluation_source, evaluation_target = pair_suite(config)
    initial_eval = grouped_eval(
        model, clean_latents[0], samples[0], evaluation_source, evaluation_target,
        stats, cache, device, dtype, coordinate_frame,
    )
    print(json.dumps({"event": "initial_evaluation", **initial_eval}), flush=True)
    if wandb_run is not None:
        wandb_run.log({
            f"initial_eval/{key}": value for key, value in initial_eval.items()
            if isinstance(value, (int, float)) and value is not None
        }, step=resume_step_offset)

    steps = int(config["steps"])
    if steps < 1:
        raise ValueError("config steps must be positive")
    if resume_step_offset >= steps:
        raise ValueError(
            f"checkpoint global_step={resume_step_offset} already reaches config steps={steps}; "
            "increase config steps or start a new run"
        )
    steps_to_run = steps - resume_step_offset
    checkpoint_every = int(config.get("checkpoint_every_steps", 0))
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every_steps must be non-negative")
    batch_size = int(config["batch_size"])
    num_pairs = int(config["num_query_pairs"])
    accumulation = int(config["gradient_accumulation"])
    if accumulation < 1:
        raise ValueError("gradient_accumulation must be positive")
    fixed = config.get("fixed_pairs")
    fixed_source, fixed_target = parse_fixed_pairs(
        fixed, int(config["clip_length"]), num_pairs
    ) if fixed is not None else (None, None)
    warmup_steps = int(config.get("warmup_steps", 0))
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    total_losses: list[float] = []
    gate_entropies: list[float] = []
    epe_values: list[float] = []
    backbone_gradient = False
    wan_gradient = False
    geometry_adapter_gradient = False
    decoder_gradient = False
    start_time = time.time()
    model.train()

    optimizer_updates = resume_optimizer_updates
    for step in range(steps_to_run):
        global_step = resume_step_offset + step
        warmup_factor = apply_linear_warmup(optimizer, optimizer_updates + 1, warmup_steps)
        if bool(config.get("matched_epoch_sampling", False)):
            # One deterministic pass is exactly len(samples) optimizer updates;
            # this makes steps=N*epochs comparable to the prior Kubric runs.
            start = (global_step * batch_size) % len(samples)
            sample_indices = (start + np.arange(batch_size, dtype=np.int64)) % len(samples)
        elif bool(config.get("fixed_clip_order", False)):
            sample_indices = np.arange(batch_size, dtype=np.int64) % len(samples)
        else:
            sample_indices = rng.integers(len(samples), size=batch_size)
        source_rows, target_rows, target_rows_xyz, valid_rows = [], [], [], []
        metric_rows = []
        for sample_index in sample_indices:
            if fixed_source is None:
                sampler = str(config.get("pair_sampling", "dense_balanced"))
                if sampler == "h001_balanced":
                    source, target = sample_h001_balanced_pairs(
                        int(config["clip_length"]), num_pairs, rng
                    )
                elif sampler == "source_all_targets":
                    source, target = sample_source_all_targets_pairs(
                        int(config["clip_length"]), num_pairs, rng
                    )
                elif sampler == "dense_balanced":
                    source, target = sample_dense_pairs(int(config["clip_length"]), num_pairs, rng)
                else:
                    raise ValueError(f"unknown pair_sampling={sampler!r}")
            else:
                source, target = fixed_source.copy(), fixed_target.copy()
            normalized, metric, _, valid = dense_pair_targets(
                samples[int(sample_index)], source, target, stats, cache,
                coordinate_frame=coordinate_frame,
            )
            source_rows.append(source); target_rows.append(target)
            target_rows_xyz.append(normalized); metric_rows.append(metric); valid_rows.append(valid)
        clean = torch.cat([clean_latents[int(index)] for index in sample_indices]).to(device=device, dtype=dtype)
        source_tensor = torch.from_numpy(np.stack(source_rows)).to(device)
        target_tensor = torch.from_numpy(np.stack(target_rows)).to(device)
        target_xyz = torch.from_numpy(np.stack(target_rows_xyz)).to(device)
        valid_tensor = torch.from_numpy(np.stack(valid_rows)).to(device)

        with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype == torch.bfloat16):
            prediction, z4d, _ = model(clean, source_tensor, target_tensor)
            geometry_loss = masked_pair_smooth_l1(
                prediction.float(), target_xyz.float(), valid_tensor,
                beta=float(config.get("smooth_l1_beta", 0.05)),
            )
            entropy_fn = getattr(model.backbone, "layer_gate_entropy", None)
            gate_entropy = entropy_fn() if entropy_fn is not None else geometry_loss.new_zeros(())
            loss = geometry_loss + float(config.get("layer_gate_entropy_weight", 0.0)) * gate_entropy
        (loss / accumulation).backward()
        if step == 0:
            backbone_gradient = any(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                    for parameter in model.backbone.parameters() if parameter.requires_grad)
            mapping = getattr(model.backbone, "mapping", None)
            wan_gradient = mapping is not None and any(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in mapping.parameters() if parameter.requires_grad
            )
            geometry_adapter_gradient = any(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in getattr(model.backbone, "adapter_parameters", []) if parameter.requires_grad
            )
            decoder_gradient = any(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                   for parameter in model.decoder.parameters() if parameter.requires_grad)
            trainable_mode = str(config.get("trainable_mode", "full"))
            if trainable_mode == "full" and mapping is not None and not wan_gradient:
                raise RuntimeError("XYZ loss did not reach a trainable Wan DiT parameter")
            if trainable_mode == "full" and mapping is None and not backbone_gradient:
                raise RuntimeError("XYZ loss did not reach the trainable backbone")
            if trainable_mode == "geometry_adapter" and not geometry_adapter_gradient:
                raise RuntimeError("XYZ loss did not reach the geometry adapter")
            if not decoder_gradient:
                raise RuntimeError("XYZ loss did not reach decoder parameters")
        did_optimizer_step = (step + 1) % accumulation == 0 or step + 1 == steps_to_run
        if did_optimizer_step:
            torch.nn.utils.clip_grad_norm_(
                [parameter for group in groups for parameter in group["params"]],
                float(config.get("gradient_clip", 1.0)),
            )
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            optimizer_updates += 1

        raw_loss = float(geometry_loss.detach())
        total_loss = float(loss.detach())
        gate_entropy_value = float(gate_entropy.detach())
        losses.append(raw_loss)
        total_losses.append(total_loss)
        gate_entropies.append(gate_entropy_value)
        prediction_metric = (
            prediction.detach().float().cpu().numpy() * stats.scale[None, None, :, None, None]
            + stats.mean[None, None, :, None, None]
        )
        metric = np.stack(metric_rows)
        valid = np.stack(valid_rows)
        epe = np.linalg.norm(prediction_metric - metric, axis=2)
        mean_epe = float(epe[valid].mean())
        epe_values.append(mean_epe)
        print(json.dumps({
            "step": global_step + 1, "loss": raw_loss, "total_loss": total_loss,
            "warmup_factor": warmup_factor,
            "layer_gate_entropy": gate_entropy_value, "train_epe": mean_epe,
            "pairs": np.stack((source_rows[0], target_rows[0]), axis=-1).tolist(),
            "z4d_shape": list(z4d.shape),
        }), flush=True)
        wandb_log_every = int(config.get("tracking", {}).get("log_every", 10))
        if wandb_run is not None and (step == 0 or (global_step + 1) % wandb_log_every == 0):
            wandb_run.log({
                "train/loss": raw_loss,
                "train/total_loss": total_loss,
                "train/warmup_factor": warmup_factor,
                "train/layer_gate_entropy": gate_entropy_value,
                "train/epe_m": mean_epe,
                "train/clips_seen": (global_step + 1) * batch_size,
                "train/passes": (global_step + 1) * batch_size / len(samples),
                "train/backbone_lr": float(next(
                    (group["lr"] for group in groups if group["name"] == "wan_backbone"),
                    next(group["lr"] for group in groups if group["name"] in {"geometry_adapter", "dense_decoder"}),
                )),
                "system/peak_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
            }, step=global_step + 1)
        if (did_optimizer_step and checkpoint_every and global_step + 1 < steps
                and (global_step + 1) % checkpoint_every == 0):
            periodic_path = save_checkpoint(
                output_dir / "checkpoint.pt", model, config, stats.mean, stats.scale,
                extra={"steps": steps, "total_steps": global_step + 1, "seed": seed,
                       "checkpoint_kind": "periodic", "initial_eval": initial_eval},
                optimizer=optimizer,
                training_state={
                    "global_step": global_step + 1,
                    "optimizer_updates": optimizer_updates,
                    "rng_state": capture_rng_state(rng),
                },
            )
            print(f"CHECKPOINT_PERIODIC: {periodic_path} (global_step={global_step + 1})", flush=True)

    total_steps = steps
    checkpoint = None
    checkpoint_load_ok = False
    if bool(config.get("save_checkpoint", True)):
        # Persist the exact optimizer boundary before evaluation. A diagnostics
        # failure must not discard a completed expensive training run.
        checkpoint = save_checkpoint(
            output_dir / "checkpoint.pt", model, config, stats.mean, stats.scale,
            extra={"steps": steps, "total_steps": total_steps, "seed": seed,
                   "initial_eval": initial_eval, "checkpoint_kind": "trained_pre_evaluation"},
            optimizer=optimizer,
            training_state={
                "global_step": total_steps,
                "optimizer_updates": optimizer_updates,
                "rng_state": capture_rng_state(rng),
            },
        )
        print(f"TRAIN_STATE_SAVED: {checkpoint}", flush=True)
        loaded = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
        readout = str(config.get("backbone_readout", "wan_velocity"))
        backbone_payload_ok = (
            readout == "clean_latent"
            or (readout == "wan_hidden_structured" and "backbone.temporal_logits" in loaded["model"])
            or (readout == "wan_velocity" and "backbone.mapping.dit.proj_out.weight" in loaded["model"])
        )
        checkpoint_load_ok = (
            loaded["extra"]["steps"] == steps
            and loaded["extra"]["total_steps"] == total_steps
            and "optimizer" in loaded
            and "training_state" in loaded
            and "decoder.source_embedding.weight" in loaded["model"]
            and backbone_payload_ok
        )
        del loaded

    final_eval = grouped_eval(
        model, clean_latents[0], samples[0], evaluation_source, evaluation_target,
        stats, cache, device, dtype, coordinate_frame,
    )
    if checkpoint is not None:
        print(f"CHECKPOINT_CREATED: {checkpoint}", flush=True)
        if not bool(config.get("keep_checkpoint", True)):
            checkpoint.unlink()
            checkpoint = None

    result = {
        "seed": seed, "steps": steps, "run_steps": steps_to_run, "total_steps": total_steps,
        "optimizer_updates": optimizer_updates, "warmup_steps": warmup_steps,
        "checkpoint_every_steps": checkpoint_every,
        "initial_global_step": resume_step_offset, "clips": len(samples), "batch_size": batch_size,
        "num_query_pairs": num_pairs, "trainable_mode": config.get("trainable_mode", "full"),
        "trainable_parameters": optimizer_trainable_count(groups),
        "initial_train_loss": losses[0], "final_train_loss": losses[-1],
        "train_loss_ratio": losses[-1] / losses[0],
        "initial_total_loss": total_losses[0], "final_total_loss": total_losses[-1],
        "initial_layer_gate_entropy": gate_entropies[0], "final_layer_gate_entropy": gate_entropies[-1],
        "final_layer_weights": getattr(model.backbone, "soft_layer_weights", lambda: torch.empty(0))().detach().cpu().tolist(),
        "initial_train_epe": epe_values[0], "final_train_epe": epe_values[-1],
        "initial_evaluation": initial_eval, "final_evaluation": final_eval,
        "evaluation_loss_ratio": final_eval["normalized_smooth_l1"] / initial_eval["normalized_smooth_l1"],
        "backbone_gradient": backbone_gradient, "wan_gradient": wan_gradient,
        "geometry_adapter_gradient": geometry_adapter_gradient, "decoder_gradient": decoder_gradient,
        "clean_latent_shape": list(clean_latents[0].shape), "z4d_shape": final_eval["z4d_shape"],
        "coordinate_mean": stats.mean.tolist(), "coordinate_scale": stats.scale.tolist(),
        "elapsed_seconds": time.time() - start_time,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        "checkpoint": str(checkpoint) if checkpoint else None, "checkpoint_load_ok": checkpoint_load_ok,
        "wandb_url": wandb_run.url if wandb_run is not None else None,
        "environment": {"torch": torch.__version__, "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES")},
    }
    (output_dir / "train_metrics.json").write_text(json.dumps(result, indent=2))
    if wandb_run is not None:
        final_scalars = {
            f"final_eval/{key}": value for key, value in final_eval.items()
            if isinstance(value, (int, float)) and value is not None
        }
        final_scalars.update({
            "final/train_loss": losses[-1],
            "final/total_loss": total_losses[-1],
            "final/layer_gate_entropy": gate_entropies[-1],
            "final/train_epe_m": epe_values[-1],
            "final/elapsed_seconds": result["elapsed_seconds"],
            "final/peak_cuda_memory_gib": result["peak_cuda_memory_gib"],
        })
        wandb_run.log(final_scalars, step=total_steps)
        wandb_run.summary.update(final_scalars)
        wandb_run.summary["checkpoint"] = result["checkpoint"]
        wandb_run.summary["checkpoint_load_ok"] = checkpoint_load_ok
        wandb_run.finish()
    print(json.dumps(result, indent=2), flush=True)
    print("DENSE4D_TRAIN_OK", flush=True)


if __name__ == "__main__":
    main()
