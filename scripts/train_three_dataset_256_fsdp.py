#!/usr/bin/env python3
"""Four-rank FULL_SHARD training for the 256px/200M three-dataset route."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from contextlib import nullcontext
import datetime as dt
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullOptimStateDictConfig, FullStateDictConfig, FullyShardedDataParallel as FSDP,
    MixedPrecision, ShardingStrategy, StateDictType,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from worldbridge.dense4d import masked_pair_smooth_l1
from worldbridge.dense4d_runtime import (
    build_real_model, capture_rng_state, parameter_groups,
    precision_dtype, restore_rng_state,
)
from worldbridge.training256 import (
    DATASET_NAMES, apply_cosine_schedule, dataset_for_step,
    deterministic_sample_plan, load_training_datasets, prepare_training_indexes,
    sample_eligible_targets, source_with_eligible_targets,
)
from worldbridge.wan import WAN_LATENT_SHAPE_256, WanVAEEncoder
from worldbridge.text_conditions import load_dataset_text_conditions

_STOP = False


def stop_signal(_signum: int, _frame: Any) -> None:
    global _STOP
    _STOP = True


def initialize_distributed() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    if not torch.cuda.is_available():
        raise RuntimeError("256px distributed training requires CUDA")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group(
        "nccl", init_method="env://", timeout=dt.timedelta(hours=24), device_id=device,
    )
    return rank, world, local, device


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def validate_config(config: dict[str, Any], world: int, allow_two_gpu: bool) -> None:
    expected = {
        "image_size": 256, "clip_length": 21, "latent_spatial_size": 32,
        "query_dim": 1536, "embedding_dim": 768, "num_cross_attn_layers": 5,
        "num_heads": 12, "geometry_dim": 512, "geometry_spatial_size": 32,
        "motion_slots": 8, "gradient_accumulation": 2, "microbatch_per_gpu": 1,
    }
    mismatches = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if mismatches:
        raise ValueError(f"256/200M frozen configuration mismatch: {mismatches}")
    if list(config.get("wan_hidden_layers", [])) != [13, 14, 15, 29]:
        raise ValueError("wan_hidden_layers must be [13,14,15,29]")
    logits = np.asarray(config.get("layer_gate_initial_logits"), dtype=np.float64)
    expected_logits = np.array([0.0, 0.0, 0.0, -1.0986122887])
    if logits.shape != (4,) or not np.allclose(logits, expected_logits, atol=1e-10):
        raise ValueError(f"incorrect readout initialization: {logits}")
    weights = np.exp(logits - logits.max()); weights /= weights.sum()
    if not np.allclose(weights, [0.3, 0.3, 0.3, 0.1], atol=1e-8):
        raise ValueError(f"incorrect initial layer weights: {weights}")
    if world != 4 and not (allow_two_gpu and world == 2):
        raise ValueError(f"formal training requires 4 ranks; got {world} (use --allow-two-gpu-gate only for the gate)")
    k = int(config["targets_per_source"])
    if k not in (4, 6):
        raise ValueError("targets_per_source must be the gated K=6 or K=4")


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while value := stream.read(chunk):
            digest.update(value)
    return digest.hexdigest()


def required_latent_indices(datasets: dict[str, Any], seed: int, start_step: int,
                            target_steps: int, rank: int, accumulation: int
                            ) -> dict[str, list[int]]:
    required: dict[str, set[int]] = {name: set() for name in DATASET_NAMES}
    for step in range(int(start_step), int(target_steps)):
        name = dataset_for_step(step, seed)
        dataset = datasets[name]
        for micro in range(int(accumulation)):
            index, _, _ = deterministic_sample_plan(
                dataset, name, seed, step, micro, rank, accumulation
            )
            required[name].add(int(index))
    return {name: sorted(indices) for name, indices in required.items()}


def warm_lazy_latents(config: dict[str, Any], datasets: dict[str, Any],
                      required: dict[str, list[int]], device: torch.device,
                      rank: int) -> dict[str, int]:
    """Populate only this rank's planned cache misses, then release the VAE."""
    checkpoint = Path(config.get(
        "vae_checkpoint", Path(config["wan_root"]) / "Wan2.1_VAE.pth"
    )).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"WAN VAE checkpoint missing: {checkpoint}")
    checksum = sha256(checkpoint)
    for dataset in datasets.values():
        dataset.set_lazy_vae_sha256(checksum)
    missing = [(name, index) for name in DATASET_NAMES for index in required[name]
               if not datasets[name].latent_cached(index)]
    counts = {"required": sum(len(x) for x in required.values()), "misses": len(missing),
              "written": 0, "reused_after_wait": 0}
    if not missing:
        print(json.dumps({"event": "lazy_vae_warmup", "rank": rank, **counts}), flush=True)
        return counts
    # FP32 is the canonical offline contract. The VAE is deleted before the
    # trainable 1.3B model/FSDP/optimizer are constructed, avoiding co-residency.
    encoder = WanVAEEncoder(
        checkpoint, device=device, dtype=torch.float32,
        expected_shape=WAN_LATENT_SHAPE_256,
    )
    for name, index in missing:
        dataset = datasets[name]
        if dataset.latent_cached(index):
            counts["reused_after_wait"] += 1
            continue
        rgb = torch.from_numpy(np.asarray(dataset.rgb(index))).permute(0, 3, 1, 2)[None]
        with torch.inference_mode():
            value = encoder(rgb).float().cpu().numpy()[0]
        if dataset.cache_latent(index, value, checksum):
            counts["written"] += 1
        else:
            counts["reused_after_wait"] += 1
        print(json.dumps({
            "event": "lazy_vae_clip", "rank": rank, "dataset": name,
            "index": index, "written": counts["written"], "misses": len(missing),
        }), flush=True)
    del encoder
    torch.cuda.empty_cache()
    print(json.dumps({"event": "lazy_vae_warmup", "rank": rank, **counts}), flush=True)
    return counts


def load_stats(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    path = Path(config["mixture_coordinate_stats"])
    if not path.is_file():
        raise FileNotFoundError(f"train-only mixture coordinate statistics missing: {path}")
    with np.load(path) as values:
        mean = np.asarray(values["mean"], np.float32).reshape(3)
        scale = np.asarray(values["scale"], np.float32).reshape(3)
        if str(values.get("coordinate_frame", "")) != "source":
            raise ValueError("mixture stats must use the source camera frame")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("invalid mixture coordinate statistics")
    return mean, scale


def init_wandb(config: dict[str, Any], output: Path, rank: int, disabled: bool):
    tracking = config.get("tracking", {})
    if rank != 0 or disabled or not tracking.get("enabled", True) or os.environ.get("WANDB_MODE") == "disabled":
        return None
    import wandb
    run_id_path = output / "wandb_run_id"
    run_id = run_id_path.read_text().strip() if run_id_path.exists() else wandb.util.generate_id()
    run_id_path.write_text(run_id + "\n")
    mode = os.environ.get("WANDB_MODE", "online")
    if mode == "online" and not os.environ.get("WANDB_API_KEY") and not (Path.home() / ".netrc").exists():
        mode = "offline"
    run = wandb.init(
        id=run_id, resume="allow", mode=mode, project=tracking.get("project", "worldbridge4d"),
        entity=tracking.get("entity"), group=tracking.get("group"), tags=tracking.get("tags"),
        name=os.environ.get("WANDB_NAME", "worldbridge4d-256-three-dataset-200m"), config=config,
    )
    run.define_metric("global_step")
    run.define_metric("train/*", step_metric="global_step")
    run.define_metric("system/*", step_metric="global_step")
    run.define_metric("sampling/*", step_metric="global_step")
    return run


def fsdp_state_context(model: FSDP):
    return FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    )


def save_checkpoint(path: Path, model: FSDP, optimizer: torch.optim.Optimizer,
                    config: dict[str, Any], training_state: dict[str, Any],
                    mean: np.ndarray, scale: np.ndarray, rank: int, world: int) -> None:
    local_rng = capture_rng_state()
    rng_states: list[Any] | None = [None] * world if rank == 0 else None
    dist.gather_object(local_rng, rng_states, dst=0)
    with fsdp_state_context(model):
        model_state = model.state_dict()
        optimizer_state = FSDP.optim_state_dict(model, optimizer)
    if rank == 0:
        payload = {
            "format": 3, "model": model_state, "optimizer": optimizer_state,
            "config": config, "coordinate_mean": mean.tolist(), "coordinate_scale": scale.tolist(),
            "training_state": {**training_state, "rng_states": rng_states, "world_size": world},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
    dist.barrier()


def load_checkpoint(path: Path, model: FSDP, optimizer: torch.optim.Optimizer,
                    rank: int, world: int) -> dict[str, Any]:
    # Only rank zero reads the potentially multi-GiB full checkpoint. FSDP then
    # synchronizes model parameters and scatters Adam state to FULL_SHARD ranks.
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True) if rank == 0 else None
    metadata = [(
        int(payload["training_state"].get("world_size", world)),
        payload["training_state"], payload["training_state"].get("rng_states", []),
    ) if rank == 0 else None]
    dist.broadcast_object_list(metadata, src=0)
    saved_world, training_state, states = metadata[0]
    if saved_world != world:
        raise ValueError(f"exact resume world-size mismatch: checkpoint={saved_world}, current={world}")
    with fsdp_state_context(model):
        model.load_state_dict(payload["model"] if rank == 0 else {}, strict=(rank == 0))
    full_optimizer_state = payload["optimizer"] if rank == 0 else None
    optimizer_state = FSDP.scatter_full_optim_state_dict(
        full_optimizer_state, model, optim=optimizer,
    )
    optimizer.load_state_dict(optimizer_state)
    if len(states) != world:
        raise ValueError("checkpoint lacks one RNG state per rank")
    restore_rng_state(states[rank])
    return training_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--allow-two-gpu-gate", action="store_true")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--checkpoint-at-end", action="store_true")
    parser.add_argument("--lazy-vae-cache", action="store_true",
                        help="encode/cache only planned missing latents before constructing FSDP")
    args = parser.parse_args()
    for value in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        signal.signal(value, stop_signal)
    config = yaml.safe_load(Path(args.config).read_text())
    rank, world, local, device = initialize_distributed()
    validate_config(config, world, args.allow_two_gpu_gate)
    seed = int(config.get("seed", 20260812))
    random.seed(seed + rank); np.random.seed(seed + rank); torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    output = Path(args.output_dir).resolve()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        prepare_training_indexes(config)
    dist.barrier()
    mean, scale = load_stats(config)
    conditions, prompt_metadata = load_dataset_text_conditions(config)
    datasets = load_training_datasets(config, allow_missing_latents=args.lazy_vae_cache)
    target_steps = int(args.steps if args.steps is not None else config["max_steps"])
    resume = Path(args.resume) if args.resume else (output / "latest.pt")
    # Cache generation needs to know exact post-resume indices before model
    # construction. A tiny rank-0 metadata read avoids loading model/Adam here.
    metadata = [None]
    if rank == 0 and resume.is_file():
        checkpoint_metadata = torch.load(resume, map_location="cpu", mmap=True, weights_only=True)
        metadata[0] = int(checkpoint_metadata["training_state"]["global_step"])
        del checkpoint_metadata
    dist.broadcast_object_list(metadata, src=0)
    planned_start = int(metadata[0] or 0)
    if planned_start >= target_steps:
        raise ValueError(f"checkpoint step {planned_start} already reaches target {target_steps}")
    lazy_counts = None
    if args.lazy_vae_cache:
        local_required = required_latent_indices(
            datasets, seed, planned_start, target_steps, rank,
            int(config["gradient_accumulation"]),
        )
        gathered: list[Any] = [None] * world
        dist.all_gather_object(gathered, local_required)
        # A clip needed by multiple ranks gets one deterministic owner, so the
        # expensive VAE forward is not duplicated. File locks still protect
        # against independent jobs sharing the same persistent cache.
        owned = {name: [] for name in DATASET_NAMES}
        for name in DATASET_NAMES:
            all_indices = sorted({index for item in gathered for index in item[name]})
            for index in all_indices:
                owner = next(owner for owner, item in enumerate(gathered) if index in item[name])
                if owner == rank:
                    owned[name].append(index)
        lazy_counts = warm_lazy_latents(config, datasets, owned, device, rank)
        dist.barrier()
    # Fail closed now: every planned access must resolve before the expensive
    # FSDP model is constructed and ranks enter collectives.
    required_check = required_latent_indices(
        datasets, seed, planned_start, target_steps, rank,
        int(config["gradient_accumulation"]),
    )
    for name, indices in required_check.items():
        for index in indices:
            datasets[name].clean_latent(index)
    if rank == 0:
        print(json.dumps({"event": "three_dataset_cache_ready", "clips": {k: len(v) for k, v in datasets.items()}}), flush=True)

    model = build_real_model(config, device)
    layer_weights = model.backbone.layer_weights().detach().float().cpu().numpy()
    # build_real_model intentionally materializes the BF16 training model before
    # this audit. Compare against the configured logits after the same dtype
    # quantization, rather than impossible exact FP32 probabilities.
    quantized_logits = torch.as_tensor(
        config["layer_gate_initial_logits"], dtype=model.backbone.layer_logits.dtype,
    ).float()
    expected_layer_weights = quantized_logits.softmax(0).cpu().numpy()
    if not np.allclose(layer_weights, expected_layer_weights, atol=1e-7):
        raise RuntimeError(
            f"constructed readout weights changed: {layer_weights} != {expected_layer_weights}"
        )
    adapter_count = sum(p.numel() for p in model.backbone.adapter_parameters)
    decoder_count = sum(p.numel() for p in model.decoder.parameters())
    non_wan_count = adapter_count + decoder_count
    expected_non_wan = int(config.get("expected_non_wan_parameters", 193586693))
    if non_wan_count != expected_non_wan:
        raise RuntimeError(f"non-Wan readout parameters {non_wan_count:,} != expected {expected_non_wan:,}")
    groups = parameter_groups(model, config)
    dtype = precision_dtype(config["precision"])
    auto_wrap = lambda module, recurse, nonwrapped_numel: size_based_auto_wrap_policy(
        module, recurse, nonwrapped_numel, min_num_params=int(config.get("fsdp_min_num_params", 5_000_000))
    )
    fsdp = FSDP(
        model, sharding_strategy=ShardingStrategy.FULL_SHARD, auto_wrap_policy=auto_wrap,
        mixed_precision=MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype),
        device_id=device, sync_module_states=True, use_orig_params=True,
        limit_all_gathers=True, forward_prefetch=False,
    )
    optimizer = torch.optim.AdamW(groups, weight_decay=float(config["weight_decay"]))
    start_step = 0
    clips_seen = {name: 0 for name in DATASET_NAMES}
    if resume.is_file():
        state = load_checkpoint(resume, fsdp, optimizer, rank, world)
        start_step = int(state["global_step"])
        clips_seen.update({key: int(value) for key, value in state["clips_seen"].items()})
    if start_step >= target_steps:
        raise ValueError(f"checkpoint step {start_step} already reaches target {target_steps}")
    run = init_wandb(config, output, rank, args.disable_wandb)
    accumulation = int(config["gradient_accumulation"])
    k = int(config["targets_per_source"])
    diagnostic_every = int(config.get("diagnostic_every_steps", 20))
    checkpoint_steps = {int(value) for value in config.get("checkpoint_steps", [])}
    checkpoint_every = int(config.get("checkpoint_every_after", 5000))
    graceful_seconds = float(config.get("graceful_stop_hours", 68)) * 3600
    started = time.perf_counter()
    completed = start_step
    pool = ThreadPoolExecutor(max_workers=accumulation, thread_name_prefix="three-dataset-geometry")
    optimizer.zero_grad(set_to_none=True)
    try:
        for step in range(start_step, target_steps):
            name = dataset_for_step(step, seed)
            dataset = datasets[name]
            plans = [deterministic_sample_plan(
                dataset, name, seed, step, micro, rank, accumulation
            ) for micro in range(accumulation)]
            futures = [pool.submit(
                source_with_eligible_targets, dataset, index,
                np.random.default_rng(np.random.SeedSequence([seed, step, micro, rank, 771])).permutation(21)
            ) for micro, (index, _source, _rng) in enumerate(plans)]
            update_loss = 0.0
            update_epe = 0.0
            valid_points = 0
            pair_count = 0
            source_hist = torch.zeros(21, device=device, dtype=torch.float64)
            target_hist = torch.zeros(21, device=device, dtype=torch.float64)
            gap_hist = torch.zeros(21, device=device, dtype=torch.float64)
            step_started = time.perf_counter()
            for micro, ((index, _source, rng), future) in enumerate(zip(plans, futures)):
                source, xyz_all, valid_all = future.result()
                targets = sample_eligible_targets(valid_all, k, rng)
                xyz_np = xyz_all[targets]
                valid_np = valid_all[targets]
                normalized = (xyz_np - mean[None, :, None, None]) / scale[None, :, None, None]
                latent = torch.from_numpy(dataset.clean_latent(index))[None].to(device, dtype=dtype, non_blocking=True)
                source_t = torch.full((1, len(targets)), source, device=device, dtype=torch.long)
                target_t = torch.from_numpy(targets)[None].to(device, non_blocking=True)
                xyz = torch.from_numpy(normalized)[None].to(device, non_blocking=True)
                valid = torch.from_numpy(valid_np)[None].to(device, non_blocking=True)
                condition = conditions[name].to(device, dtype=dtype, non_blocking=True)
                sync = fsdp.no_sync() if micro + 1 < accumulation else nullcontext()
                with sync, torch.autocast("cuda", dtype=dtype):
                    prediction, _, _ = fsdp(latent, source_t, target_t, condition)
                    loss = masked_pair_smooth_l1(
                        prediction.float(), xyz.float(), valid,
                        beta=float(config.get("smooth_l1_beta", 0.05)),
                    ) / accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step={step}, micro={micro}")
                loss.backward()
                update_loss += float(loss.detach())
                with torch.no_grad():
                    metric_error = (prediction.float() - xyz) * torch.as_tensor(scale, device=device).view(1, 1, 3, 1, 1)
                    epe = torch.linalg.vector_norm(metric_error, dim=2)
                    update_epe += float(epe[valid].sum())
                    valid_points += int(valid.sum())
                pair_count += len(targets)
                source_hist[source] += 1
                for target_index in targets.tolist():
                    target_hist[target_index] += 1
                    gap_hist[abs(int(target_index) - source)] += 1
            gradient_norm = fsdp.clip_grad_norm_(float(config["gradient_clip"]))
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite gradient norm at step={step + 1}")
            lr_factor = apply_cosine_schedule(
                optimizer, step + 1, int(config["warmup_steps"]), int(config["schedule_horizon_steps"])
            )
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            completed = step + 1
            clips_seen[name] += world * accumulation
            elapsed = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated(device) / 2**30
            diagnostic = completed == start_step + 1 or completed % diagnostic_every == 0 or completed == target_steps
            if diagnostic:
                scalars = torch.tensor(
                    [update_loss, update_epe, valid_points, pair_count], device=device, dtype=torch.float64
                )
                dist.all_reduce(scalars, op=dist.ReduceOp.SUM)
                for histogram in (source_hist, target_hist, gap_hist):
                    dist.all_reduce(histogram, op=dist.ReduceOp.SUM)
            if rank == 0 and diagnostic:
                global_loss, global_epe_sum, global_valid, global_pairs = scalars.tolist()
                weights = fsdp.module.backbone.layer_weights().detach().float().cpu().tolist()
                payload = {
                    "global_step": completed, "train/loss": global_loss / world,
                    "train/raw_epe_m": global_epe_sum / max(global_valid, 1),
                    "train/dataset": DATASET_NAMES.index(name), "train/pairs": int(global_pairs),
                    "train/clips_seen_total": sum(clips_seen.values()),
                    **{f"train/clips_seen_{key}": value for key, value in clips_seen.items()},
                    **{f"train/layer_weight_{layer}": weight for layer, weight in zip(config["wan_hidden_layers"], weights)},
                    "system/world_size": world, "system/peak_cuda_memory_gib": peak,
                    "system/step_seconds": time.perf_counter() - step_started,
                    "system/elapsed_seconds": elapsed, "train/lr_factor": lr_factor,
                    "train/gradient_norm": float(gradient_norm),
                    **{f"sampling/source_{index}": int(value) for index, value in enumerate(source_hist.tolist())},
                    **{f"sampling/target_{index}": int(value) for index, value in enumerate(target_hist.tolist())},
                    **{f"sampling/gap_{index}": int(value) for index, value in enumerate(gap_hist.tolist())},
                }
                print(json.dumps(payload), flush=True)
                if run is not None:
                    run.log(payload, step=completed)
            local_stop = _STOP or elapsed >= graceful_seconds
            stop_tensor = torch.tensor(int(local_stop), device=device)
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
            periodic = completed in checkpoint_steps or (completed > 10000 and checkpoint_every and completed % checkpoint_every == 0)
            final = completed == target_steps or bool(stop_tensor.item())
            if periodic or final or (args.checkpoint_at_end and completed == target_steps):
                state = {
                    "global_step": completed, "clips_seen": clips_seen,
                    "dataset_cycle_offset": completed % 20,
                    "prompt_metadata": prompt_metadata,
                }
                save_checkpoint(output / "latest.pt", fsdp, optimizer, config, state, mean, scale, rank, world)
                if periodic:
                    save_checkpoint(output / f"checkpoint-{completed:07d}.pt", fsdp, optimizer, config, state, mean, scale, rank, world)
            if bool(stop_tensor.item()):
                break
    finally:
        pool.shutdown(wait=True)
        if run is not None:
            run.finish()
    dist.barrier()
    if rank == 0:
        result = {
            "completed_steps": completed, "target_steps": target_steps, "world_size": world,
            "clips_seen": clips_seen, "non_wan_parameters": non_wan_count,
            "targets_per_source": k, "peak_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "elapsed_seconds": time.perf_counter() - started,
            "lazy_vae_cache": bool(args.lazy_vae_cache),
            "lazy_vae_rank0": lazy_counts,
        }
        atomic_json(output / "train_status.json", result)
        print(json.dumps(result, indent=2), flush=True)
        print("THREE_DATASET_256_FSDP_OK", flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
