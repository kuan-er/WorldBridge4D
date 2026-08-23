"""FSDP orchestration for the registered WorldBridge4D training route."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
from pathlib import Path
import random
import signal
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import yaml

from ..data.constants import DATASET_NAMES
from ..data.factory import load_training_datasets, prepare_training_indexes
from ..data.sampling import deterministic_sample_plan, sample_eligible_targets, source_with_eligible_targets
from ..models.factory import build_real_model, precision_dtype
from ..data.text_conditions import load_dataset_text_conditions
from ..utils.io import atomic_json
from .config import validate_config
from .distributed import initialize_distributed
from .fsdp_checkpoint import (
    launch_durable_checkpoint_replica, load_optimizer_checkpoint,
    load_unwrapped_model_checkpoint, prune_periodic_checkpoints,
    save_checkpoint, update_latest_checkpoint,
)
from .lazy_vae import (
    LazyVAEPipeline, lazy_latent_owner, pipeline_work_for_rank, required_latent_indices,
    required_latent_requests, set_lazy_vae_identity, warm_lazy_latents,
)
from .objective import masked_pair_smooth_l1
from .optimizer import apply_fresh_group_warmup, parameter_groups
from .schedulers import apply_cosine_schedule, dataset_for_step, training_diagnostic_due
from .tracking import init_wandb, load_stats

_STOP = False

def stop_signal(_signum: int, _frame: Any) -> None:
    global _STOP
    _STOP = True

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--checkpoint-dir", help="fast local checkpoint directory; defaults to output-dir")
    parser.add_argument("--durable-checkpoint", help="best-effort asynchronous replica path for latest checkpoint")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument(
        "--wandb-log-after-step", type=int, default=-1,
        help="initialize/resume W&B normally, but upload metrics only after this global step",
    )
    parser.add_argument(
        "--no-checkpoint", action="store_true",
        help="debug only: skip periodic/final checkpoint materialization",
    )
    parser.add_argument("--lazy-vae-cache", action="store_true",
                        help="encode/cache all planned missing latents before constructing FSDP")
    parser.add_argument("--lazy-vae-pipeline", action="store_true",
                        help="warm a short prefix, then encode future clips beside training")
    parser.add_argument("--pipeline-lookahead-steps", type=int, default=16)
    args = parser.parse_args()
    if args.lazy_vae_cache and args.lazy_vae_pipeline:
        parser.error("--lazy-vae-cache and --lazy-vae-pipeline are mutually exclusive")
    if args.pipeline_lookahead_steps < 1:
        parser.error("--pipeline-lookahead-steps must be positive")
    for value in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        signal.signal(value, stop_signal)
    config = yaml.safe_load(Path(args.config).read_text())
    rank, world, local, device = initialize_distributed()
    validate_config(config, world)
    seed = int(config.get("seed", 20260812))
    random.seed(seed + rank); np.random.seed(seed + rank); torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    output = Path(args.output_dir).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else output
    durable_checkpoint = Path(args.durable_checkpoint).resolve() if args.durable_checkpoint else None
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for stale in output.glob("pipeline_error_rank*.json"):
            stale.unlink()
        prepare_training_indexes(config)
    dist.barrier()
    mean, scale = load_stats(config)
    conditions, prompt_metadata = load_dataset_text_conditions(config)
    datasets = load_training_datasets(
        config, allow_missing_latents=args.lazy_vae_cache or args.lazy_vae_pipeline,
    )
    target_steps = int(args.steps if args.steps is not None else config["max_steps"])
    resume = Path(args.resume) if args.resume else (checkpoint_dir / "latest.pt")
    # Cache planning needs only the checkpoint step. Use the tiny atomically
    # written status sidecar rather than faulting the 9+ GiB checkpoint from a
    # contended HDD before model construction; full checkpoint validation still
    # occurs once in load_checkpoint below.
    metadata = [None]
    if args.resume and not resume.is_file():
        raise FileNotFoundError(resume)
    if rank == 0 and resume.is_file():
        status_path = Path(config.get(
            "resume_status_path", resume.parent / "train_status.json"
        ))
        if not status_path.is_file():
            raise FileNotFoundError(
                f"checkpoint planning sidecar missing: {status_path}; "
                "refusing a second full checkpoint read"
            )
        status = json.loads(status_path.read_text())
        metadata[0] = int(status["completed_steps"])
    dist.broadcast_object_list(metadata, src=0)
    planned_start = int(metadata[0] or 0)
    if planned_start >= target_steps:
        raise ValueError(f"checkpoint step {planned_start} already reaches target {target_steps}")
    lazy_counts = None
    pipeline: LazyVAEPipeline | None = None
    if args.lazy_vae_cache:
        local_required = required_latent_indices(
            datasets, seed, planned_start, target_steps, rank,
            int(config["gradient_accumulation"]), int(config["microbatch_per_gpu"]),
        )
        gathered: list[Any] = [None] * world
        dist.all_gather_object(gathered, local_required)
        # A clip needed by multiple ranks gets one deterministic hash owner, so
        # overlapping rank-local plans remain balanced instead of assigning
        # almost every shared clip to the first requester (rank zero). File
        # locks still protect independent jobs sharing the persistent cache.
        owned = {name: [] for name in DATASET_NAMES}
        for name in DATASET_NAMES:
            all_indices = sorted({index for item in gathered for index in item[name]})
            for index in all_indices:
                if lazy_latent_owner(name, index, world) == rank:
                    owned[name].append(index)
        lazy_counts = warm_lazy_latents(config, datasets, owned, device, rank)
        dist.barrier()
    elif args.lazy_vae_pipeline:
        local_requests = required_latent_requests(
            datasets, seed, planned_start, target_steps, rank,
            int(config["gradient_accumulation"]), int(config["microbatch_per_gpu"]),
        )
        gathered_requests: list[Any] = [None] * world
        dist.all_gather_object(gathered_requests, local_requests)
        work = pipeline_work_for_rank(gathered_requests, rank, world)
        pipeline = LazyVAEPipeline(config, work, device, rank, output)
        set_lazy_vae_identity(datasets, pipeline.checksum)
        pipeline.warm_through(planned_start + args.pipeline_lookahead_steps)
        dist.barrier()
        pipeline.start()
        lazy_counts = pipeline.snapshot()
        if rank == 0:
            print(json.dumps({
                "event": "lazy_vae_pipeline_training_start",
                "planned_start": planned_start,
                "lookahead_steps": args.pipeline_lookahead_steps,
            }), flush=True)
    else:
        # Complete immutable shards remain fail-closed for formal training.
        required_check = required_latent_indices(
            datasets, seed, planned_start, target_steps, rank,
            int(config["gradient_accumulation"]), int(config["microbatch_per_gpu"]),
        )
        for name, indices in required_check.items():
            for index in indices:
                datasets[name].clean_latent(index)
    if rank == 0 and not args.lazy_vae_pipeline:
        print(json.dumps({"event": "three_dataset_cache_ready", "clips": {k: len(v) for k, v in datasets.items()}}), flush=True)

    # A resume checkpoint is a strict full-model state dict. Construct the Wan
    # architecture without reading the original pretrained tensor file. Rank 0
    # loads the full model before FSDP construction; sync_module_states then
    # broadcasts those exact weights. Loading only rank 0 after wrapping leaves
    # nonzero ranks at their construction weights and is not an exact resume.
    # Fresh runs still require and load the native Wan checkpoint.
    load_wan_pretrained = not resume.is_file()
    if rank == 0 and not load_wan_pretrained:
        print(json.dumps({
            "event": "wan_pretrained_load_skipped_for_full_resume",
            "checkpoint": str(resume),
        }), flush=True)
    model = build_real_model(
        config, device, load_wan_pretrained=load_wan_pretrained,
    )
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
    resume_payload: dict[str, Any] | None = None
    resume_state: dict[str, Any] | None = None
    resume_rng_states: list[Any] = []
    if resume.is_file():
        resume_payload, resume_state, resume_rng_states = load_unwrapped_model_checkpoint(
            resume, model, rank, world,
        )

    groups = parameter_groups(model, config)
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    current_group_names = {
        str(group["name"]): [parameter_names[id(parameter)] for parameter in group["params"]]
        for group in groups
    }
    trainable_names = [name for names in current_group_names.values() for name in names]
    trainable_count = sum(
        parameter.numel() for group in groups for parameter in group["params"]
    )
    trainable_mode = str(config.get("trainable_mode"))
    if trainable_mode == "source_rgb_plus_wan_decoder":
        adapter_ids = {id(value) for value in model.backbone.adapter_parameters}
        bypassed_ids = {id(value) for value in model.backbone.bypassed_parameters}
        expected_joint = {
            name for name, parameter in model.named_parameters()
            if (
                name.startswith("decoder.")
                or (
                    name.startswith("backbone.")
                    and id(parameter) not in adapter_ids
                    and id(parameter) not in bypassed_ids
                )
            )
        }
        actual_joint = set(trainable_names)
        if actual_joint != expected_joint:
            raise RuntimeError(
                "joint Wan+decoder freeze audit failed: "
                f"missing={sorted(expected_joint - actual_joint)[:8]}, "
                f"unexpected={sorted(actual_joint - expected_joint)[:8]}"
            )
    if rank == 0:
        print(json.dumps({
            "event": "trainable_parameter_groups",
            "mode": str(config.get("trainable_mode")),
            "trainable_parameters": trainable_count,
            "groups": {
                name: {
                    "parameters": len(current_group_names[name]),
                    "lr": float(group["lr"]),
                    "weight_decay": float(group.get("weight_decay", config["weight_decay"])),
                }
                for name, group in ((str(value["name"]), value) for value in groups)
            },
        }), flush=True)
    dtype = precision_dtype(config["precision"])
    auto_wrap = lambda module, recurse, nonwrapped_numel: size_based_auto_wrap_policy(
        module, recurse, nonwrapped_numel,
        min_num_params=int(config.get("fsdp_min_num_params", 5_000_000)),
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
    if resume_state is not None:
        load_optimizer_checkpoint(
            resume_payload, fsdp, optimizer, resume_rng_states, rank,
        )
        start_step = int(resume_state["global_step"])
        clips_seen.update({key: int(value) for key, value in resume_state["clips_seen"].items()})
        del resume_payload
        if rank == 0:
            print(json.dumps({
                "event": "resume_state_loaded", "step": start_step,
                "world_size": world, "optimizer": "restored", "rng_states": len(resume_rng_states),
            }), flush=True)
    if start_step >= target_steps:
        raise ValueError(f"checkpoint step {start_step} already reaches target {target_steps}")
    run = init_wandb(config, output, rank, args.disable_wandb)
    accumulation = int(config["gradient_accumulation"])
    microbatch_per_gpu = int(config["microbatch_per_gpu"])
    k = int(config["targets_per_source"])
    use_source_rgb = bool(config.get("source_rgb_pyramid", False))
    diagnostic_every = int(config.get("diagnostic_every_steps", 20))
    ensure_dataset_diagnostics = bool(
        config.get("diagnostic_ensure_dataset_coverage", False)
    )
    last_diagnostic_cycle: dict[str, int] = {}
    completed_diagnostic_cycles: set[int] = set()
    extension_start = config.get("schedule_extension_start_step")
    extension_horizon = config.get("schedule_extension_horizon_steps")
    checkpoint_steps = {int(value) for value in config.get("checkpoint_steps", [])}
    checkpoint_every = int(config.get("checkpoint_every_after", 5000))
    graceful_seconds = float(config.get("graceful_stop_hours", 68)) * 3600
    started = time.perf_counter()
    completed = start_step
    prefetch_depth = int(config.get("geometry_prefetch_depth", 2))
    prefetch_workers = int(config.get(
        "geometry_prefetch_workers", min(4, accumulation * microbatch_per_gpu * 2),
    ))
    pool = ThreadPoolExecutor(
        max_workers=prefetch_workers,
        thread_name_prefix="three-dataset-geometry",
    )
    optimizer.zero_grad(set_to_none=True)
    try:
        slots_per_rank = accumulation * microbatch_per_gpu

        def timed_geometry(step_dataset, index: int, permutation: np.ndarray,
                           required_targets: int, fallback_seed: int):
            task_started = time.perf_counter()
            candidates = [int(index)]
            fallback_rng = np.random.default_rng(int(fallback_seed))
            fallback_order = fallback_rng.permutation(len(step_dataset))
            candidates.extend(int(value) for value in fallback_order if int(value) != int(index))
            last_error = None
            for candidate_number, candidate in enumerate(candidates):
                candidate_sources = permutation if candidate_number == 0 else fallback_rng.permutation(21)
                try:
                    source, xyz, valid = source_with_eligible_targets(
                        step_dataset, candidate, candidate_sources,
                        min_targets=required_targets,
                    )
                except ValueError as error:
                    last_error = error
                    continue
                # RGB failures are data-contract errors, not a reason to change
                # the deterministic geometry fallback clip/source.
                source_rgb = (
                    step_dataset.source_rgb(candidate, source)
                    if use_source_rgb else None
                )
                return (candidate, source, xyz, valid, source_rgb), time.perf_counter() - task_started
            raise ValueError(
                f"dataset has no clip/source with K={required_targets} eligible targets"
            ) from last_error

        def plan_step(step: int):
            """Plan one update and submit its deterministic geometry futures."""
            step_name = dataset_for_step(step, seed)
            step_dataset = datasets[step_name]
            step_plans = [deterministic_sample_plan(
                step_dataset, step_name, seed, step, slot, rank, slots_per_rank
            ) for slot in range(slots_per_rank)]
            step_futures = [pool.submit(
                timed_geometry, step_dataset, index,
                np.random.default_rng(np.random.SeedSequence([seed, step, slot, rank, 771])).permutation(21),
                k,
                int(np.random.SeedSequence([seed, step, slot, rank, 772]).generate_state(1)[0]),
            ) for slot, (index, _source, _rng) in enumerate(step_plans)]
            return step_name, step_dataset, step_plans, step_futures

        # Depth=2 preserves the previous current+next-step submission policy.
        # The optimized route uses four in-flight steps without adding workers.
        pending = deque()
        next_plan_step = start_step

        def refill_plans() -> None:
            nonlocal next_plan_step
            while len(pending) < prefetch_depth and next_plan_step < target_steps:
                pending.append((next_plan_step, *plan_step(next_plan_step)))
                next_plan_step += 1

        refill_plans()
        for step in range(start_step, target_steps):
            planned_step, name, dataset, plans, futures = pending.popleft()
            if planned_step != step:
                raise RuntimeError(f"prefetch plan order mismatch: {planned_step} != {step}")
            refill_plans()
            if pipeline is not None:
                pipeline.wait_for(
                    dataset, name, [index for index, _source, _rng in plans],
                    float(config.get("pipeline_wait_timeout_seconds", 3600)),
                )
            update_loss = 0.0
            update_epe = 0.0
            valid_points = 0
            pair_count = 0
            source_hist = torch.zeros(21, device=device, dtype=torch.float64)
            target_hist = torch.zeros(21, device=device, dtype=torch.float64)
            gap_hist = torch.zeros(21, device=device, dtype=torch.float64)
            step_started = time.perf_counter()
            geometry_wait_seconds = 0.0
            geometry_task_max_seconds = 0.0
            latent_load_seconds = 0.0
            for micro in range(accumulation):
                begin = micro * microbatch_per_gpu
                group = list(zip(
                    plans[begin:begin + microbatch_per_gpu],
                    futures[begin:begin + microbatch_per_gpu],
                ))
                batch_values = []
                target_counts = []
                for (_planned_index, _source, rng), future in group:
                    wait_started = time.perf_counter()
                    (index, source, xyz_all, valid_all, source_rgb_np), task_seconds = future.result()
                    geometry_wait_seconds += time.perf_counter() - wait_started
                    geometry_task_max_seconds = max(geometry_task_max_seconds, task_seconds)
                    targets = sample_eligible_targets(valid_all, k, rng)
                    batch_values.append((
                        index, source, targets, xyz_all[targets], valid_all[targets],
                        source_rgb_np,
                    ))
                    target_counts.append(len(targets))
                if len(set(target_counts)) != 1:
                    raise ValueError(
                        f"microbatch clips have different eligible target counts: {target_counts}"
                    )
                latent_started = time.perf_counter()
                latents_np = np.stack([dataset.clean_latent(value[0]) for value in batch_values])
                latent_load_seconds += time.perf_counter() - latent_started
                normalized_np = np.stack([
                    (value[3] - mean[None, :, None, None]) / scale[None, :, None, None]
                    for value in batch_values
                ])
                valid_np = np.stack([value[4] for value in batch_values])
                source_rgb_t = None
                if use_source_rgb:
                    source_rgb_np = np.stack([value[5] for value in batch_values])
                    if source_rgb_np.shape != (len(batch_values), 256, 256, 3) \
                            or source_rgb_np.dtype != np.uint8:
                        raise RuntimeError(
                            f"source RGB batch must be uint8 [B,256,256,3], got "
                            f"{source_rgb_np.dtype} {source_rgb_np.shape}"
                        )
                    source_rgb_t = torch.from_numpy(source_rgb_np).permute(0, 3, 1, 2).to(
                        device, dtype=dtype, non_blocking=True,
                    )
                    source_rgb_t = source_rgb_t / 127.5 - 1.0
                latent = torch.from_numpy(latents_np).to(device, dtype=dtype, non_blocking=True)
                source_t = torch.tensor([
                    [value[1]] * len(value[2]) for value in batch_values
                ], device=device, dtype=torch.long)
                target_t = torch.from_numpy(np.stack([value[2] for value in batch_values])).to(
                    device, non_blocking=True
                )
                xyz = torch.from_numpy(normalized_np).to(device, non_blocking=True)
                valid = torch.from_numpy(valid_np).to(device, non_blocking=True)
                condition = conditions[name].to(device, dtype=dtype, non_blocking=True)
                sync = fsdp.no_sync() if micro + 1 < accumulation else nullcontext()
                with sync, torch.autocast("cuda", dtype=dtype):
                    prediction, _, _ = fsdp(
                        latent, source_t, target_t, condition, source_rgb_t,
                    )
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
                pair_count += sum(target_counts)
                for _index, source, targets, _xyz, _valid, _source_rgb in batch_values:
                    source_hist[source] += 1
                    for target_index in targets.tolist():
                        target_hist[target_index] += 1
                        gap_hist[abs(int(target_index) - source)] += 1
            gradient_norm = fsdp.clip_grad_norm_(float(config["gradient_clip"]))
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite gradient norm at step={step + 1}")
            lr_factor = apply_cosine_schedule(
                optimizer,
                step + 1,
                int(config["warmup_steps"]),
                int(config["schedule_horizon_steps"]),
                None if extension_start is None else int(extension_start),
                None if extension_horizon is None else int(extension_horizon),
            )
            fresh_group_warmup_factor = apply_fresh_group_warmup(
                optimizer,
                {"wan_backbone", "dense_decoder"},
                step + 1,
                start_step,
                int(config.get("joint_fresh_group_warmup_steps", 0)),
                float(config.get("joint_fresh_group_max_lr_scale", 1.0)),
            ) if str(config.get("trainable_mode")) == "source_rgb_plus_wan_decoder" else 1.0
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            completed = step + 1
            clips_seen[name] += world * accumulation * microbatch_per_gpu
            elapsed = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated(device) / 2**30
            diagnostic = training_diagnostic_due(
                completed,
                start_step,
                target_steps,
                name,
                step,
                diagnostic_every,
                ensure_dataset_diagnostics,
                last_diagnostic_cycle,
            )
            if diagnostic:
                last_diagnostic_cycle[name] = step // 20
                scalars = torch.tensor(
                    [update_loss, update_epe, valid_points, pair_count], device=device, dtype=torch.float64
                )
                timing_max = torch.tensor([
                    geometry_wait_seconds, geometry_task_max_seconds, latent_load_seconds,
                ], device=device, dtype=torch.float64)
                dist.all_reduce(scalars, op=dist.ReduceOp.SUM)
                dist.all_reduce(timing_max, op=dist.ReduceOp.MAX)
                for histogram in (source_hist, target_hist, gap_hist):
                    dist.all_reduce(histogram, op=dist.ReduceOp.SUM)
                rgb_alpha_names = list(fsdp.module.decoder.upsampler.source_fusions)
                if rgb_alpha_names:
                    rgb_alpha_stats = torch.stack([
                        torch.stack((
                            fusion.alpha.detach().double().sum(),
                            torch.as_tensor(
                                fusion.alpha.numel(), device=device, dtype=torch.float64,
                            ),
                        ))
                        for fusion in fsdp.module.decoder.upsampler.source_fusions.values()
                    ])
                    dist.all_reduce(rgb_alpha_stats, op=dist.ReduceOp.SUM)
                else:
                    rgb_alpha_stats = torch.empty((0, 2), device=device, dtype=torch.float64)
            if rank == 0 and diagnostic:
                global_loss, global_epe_sum, global_valid, global_pairs = scalars.tolist()
                dataset_loss = global_loss / world
                raw_epe_m = global_epe_sum / max(global_valid, 1)
                weights = fsdp.module.backbone.layer_weights().detach().float().cpu().tolist()
                rgb_alphas = {
                    scale_name: float(rgb_alpha_stats[index, 0] / rgb_alpha_stats[index, 1])
                    for index, scale_name in enumerate(rgb_alpha_names)
                }
                payload = {
                    "global_step": completed, "train/loss": dataset_loss,
                    f"train/loss_by_dataset/{name}": dataset_loss,
                    "train/raw_epe_m": raw_epe_m,
                    f"train/raw_epe_m_by_dataset/{name}": raw_epe_m,
                    "train/dataset": DATASET_NAMES.index(name), "train/pairs": int(global_pairs),
                    "train/clips_seen_total": sum(clips_seen.values()),
                    **{f"train/clips_seen_{key}": value for key, value in clips_seen.items()},
                    **{f"train/layer_weight_{layer}": weight for layer, weight in zip(config["wan_hidden_layers"], weights)},
                    **{f"train/source_rgb_alpha_{scale_name}": value for scale_name, value in rgb_alphas.items()},
                    "system/world_size": world, "system/peak_cuda_memory_gib": peak,
                    "system/step_seconds": time.perf_counter() - step_started,
                    "system/geometry_wait_seconds_max_rank": float(timing_max[0]),
                    "system/geometry_task_seconds_max_rank": float(timing_max[1]),
                    "system/latent_load_seconds_max_rank": float(timing_max[2]),
                    "system/geometry_prefetch_depth": prefetch_depth,
                    "system/geometry_prefetch_workers": prefetch_workers,
                    "system/diagnostic_dataset_coverage": int(ensure_dataset_diagnostics),
                    "system/elapsed_seconds": elapsed, "train/lr_factor": lr_factor,
                    "train/fresh_group_warmup_factor": fresh_group_warmup_factor,
                    "train/gradient_norm": float(gradient_norm),
                    **{f"sampling/source_{index}": int(value) for index, value in enumerate(source_hist.tolist())},
                    **{f"sampling/target_{index}": int(value) for index, value in enumerate(target_hist.tolist())},
                    **{f"sampling/gap_{index}": int(value) for index, value in enumerate(gap_hist.tolist())},
                }
                print(json.dumps(payload), flush=True)
                if run is not None and completed > args.wandb_log_after_step:
                    run.log(payload, step=completed)
                diagnostic_cycle = step // 20
                if diagnostic_cycle not in completed_diagnostic_cycles and all(
                    last_diagnostic_cycle.get(dataset_name) == diagnostic_cycle
                    for dataset_name in DATASET_NAMES
                ):
                    completed_diagnostic_cycles.add(diagnostic_cycle)
                    print(json.dumps({
                        "event": "diagnostic_dataset_cycle_complete",
                        "cycle": diagnostic_cycle,
                        "global_step": completed,
                        "datasets": list(DATASET_NAMES),
                        "metrics": ["loss", "raw_epe_m"],
                    }), flush=True)
            local_stop = _STOP or elapsed >= graceful_seconds
            stop_tensor = torch.tensor(int(local_stop), device=device)
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
            periodic = completed in checkpoint_steps or (completed > 10000 and checkpoint_every and completed % checkpoint_every == 0)
            final = completed == target_steps or bool(stop_tensor.item())
            if not args.no_checkpoint and (periodic or final):
                state = {
                    "global_step": completed, "clips_seen": clips_seen,
                    "dataset_cycle_offset": completed % 20,
                    "prompt_metadata": prompt_metadata,
                }
                # Materialize one immutable checkpoint on the fast local tier.
                # ``latest.pt`` is an atomic hard link, avoiding a second 9+ GiB
                # serialization. A detached low-priority copier independently
                # mirrors this immutable file to durable storage; its failures
                # are deliberately outside the distributed training process.
                checkpoint = checkpoint_dir / f"checkpoint-{completed:07d}.pt"
                save_checkpoint(checkpoint, fsdp, optimizer, config, state, mean, scale, rank, world)
                if rank == 0:
                    checkpoint_status = {
                        "completed_steps": completed,
                        "world_size": world,
                        "clips_seen": clips_seen,
                    }
                    atomic_json(checkpoint_dir / "train_status.json", checkpoint_status)
                    if output != checkpoint_dir:
                        atomic_json(output / "train_status.json", checkpoint_status)
                    update_latest_checkpoint(checkpoint, checkpoint_dir / "latest.pt")
                    if durable_checkpoint is not None:
                        launch_durable_checkpoint_replica(
                            checkpoint, durable_checkpoint, completed,
                        )
                    removed = prune_periodic_checkpoints(
                        checkpoint_dir, int(config.get("checkpoint_keep_last", 0))
                    )
                    if removed:
                        print(json.dumps({
                            "event": "checkpoint_prune", "removed": removed,
                        }), flush=True)
                dist.barrier()
            if bool(stop_tensor.item()):
                break
    finally:
        pool.shutdown(wait=True)
        if pipeline is not None:
            pipeline.close()
            lazy_counts = pipeline.snapshot()
        if run is not None:
            run.finish()
    dist.barrier()
    if rank == 0:
        result = {
            "completed_steps": completed, "target_steps": target_steps, "world_size": world,
            "clips_seen": clips_seen, "non_wan_parameters": non_wan_count,
            "targets_per_source": k, "peak_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "elapsed_seconds": time.perf_counter() - started,
            "source_rgb_pyramid": use_source_rgb,
            "trainable_mode": str(config.get("trainable_mode")),
            "trainable_parameters": trainable_count,
            "lazy_vae_cache": bool(args.lazy_vae_cache),
            "lazy_vae_pipeline": bool(args.lazy_vae_pipeline),
            "lazy_vae_rank0": lazy_counts,
        }
        atomic_json(output / "train_status.json", result)
        print(json.dumps(result, indent=2), flush=True)
        print("THREE_DATASET_256_FSDP_OK", flush=True)
    dist.barrier(); dist.destroy_process_group()


class WorldBridgeTrainer:
    """Explicit training entry point used by the thin CLI wrapper."""

    def fit(self) -> None:
        main()
