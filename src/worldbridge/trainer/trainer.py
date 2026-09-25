"""FSDP orchestration for the registered WorldBridge4D training route."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import faulthandler
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
from ..data.sampling import sample_eligible_targets
from ..models.decoder import DenseUpsampler2D
from ..models.decoder.query_decoder import CAMERA_MODULE_PREFIXES
from ..models.factory import build_real_model, precision_dtype
from ..data.text_conditions import load_dataset_text_conditions
from ..utils.io import atomic_json
from .batching import GeometryPrefetcher
from .config import validate_config
from .distributed import initialize_distributed
from .fsdp_checkpoint import (
    launch_durable_checkpoint_replica, load_filtered_optimizer_checkpoint,
    load_optimizer_checkpoint, load_unwrapped_model_checkpoint, prune_periodic_checkpoints,
    save_checkpoint, update_latest_checkpoint,
)
from .lazy_vae import set_lazy_vae_identity
from .camera_objective import camera_ray_objective, supervision_camera_batch

# H033 reads the camera from decoder-native modules; these prefixes are the
# audited fresh parameters for a structural migration into an older trunk.
CAMERA_PARAMETER_PREFIXES = tuple(f'decoder.{prefix}' for prefix in CAMERA_MODULE_PREFIXES)
from .objective import masked_pair_smooth_l1
from .optimizer import apply_fresh_group_warmup, parameter_groups
from .precision import assert_fp32_optimizer_storage, prepare_fsdp_master_parameters
from .preflight import DEFAULT_PREFLIGHT_UPDATES, validate_startup_latents
from .schedulers import (apply_cosine_schedule, apply_lr_restart_schedule, training_diagnostic_due,
                         validated_mix_counts, deterministic_dataset_schedule)
from .tracking import init_wandb, load_stats

_STOP = False


def fsdp_auto_wrap_policy(
    module: torch.nn.Module, recurse: bool, nonwrapped_numel: int,
    *, min_num_params: int,
) -> bool:
    """Size-wrap modules without wrapping the upsampler method boundary.

    The pre-attention RGB path calls ``DenseUpsampler2D.encode_source_rgb``
    from inside the decoder forward. Wrapping the upsampler itself would make
    that custom method bypass FSDP's forward all-gather and expose a sharded
    one-dimensional convolution weight. Its large children may still be
    wrapped; the remaining parameters are gathered by the decoder's wrapper.
    """
    if isinstance(module, DenseUpsampler2D) and not recurse:
        return False
    return size_based_auto_wrap_policy(
        module, recurse, nonwrapped_numel, min_num_params=min_num_params,
    )


def non_wan_parameter_counts(model) -> tuple[int, int, int]:
    """Adapter, decoder (camera readout included) and camera parameter counts."""
    adapter = sum(p.numel() for p in model.backbone.adapter_parameters)
    decoder_parameters = list(model.decoder.parameters())
    decoder_ids = {id(p) for p in decoder_parameters}
    camera_parameters = list(model.camera_head_parameters())
    if any(id(p) not in decoder_ids for p in camera_parameters):
        raise RuntimeError('camera parameters must live inside the decoder')
    return (adapter, sum(p.numel() for p in decoder_parameters),
            sum(p.numel() for p in camera_parameters))


def execution_stop_step(start: int, horizon: int, stop_after_updates: int | None) -> int:
    """Bound a diagnostic invocation without changing the configured LR horizon."""
    if stop_after_updates is None:
        return horizon
    if stop_after_updates < 1:
        raise ValueError('stop-after-updates must be positive')
    return min(horizon, start + stop_after_updates)


def stop_signal(_signum: int, _frame: Any) -> None:
    global _STOP
    _STOP = True

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument(
        "--finetune-from",
        help="load an audited structural extension and retained optimizer moments",
    )
    parser.add_argument("--checkpoint-dir", help="fast local checkpoint directory; defaults to output-dir")
    parser.add_argument("--durable-checkpoint", help="best-effort asynchronous replica path for latest checkpoint")
    parser.add_argument("--steps", type=int)
    parser.add_argument('--stop-after-updates', type=int,
                        help='bounded full-resume diagnosis: checkpoint and exit without changing LR horizon')
    parser.add_argument('--startup-preflight-updates', type=int, default=DEFAULT_PREFLIGHT_UPDATES,
                        help='non-lazy startup payload check: first N updates (default 5); 0 checks full invocation; later reads remain strict')
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument(
        "--wandb-log-after-step", type=int, default=-1,
        help="initialize/resume W&B normally, but upload metrics only after this global step",
    )
    parser.add_argument(
        "--no-checkpoint", action="store_true",
        help="debug only: skip periodic/final checkpoint materialization",
    )
    args = parser.parse_args()
    if args.resume and args.finetune_from:
        parser.error("--resume and --finetune-from are mutually exclusive")
    if args.stop_after_updates is not None and (args.stop_after_updates < 1 or not (args.resume or args.finetune_from)
            or args.no_checkpoint):
        parser.error('--stop-after-updates requires positive full-resume checkpointed diagnosis')
    if args.startup_preflight_updates < 0:
        parser.error('--startup-preflight-updates must be nonnegative (0 = full invocation)')
    for value in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        signal.signal(value, stop_signal)
    config = yaml.safe_load(Path(args.config).read_text())
    rank, world, local, device = initialize_distributed(
        timeout_seconds=float(config.get("distributed_timeout_seconds", 86400)),
    )
    validate_config(config, world)
    camera_cfg = config.get('camera_supervision')
    if args.finetune_from and args.stop_after_updates and camera_cfg is None:
        raise ValueError('bounded structural migration is restricted to camera extension')
    dataset_mix_counts = validated_mix_counts(config.get('dataset_mix_counts'))
    if rank == 0:
        print(json.dumps({'event': 'dataset_schedule_protocol', 'counts_per20': dataset_mix_counts,
                          'schedule': deterministic_dataset_schedule(config.get('seed', 20260812), dataset_mix_counts),
                          'phase_origin': config.get('selected_checkpoint_step'),
                          'K': config['targets_per_source']}), flush=True)
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
    datasets = load_training_datasets(config)
    target_steps = int(args.steps if args.steps is not None else config["max_steps"])
    if config.get('native_kubric512_b1_a4_k15', False) or config.get('native_kubric512_b1_a4_k9', False) or config.get('native_kubric512_b1_a4_k5', False):
        if target_steps != int(config['max_steps']) or not args.resume:
            raise ValueError('native training requires the declared full-resume budget and no lazy input generation')
        from ..data.cache.native import file_sha256
        set_lazy_vae_identity(datasets, file_sha256(config['vae_checkpoint']))
    resume = Path(args.resume) if args.resume else (checkpoint_dir / "latest.pt")
    finetune_from = Path(args.finetune_from) if args.finetune_from else None
    # Cache planning needs only the checkpoint step. Use the tiny atomically
    # written status sidecar rather than faulting the 9+ GiB checkpoint from a
    # contended HDD before model construction; full checkpoint validation still
    # occurs once in load_checkpoint below.
    metadata = [None]
    if args.resume and not resume.is_file():
        raise FileNotFoundError(resume)
    if finetune_from is not None and not finetune_from.is_file():
        raise FileNotFoundError(finetune_from)
    if rank == 0 and finetune_from is not None:
        metadata[0] = int(config.get("finetune_expected_global_step", -1))
        if metadata[0] < 0:
            raise ValueError("structural fine-tune requires finetune_expected_global_step")
    elif rank == 0 and resume.is_file():
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
    execution_end = execution_stop_step(planned_start, target_steps, args.stop_after_updates)
    if args.stop_after_updates is not None:
        print(json.dumps({'event': 'bounded_resume_diagnostic', 'rank': rank,
                          'start': planned_start, 'execution_end': execution_end,
                          'unchanged_LR_horizon': target_steps}), flush=True)
    if planned_start >= target_steps:
        raise ValueError(f"checkpoint step {planned_start} already reaches target {target_steps}")
    # Keep complete catalog checks above, but do not scan the whole future phase.
    # Actual batch indices are still read/validated before every forward below.
    startup_preflight = validate_startup_latents(
        datasets, seed, planned_start, execution_end, rank,
        int(config["gradient_accumulation"]), int(config["microbatch_per_gpu"]),
        updates=args.startup_preflight_updates, dataset_mix_counts=dataset_mix_counts,
    )
    if rank == 0:
        print(json.dumps({"event": "three_dataset_cache_ready", "clips": {k: len(v) for k, v in datasets.items()},
                          "startup_preflight": startup_preflight,
                          "note": "catalog ready; prefix validation does not certify every future payload"}), flush=True)

    # A resume checkpoint is a strict full-model state dict. Construct the Wan
    # architecture without reading the original pretrained tensor file. Rank 0
    # loads the full model before FSDP construction; sync_module_states then
    # broadcasts those exact weights. Loading only rank 0 after wrapping leaves
    # nonzero ranks at their construction weights and is not an exact resume.
    # Fresh runs still require and load the native Wan checkpoint.
    load_wan_pretrained = not resume.is_file() and finetune_from is None
    if rank == 0 and not load_wan_pretrained:
        print(json.dumps({
            "event": (
                "wan_pretrained_load_skipped_for_structural_finetune"
                if finetune_from is not None
                else "wan_pretrained_load_skipped_for_full_resume"
            ),
            "checkpoint": str(finetune_from if finetune_from is not None else resume),
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
    adapter_count, decoder_count, camera_count = non_wan_parameter_counts(model)
    # The H033 camera readout lives inside the decoder, so decoder_count already
    # contains camera_count; adding it again double-counts 926,218 parameters.
    non_wan_count = adapter_count + decoder_count
    expected_non_wan = int(config.get("expected_non_wan_parameters", 193586693))
    if non_wan_count != expected_non_wan:
        raise RuntimeError(f"non-Wan readout parameters {non_wan_count:,} != expected {expected_non_wan:,}")
    master_precision = str(config.get("fsdp_master_precision", "model"))
    prepare_fsdp_master_parameters(model, master_precision)
    finetune_payload: dict[str, Any] | None = None
    finetune_state: dict[str, Any] | None = None
    finetune_rng_states: list[Any] = []
    if finetune_from is not None:
        finetune_payload, finetune_state, finetune_rng_states = load_unwrapped_model_checkpoint(
            finetune_from, model, rank, world,
            allowed_missing_prefixes=(CAMERA_PARAMETER_PREFIXES if camera_cfg else ("decoder.query_rgb_projection.",)),
            allowed_unexpected_prefixes=tuple(config.get('finetune_drop_prefixes', ())),
        )
        expected_step = int(config.get("finetune_expected_global_step", -1))
        if int(finetune_state["global_step"]) != expected_step:
            raise RuntimeError(
                f"fine-tune source step {finetune_state['global_step']} != expected {expected_step}"
            )
        expected_clips = {
            key: int(value)
            for key, value in config.get("finetune_expected_clips_seen", {}).items()
        }
        actual_clips = {
            key: int(value) for key, value in finetune_state["clips_seen"].items()
        }
        if actual_clips != expected_clips:
            raise RuntimeError(
                f"fine-tune source clip counters {actual_clips} != expected {expected_clips}"
            )
    resume_payload: dict[str, Any] | None = None
    resume_state: dict[str, Any] | None = None
    resume_rng_states: list[Any] = []
    if resume.is_file() and finetune_from is None:
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
    if trainable_mode == "decoder_only":
        expected_decoder = {
            name for name, _parameter in model.named_parameters()
            if name.startswith("decoder.")
        }
        actual_decoder = set(trainable_names)
        if actual_decoder != expected_decoder:
            raise RuntimeError(
                "decoder-only freeze audit failed: "
                f"missing={sorted(expected_decoder - actual_decoder)[:8]}, "
                f"unexpected={sorted(actual_decoder - expected_decoder)[:8]}"
            )
    elif trainable_mode == "source_rgb_plus_wan_decoder":
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
    auto_wrap = lambda module, recurse, nonwrapped_numel: fsdp_auto_wrap_policy(
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
    if finetune_state is not None:
        load_filtered_optimizer_checkpoint(
            finetune_payload, fsdp, optimizer, finetune_rng_states,
            current_group_names, rank,
            allowed_fresh_prefixes=(CAMERA_PARAMETER_PREFIXES if camera_cfg else ("decoder.query_rgb_projection.",)),
            require_all_source_state=camera_cfg is not None,
            allowed_unexpected_prefixes=tuple(config.get('finetune_drop_prefixes', ())),
        )
        start_step = int(finetune_state["global_step"])
        clips_seen.update({
            key: int(value) for key, value in finetune_state["clips_seen"].items()
        })
        del finetune_payload
        if rank == 0:
            print(json.dumps({
                "event": "structural_finetune_state_loaded", "step": start_step,
                "world_size": world, "optimizer": "filtered_restored",
                "rng_states": len(finetune_rng_states),
            }), flush=True)
    if resume_state is not None:
        if config.get('full_mode_unfreeze_resume'):
            load_filtered_optimizer_checkpoint(
                resume_payload, fsdp, optimizer, resume_rng_states,
                current_group_names, rank, allowed_fresh_prefixes=("backbone.",),
            )
            optimizer_label = "filtered_restored"
        else:
            load_optimizer_checkpoint(
                resume_payload, fsdp, optimizer, resume_rng_states, rank,
            )
            optimizer_label = "restored"
        start_step = int(resume_state["global_step"])
        clips_seen.update({key: int(value) for key, value in resume_state["clips_seen"].items()})
        del resume_payload
        if rank == 0:
            print(json.dumps({
                "event": "resume_state_loaded", "step": start_step,
                "world_size": world, "optimizer": optimizer_label, "rng_states": len(resume_rng_states),
            }), flush=True)
    if args.stop_after_updates is not None and start_step != planned_start:
        raise ValueError('diagnostic checkpoint planning sidecar does not match restored step')
    if master_precision == "fp32":
        assert_fp32_optimizer_storage(optimizer)
        if rank == 0:
            print(json.dumps({"event": "fp32_master_storage_verified", "step": start_step,
                              "parameters": "float32", "adam_moments": "float32",
                              "compute_precision": config["precision"]}), flush=True)
    if (config.get('native_kubric512_k5_10k') or config.get('native_kubric512_k9_mix_trial')) and start_step < int(config['selected_checkpoint_step']):
        raise ValueError('native continuation cannot resume before its verified phase origin')
    if start_step >= target_steps:
        raise ValueError(f"checkpoint step {start_step} already reaches target {target_steps}")
    lr_restart = config.get("lr_restart")
    if lr_restart is not None:
        if not int(lr_restart["start_step"]) <= start_step < target_steps <= int(lr_restart["end_step"]):
            raise ValueError("restored checkpoint/target outside LR restart phase")
        if rank == 0:
            print(json.dumps({"event": "lr_restart_phase", "restored_step": start_step,
                              "protocol": lr_restart, "optimizer_moments": "preserved"}), flush=True)
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
    checkpoint_steps = {int(value) for value in config.get("checkpoint_steps", [])}
    checkpoint_every = int(config.get("checkpoint_every_after", 5000))
    graceful_seconds = float(config.get("graceful_stop_hours", 68)) * 3600
    started = time.perf_counter()
    completed = start_step
    prefetch_depth = int(config.get("geometry_prefetch_depth", 2))
    prefetch_workers = int(config.get(
        "geometry_prefetch_workers", min(4, accumulation * microbatch_per_gpu * 2),
    ))
    prefetcher = GeometryPrefetcher(
        datasets,
        seed=seed,
        rank=rank,
        accumulation=accumulation,
        microbatch_per_gpu=microbatch_per_gpu,
        targets_per_source=k,
        use_source_rgb=use_source_rgb,
        camera_supervision=camera_cfg is not None,
        start_step=start_step,
        target_steps=execution_end,
        depth=prefetch_depth,
        workers=prefetch_workers,
        dataset_mix_counts=dataset_mix_counts,
    )
    optimizer.zero_grad(set_to_none=True)
    trace_updates = int(config.get("trace_first_updates", 0))
    stall_seconds = float(config.get("runtime_stall_traceback_seconds", 0))
    empty_cache_every = int(config.get("cuda_empty_cache_every_steps", 0))

    def trace_phase(stage: str, step: int, micro: int | None = None) -> None:
        if step < start_step + trace_updates:
            # This marks host-side progress, not completed optimizer updates or
            # synchronized GPU timings. Do not trigger global_step listeners.
            print(json.dumps({"event": "training_phase", "rank": rank,
                              "update_number": step + 1, "micro": micro,
                              "stage": stage, "monotonic_seconds": time.perf_counter()}), flush=True)

    try:
        prefetcher.refill()
        for step in range(start_step, execution_end):
            if stall_seconds > 0:
                faulthandler.dump_traceback_later(stall_seconds, repeat=True)
            trace_phase("update_start", step)
            planned = prefetcher.pop(step)
            name = planned.dataset_name
            dataset = planned.dataset
            image_size = int(getattr(dataset, 'image_size', config['image_size']))
            plans = planned.sample_plans
            futures = planned.geometry_futures
            update_loss = 0.0
            xyz_loss_sum = 0.0
            update_epe = 0.0
            valid_points = 0
            pair_count = 0
            camera_metrics = {} if camera_cfg is not None else None
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
                trace_phase("geometry_wait_start", step, micro)
                batch_values = []
                target_counts = []
                for (_planned_index, _source, rng), future in group:
                    wait_started = time.perf_counter()
                    (index, source, xyz_all, valid_all, source_rgb_np, camera), task_seconds = future.result()
                    geometry_wait_seconds += time.perf_counter() - wait_started
                    geometry_task_max_seconds = max(geometry_task_max_seconds, task_seconds)
                    targets = (sample_eligible_targets(valid_all, k, rng, diagonal_source=source)
                               if camera_cfg else sample_eligible_targets(valid_all, k, rng))
                    batch_values.append((
                        index, source, targets, xyz_all[targets], valid_all[targets],
                        source_rgb_np, camera, valid_all,
                    ))
                    target_counts.append(len(targets))
                if len(set(target_counts)) != 1:
                    raise ValueError(
                        f"microbatch clips have different eligible target counts: {target_counts}"
                    )
                trace_phase("geometry_ready", step, micro)
                latent_started = time.perf_counter()
                latents_np = np.stack([dataset.clean_latent(value[0]) for value in batch_values])
                latent_load_seconds += time.perf_counter() - latent_started
                normalized_np = np.stack([
                    (value[3] - mean[None, :, None, None]) / scale[None, :, None, None]
                    for value in batch_values
                ])
                valid_np = np.stack([value[4] for value in batch_values])
                trace_phase("batch_to_device_start", step, micro)
                source_rgb_t = None
                if use_source_rgb:
                    source_rgb_np = np.stack([value[5] for value in batch_values])
                    if source_rgb_np.shape != (len(batch_values), image_size, image_size, 3) \
                            or source_rgb_np.dtype != np.uint8:
                        raise RuntimeError(
                            f"source RGB batch must be uint8 [B,{image_size},{image_size},3], got "
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
                camera_tensors = (supervision_camera_batch([v[6] for v in batch_values], device)
                                  if camera_cfg else None)
                sync = fsdp.no_sync() if micro + 1 < accumulation else nullcontext()
                with sync, torch.autocast("cuda", dtype=dtype):
                    trace_phase("forward_start", step, micro)
                    prediction, z4d, model_output = fsdp(
                        latent, source_t, target_t, condition, source_rgb_t,
                    )
                    trace_phase("forward_enqueued", step, micro)
                    if config.get('native_kubric512_b1_a4_k15', False) or config.get('native_kubric512_b1_a4_k9', False) or config.get('native_kubric512_b1_a4_k5', False) or config.get('native_kubric512_full', False):
                        if prediction.shape != xyz.shape or tuple(prediction.shape[-2:]) != (image_size, image_size):
                            raise RuntimeError('native prediction/GT alignment mismatch')
                        if micro == 0:
                            print(json.dumps({'event': 'NATIVE_TRAIN_SHAPES', 'rank': rank,
                                'update_number': step + 1, 'dataset': name, 'latent': list(latent.shape),
                                'RGB': list(source_rgb_t.shape), 'prediction_GT': list(prediction.shape),
                                'dense': list(z4d.dense.shape), 'B': microbatch_per_gpu,
                                'A': accumulation, 'K': k, 'world': world}), flush=True)
                    with torch.set_grad_enabled(camera_cfg is None):
                        xyz_value = masked_pair_smooth_l1(
                            prediction.float(), xyz.float(), valid,
                            beta=float(config.get("smooth_l1_beta", 0.05)),
                        )
                    optimized_xyz_value = xyz_value
                    if camera_cfg:
                        with torch.autocast('cuda', enabled=False):
                            optimized_xyz_value, camera_losses = camera_ray_objective(
                                prediction, xyz, valid, source_t, target_t, model_output.camera,
                                camera_tensors, mean, scale, camera_cfg,
                                beta=float(config.get('smooth_l1_beta', 0.05)), dataset=name,
                            )
                        for metric_name, metric_value in camera_losses.items():
                            camera_metrics[metric_name] = camera_metrics.get(metric_name, 0.0) + float(metric_value.detach()) / accumulation
                    # Keep xyz_value UNWEIGHTED for cycle diagnostics and all
                    # historical train/xyz_loss comparisons.
                    loss = optimized_xyz_value / accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step={step}, micro={micro}")
                trace_phase("backward_start", step, micro)
                loss.backward()
                trace_phase("backward_enqueued", step, micro)
                update_loss += float(loss.detach())
                xyz_loss_sum += float(xyz_value.detach()) / accumulation
                with torch.no_grad():
                    metric_error = (prediction.float() - xyz) * torch.as_tensor(scale, device=device).view(1, 1, 3, 1, 1)
                    epe = torch.linalg.vector_norm(metric_error, dim=2)
                    update_epe += float(epe[valid].sum())
                    valid_points += int(valid.sum())
                pair_count += sum(target_counts)
                for _index, source, targets, _xyz, _valid, _source_rgb, _camera, _valid_all in batch_values:
                    source_hist[source] += 1
                    for target_index in targets.tolist():
                        target_hist[target_index] += 1
                        gap_hist[abs(int(target_index) - source)] += 1
                if camera_cfg:
                    del prediction, z4d, model_output, camera_losses
            gradient_norm = fsdp.clip_grad_norm_(float(config["gradient_clip"]))
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite gradient norm at step={step + 1}")
            lr_factor = (
                apply_lr_restart_schedule(optimizer, step + 1, lr_restart)
                if lr_restart is not None else apply_cosine_schedule(
                    optimizer,
                    step + 1,
                    int(config["warmup_steps"]),
                    int(config["schedule_horizon_steps"]),
                )
            )
            if camera_cfg:
                apply_fresh_group_warmup(optimizer, {'camera_head'}, step+1,
                    int(camera_cfg['phase_start_step']), int(camera_cfg['warmup_steps']))
            warmup_groups = (
                {"wan_backbone", "dense_decoder"}
                if str(config.get("trainable_mode")) == "source_rgb_plus_wan_decoder"
                else ({"dense_decoder"} if str(config.get("trainable_mode")) == "decoder_only" else set())
            )
            fresh_group_warmup_factor = apply_fresh_group_warmup(
                optimizer,
                warmup_groups,
                step + 1,
                start_step,
                int(config.get("joint_fresh_group_warmup_steps", 0)),
                float(config.get("joint_fresh_group_max_lr_scale", 1.0)),
            ) if warmup_groups and lr_restart is None else 1.0
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
                    [update_loss, xyz_loss_sum, update_epe, valid_points, pair_count],
                    device=device, dtype=torch.float64,
                )
                timing_max = torch.tensor([
                    geometry_wait_seconds, geometry_task_max_seconds, latent_load_seconds,
                ], device=device, dtype=torch.float64)
                dist.all_reduce(scalars, op=dist.ReduceOp.SUM)
                dist.all_reduce(timing_max, op=dist.ReduceOp.MAX)
                if camera_cfg:
                    camera_metric_names = sorted(camera_metrics)
                    camera_metric_values = torch.tensor([camera_metrics[n] for n in camera_metric_names],
                                                        device=device, dtype=torch.float64)
                    dist.all_reduce(camera_metric_values, op=dist.ReduceOp.SUM)
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
                (
                    global_loss, global_xyz_loss, global_epe_sum, global_valid, global_pairs,
                ) = scalars.tolist()
                dataset_loss = global_loss / world
                dataset_xyz_loss = global_xyz_loss / world
                raw_epe_m = global_epe_sum / max(global_valid, 1)
                weights = fsdp.module.backbone.layer_weights().detach().float().cpu().tolist()
                rgb_alphas = {
                    scale_name: float(rgb_alpha_stats[index, 0] / rgb_alpha_stats[index, 1])
                    for index, scale_name in enumerate(rgb_alpha_names)
                }
                payload = {
                    "global_step": completed, "train/loss": dataset_loss,
                    f"train/loss_by_dataset/{name}": dataset_loss,
                    "train/xyz_loss": dataset_xyz_loss,
                    f"train/xyz_loss_by_dataset/{name}": dataset_xyz_loss,
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
                if camera_cfg:
                    for key, value in zip(camera_metric_names, camera_metric_values.tolist()):
                        payload[f'train/camera/{key}'] = value/world
                        payload[f'train/camera_by_dataset/{name}/{key}'] = value/world
                    payload['train/camera_head_lr'] = next(g['lr'] for g in optimizer.param_groups if g['name']=='camera_head')
                    payload['train/diagonal_pairs'] = world*accumulation*microbatch_per_gpu
                    payload['train/targets_per_source'] = k
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
            final = completed == execution_end or bool(stop_tensor.item())
            if not args.no_checkpoint and (periodic or final):
                state = {
                    "global_step": completed, "clips_seen": clips_seen,
                    "dataset_cycle_offset": completed % 20,
                    "prompt_metadata": prompt_metadata,
                }
                if config.get('dataset_mix_counts') is not None:
                    state['dataset_mix_counts'] = dataset_mix_counts
                    state['dataset_mix_phase_origin'] = int(config['selected_checkpoint_step'])
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
            if empty_cache_every and completed % empty_cache_every == 0:
                trace_phase("cache_release_start", step)
                # Includes temporary full-state buffers on checkpoint updates.
                # Never changes live tensors, gradients, optimizer, or RNG.
                torch.cuda.empty_cache()
                trace_phase("cache_release_complete", step)
            if stall_seconds > 0:
                faulthandler.cancel_dump_traceback_later()
            if bool(stop_tensor.item()):
                break
    finally:
        if stall_seconds > 0:
            faulthandler.cancel_dump_traceback_later()
        prefetcher.close()
        if run is not None:
            run.finish()
    dist.barrier()
    if rank == 0:
        result = {
            "completed_steps": completed, "target_steps": target_steps, "world_size": world,
            "execution_end": execution_end, "diagnostic_stop_after_updates": args.stop_after_updates,
            "startup_preflight": startup_preflight,
            "clips_seen": clips_seen, "non_wan_parameters": non_wan_count,
            "dataset_mix_counts": dataset_mix_counts,
            "targets_per_source": k, "peak_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "elapsed_seconds": time.perf_counter() - started,
            "source_rgb_pyramid": use_source_rgb,
            "trainable_mode": str(config.get("trainable_mode")),
            "trainable_parameters": trainable_count,
        }
        atomic_json(output / "train_status.json", result)
        print(json.dumps(result, indent=2), flush=True)
        print("THREE_DATASET_256_FSDP_OK", flush=True)
    dist.barrier(); dist.destroy_process_group()


class WorldBridgeTrainer:
    """Explicit training entry point used by the thin CLI wrapper."""

    def fit(self) -> None:
        main()
