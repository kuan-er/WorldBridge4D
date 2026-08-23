#!/usr/bin/env python3
"""Four-rank FULL_SHARD training for the 256px/200M three-dataset route."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
from contextlib import nullcontext
import datetime as dt
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import threading
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
    training_diagnostic_due,
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


def clip_optimizer_grad_norm_(
    optimizer: torch.optim.Optimizer, max_norm: float, device: torch.device,
) -> torch.Tensor:
    """Collectively clip optimizer-owned sharded gradients on every rank.

    With ``use_orig_params=True`` a small trainable suffix can reside entirely
    on one rank while another rank has no local gradients. FSDP's convenience
    clipper returns early on the empty rank, which leaves the owner blocked in
    its collective. This helper always participates in one all-reduce.
    """
    max_norm = float(max_norm)
    if max_norm <= 0:
        raise ValueError("gradient clip norm must be positive")
    local_squared = torch.zeros((), device=device, dtype=torch.float64)
    gradients = []
    seen: set[int] = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            gradient = parameter.grad
            if gradient is None:
                continue
            gradients.append(gradient)
            local_squared += gradient.detach().double().square().sum()
    dist.all_reduce(local_squared, op=dist.ReduceOp.SUM)
    total_norm = local_squared.sqrt()
    coefficient = torch.clamp(
        torch.as_tensor(max_norm, device=device, dtype=torch.float64)
        / (total_norm + 1e-6),
        max=1.0,
    )
    for gradient in gradients:
        gradient.mul_(coefficient.to(dtype=gradient.dtype))
    return total_norm.float()


def validate_config(config: dict[str, Any], world: int, allow_two_gpu: bool,
                    allow_four_gpu_experiment: bool = False,
                    allow_arbitrary_world: bool = False) -> None:
    expected = {
        "image_size": 256, "clip_length": 21, "latent_spatial_size": 32,
        "query_dim": 1536, "embedding_dim": 768, "num_cross_attn_layers": 5,
        "num_heads": 12, "geometry_dim": 512, "geometry_spatial_size": 32,
        "motion_slots": 8,
    }
    mismatches = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if mismatches:
        raise ValueError(f"256/200M frozen configuration mismatch: {mismatches}")
    if bool(config.get("source_rgb_pyramid", False)):
        if list(config.get("source_rgb_channels", [])) != [32, 64, 128]:
            raise ValueError("source RGB pyramid fixes channels to [32,64,128]")
        expected_rgb_scales = (
            [32, 64, 128, 256]
            if bool(config.get("source_rgb_fusion_32", False))
            else [64, 128, 256]
        )
        if list(config.get("source_rgb_fusion_scales", [])) != expected_rgb_scales:
            raise ValueError(
                f"source RGB pyramid fixes fusion scales to {expected_rgb_scales}"
            )
        if not bool(config.get("source_rgb_zero_init", False)):
            raise ValueError("source RGB pyramid requires zero-initialized residual gates")
        if not config.get("source_rgb_cache_root"):
            raise ValueError("source RGB pyramid requires source_rgb_cache_root")
        if int(config.get("source_rgb_cache_max_open_shards", 0)) < 1:
            raise ValueError("source RGB mmap cache size must be positive")
        if bool(config.get("finetune_allow_new_source_rgb_fusion_32", False)) \
                and not bool(config.get("source_rgb_fusion_32", False)):
            raise ValueError("32px structural migration requires source_rgb_fusion_32")
    wan_num_layers = int(config.get("wan_num_layers", 30))
    if wan_num_layers < 16:
        raise ValueError("wan_num_layers must include readout blocks 13,14,15")
    expected_hidden_layers = [13, 14, 15, wan_num_layers - 1]
    if list(config.get("wan_hidden_layers", [])) != expected_hidden_layers:
        raise ValueError(
            f"wan_hidden_layers must include the model's final block: {expected_hidden_layers}"
        )
    mode = str(config.get("trainable_mode", "full"))
    allowed_modes = {
        "full", "lora", "source_rgb_only", "source_rgb_plus_wan_decoder",
    }
    if mode not in allowed_modes:
        raise ValueError(
            "three-dataset training supports full, lora, and audited source-RGB phases"
        )
    if mode.startswith("source_rgb_"):
        if not bool(config.get("source_rgb_pyramid", False)):
            raise ValueError("source-RGB trainable modes require the RGB pyramid")
        if not bool(config.get("source_rgb_separate_optimizer_group", False)):
            raise ValueError("source-RGB trainable modes require separate optimizer groups")
        multiplier = float(config.get("source_rgb_learning_rate_multiplier", 0.0))
        if multiplier not in {3.0, 10.0}:
            raise ValueError("source-RGB warm-up LR multiplier must be exactly 3 or 10")
        if not allow_two_gpu or world != 2:
            raise ValueError("source-RGB warm-up is registered only as a two-GPU experiment")
    if mode == "source_rgb_plus_wan_decoder":
        if int(config.get("joint_fresh_group_warmup_steps", 0)) < 0:
            raise ValueError("joint fresh-group warm-up steps must be non-negative")
        max_scale = float(config.get("joint_fresh_group_max_lr_scale", 1.0))
        if not 0.0 < max_scale <= 1.0:
            raise ValueError("joint fresh-group maximum LR scale must be in (0,1]")
    if mode == "lora":
        if int(config.get("lora_rank", 0)) <= 0:
            raise ValueError("lora_rank must be positive")
        truncate = config.get("wan_truncate_after_block")
        if truncate is not None and int(truncate) < max(config["wan_hidden_layers"]):
            raise ValueError("Wan truncation cannot precede the highest readout layer")
    logits = np.asarray(config.get("layer_gate_initial_logits"), dtype=np.float64)
    expected_logits = np.array([0.0, 0.0, 0.0, -1.0986122887])
    if logits.shape != (4,) or not np.allclose(logits, expected_logits, atol=1e-10):
        raise ValueError(f"incorrect readout initialization: {logits}")
    weights = np.exp(logits - logits.max()); weights /= weights.sum()
    if not np.allclose(weights, [0.3, 0.3, 0.3, 0.1], atol=1e-8):
        raise ValueError(f"incorrect initial layer weights: {weights}")
    two_gpu_experiment = allow_two_gpu and world == 2
    four_gpu_experiment = allow_four_gpu_experiment and world == 4
    # Arbitrary-world mode relaxes the fixed 2/4-rank contract.  It still
    # requires the (2,2) accumulation/microbatch whose slot plan matches the
    # already-precomputed latents (precomputed under world=5, covering ranks
    # 0..4), so world must not exceed the precompute world size.
    arbitrary_world = allow_arbitrary_world and 1 <= world <= 5
    if not arbitrary_world and world != 4 and not two_gpu_experiment:
        raise ValueError(f"formal training requires 4 ranks; got {world} (use --allow-two-gpu-gate only for the gate)")
    if allow_four_gpu_experiment and world != 4:
        raise ValueError("--allow-four-gpu-experiment requires exactly 4 ranks")
    accumulation = int(config.get("gradient_accumulation", 0))
    microbatch = int(config.get("microbatch_per_gpu", 0))
    if accumulation < 1 or microbatch < 1:
        raise ValueError("gradient_accumulation and microbatch_per_gpu must be positive")
    if arbitrary_world:
        allowed_batch_modes = {(2, 2)}
        if world == 1:
            # Same four clips/update as B2/A2, but half the live decoder batch
            # for the replicated full-depth 14B single-card capacity gate.
            allowed_batch_modes.add((4, 1))
        if (accumulation, microbatch) not in allowed_batch_modes:
            raise ValueError(
                "--allow-arbitrary-world batch mode must preserve a registered "
                f"capacity protocol; allowed={sorted(allowed_batch_modes)}"
            )
    elif (accumulation, microbatch) != (2, 1):
        two_gpu_modes = ((1, 2), (2, 2))
        four_gpu_modes = ((1, 2),)
        valid_experiment = (
            (two_gpu_experiment and (accumulation, microbatch) in two_gpu_modes)
            or (four_gpu_experiment and (accumulation, microbatch) in four_gpu_modes)
        )
        if not valid_experiment:
            raise ValueError(
                "microbatch=2 requires an explicit compatible two- or four-GPU experiment flag"
            )
    k = int(config["targets_per_source"])
    if arbitrary_world:
        allowed_k = (8, 16)
    elif two_gpu_experiment:
        allowed_k = (4, 6, 10, 16, 19, 21)
    elif four_gpu_experiment:
        allowed_k = (4, 6, 10, 16)
    else:
        allowed_k = (4, 6)
    if k not in allowed_k:
        raise ValueError(
            f"targets_per_source must be one of {allowed_k} for this launch mode"
        )
    prefetch_depth = int(config.get("geometry_prefetch_depth", 2))
    prefetch_workers = int(config.get(
        "geometry_prefetch_workers", min(4, accumulation * microbatch * 2),
    ))
    if not 1 <= prefetch_depth <= 16:
        raise ValueError("geometry_prefetch_depth must be in [1,16]")
    if not 1 <= prefetch_workers <= 32:
        raise ValueError("geometry_prefetch_workers must be in [1,32]")
    kubric = config.get("datasets", {}).get("kubric", {})
    sample_cache_size = int(kubric.get("geometry_sample_cache_size", 16))
    if not 1 <= sample_cache_size <= 256:
        raise ValueError("geometry_sample_cache_size must be in [1,256]")
    max_open_shards = kubric.get("geometry_mmap_max_open_shards")
    if max_open_shards is not None and not 1 <= int(max_open_shards) <= 4096:
        raise ValueError("geometry_mmap_max_open_shards must be in [1,4096]")
    diagnostic_every = int(config.get("diagnostic_every_steps", 20))
    if diagnostic_every < 1:
        raise ValueError("diagnostic_every_steps must be positive")
    extension_start = config.get("schedule_extension_start_step")
    extension_horizon = config.get("schedule_extension_horizon_steps")
    if (extension_start is None) != (extension_horizon is None):
        raise ValueError("both cosine schedule extension fields must be configured")
    if extension_start is not None:
        warmup = int(config["warmup_steps"])
        original_horizon = int(config["schedule_horizon_steps"])
        extension_start = int(extension_start)
        extension_horizon = int(extension_horizon)
        if not warmup < extension_start < original_horizon < extension_horizon:
            raise ValueError(
                "schedule extension must satisfy warmup < start < original horizon < extended horizon"
            )
        if int(config.get("max_steps", extension_horizon)) != extension_horizon:
            raise ValueError("extended schedule horizon must equal max_steps")


def wan_block_auto_wrap_policy(
    module: torch.nn.Module, recurse: bool, nonwrapped_numel: int,
) -> bool:
    """Shard Wan blocks, but never a parent whose custom forward is bypassed.

    Structured hidden extraction calls ``dit.patch_embedding`` and individual
    blocks directly instead of calling ``dit.forward``. Wrapping the parent
    Wan transformer would leave its flat parameters sharded during those direct
    calls. Each transformer block is invoked normally, so it is the safe FSDP
    unit for the LoRA route.
    """
    if recurse:
        return True
    return module.__class__.__name__ == "WanTransformerBlock"


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while value := stream.read(chunk):
            digest.update(value)
    return digest.hexdigest()


def required_latent_requests(datasets: dict[str, Any], seed: int, start_step: int,
                             target_steps: int, rank: int, accumulation: int,
                             microbatch_per_gpu: int = 1
                             ) -> list[tuple[int, str, int]]:
    """Return rank-local requests in update order, including repeated clips."""
    requests = []
    slots_per_rank = int(accumulation) * int(microbatch_per_gpu)
    for step in range(int(start_step), int(target_steps)):
        name = dataset_for_step(step, seed)
        dataset = datasets[name]
        for slot in range(slots_per_rank):
            index, _, _ = deterministic_sample_plan(
                dataset, name, seed, step, slot, rank, slots_per_rank
            )
            requests.append((step, name, int(index)))
    return requests


def required_latent_indices(datasets: dict[str, Any], seed: int, start_step: int,
                            target_steps: int, rank: int, accumulation: int,
                            microbatch_per_gpu: int = 1
                            ) -> dict[str, list[int]]:
    required: dict[str, set[int]] = {name: set() for name in DATASET_NAMES}
    for _step, name, index in required_latent_requests(
        datasets, seed, start_step, target_steps, rank, accumulation,
        microbatch_per_gpu,
    ):
        required[name].add(index)
    return {name: sorted(indices) for name, indices in required.items()}


def lazy_latent_owner(name: str, index: int, world: int) -> int:
    """Assign a dataset clip to one rank with a stable, balanced hash."""
    dataset_id = DATASET_NAMES.index(str(name))
    return (int(index) * 1_315_423_911 + dataset_id * 2_654_435_761) % int(world)


def pipeline_work_for_rank(gathered_requests: list[list[tuple[int, str, int]]],
                           rank: int, world: int) -> list[tuple[int, str, int]]:
    """Deduplicate globally and balance VAE work with a stable clip hash."""
    earliest: dict[tuple[str, int], int] = {}
    for requests in gathered_requests:
        for step, name, index in requests:
            key = (str(name), int(index))
            earliest[key] = min(int(step), earliest.get(key, int(step)))
    owned = []
    for (name, index), step in earliest.items():
        if lazy_latent_owner(name, index, world) == int(rank):
            owned.append((step, name, index))
    return sorted(owned, key=lambda value: (value[0], DATASET_NAMES.index(value[1]), value[2]))


def vae_checkpoint_identity(config: dict[str, Any]) -> tuple[Path, str]:
    checkpoint = Path(config.get(
        "vae_checkpoint", Path(config["wan_root"]) / "Wan2.1_VAE.pth"
    )).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"WAN VAE checkpoint missing: {checkpoint}")
    return checkpoint, sha256(checkpoint)


def set_lazy_vae_identity(datasets: dict[str, Any], checksum: str) -> None:
    for dataset in datasets.values():
        dataset.set_lazy_vae_sha256(checksum)


def warm_lazy_latents(config: dict[str, Any], datasets: dict[str, Any],
                      required: dict[str, list[int]], device: torch.device,
                      rank: int) -> dict[str, int]:
    """Populate only this rank's planned cache misses, then release the VAE."""
    checkpoint, checksum = vae_checkpoint_identity(config)
    set_lazy_vae_identity(datasets, checksum)
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


class LazyVAEPipeline:
    """Background VAE producer sharing a GPU with training on its own stream."""

    def __init__(self, config: dict[str, Any], work: list[tuple[int, str, int]],
                 device: torch.device, rank: int, output: Path) -> None:
        self.config = config
        self.work = list(work)
        self.device = device
        self.rank = int(rank)
        self.output = output
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.error: BaseException | None = None
        self.position = 0
        self.counts = {
            "assigned": len(self.work), "cached": 0, "written": 0,
            "reused_after_wait": 0,
        }
        checkpoint, self.checksum = vae_checkpoint_identity(config)
        # A separate dataset instance prevents the producer's RGB reads from
        # racing the geometry reader's mutable per-dataset caches.
        self.datasets = load_training_datasets(config, allow_missing_latents=True)
        set_lazy_vae_identity(self.datasets, self.checksum)
        self.encoder = WanVAEEncoder(
            checkpoint, device=device, dtype=torch.float32,
            expected_shape=WAN_LATENT_SHAPE_256,
        )
        self.stream = torch.cuda.Stream(device=device)
        self.stream.wait_stream(torch.cuda.current_stream(device))

    def _produce(self, step: int, name: str, index: int) -> None:
        dataset = self.datasets[name]
        if dataset.latent_cached(index):
            self.counts["cached"] += 1
            return
        rgb = torch.from_numpy(np.asarray(dataset.rgb(index))).permute(0, 3, 1, 2)[None]
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            value = self.encoder(rgb).float().cpu().numpy()[0]
        if dataset.cache_latent(index, value, self.checksum):
            self.counts["written"] += 1
        else:
            self.counts["reused_after_wait"] += 1
        print(json.dumps({
            "event": "lazy_vae_pipeline_clip", "rank": self.rank,
            "needed_by_step": step, "dataset": name, "index": index,
            "position": self.position + 1, **self.counts,
        }), flush=True)

    def warm_through(self, exclusive_step: int) -> None:
        while self.position < len(self.work) and self.work[self.position][0] < int(exclusive_step):
            self._produce(*self.work[self.position])
            self.position += 1
        print(json.dumps({
            "event": "lazy_vae_pipeline_warm", "rank": self.rank,
            "exclusive_step": int(exclusive_step), "remaining": len(self.work) - self.position,
            **self.counts,
        }), flush=True)

    def _record_error(self, exc: BaseException) -> None:
        self.error = exc
        value = {"rank": self.rank, "type": type(exc).__name__, "message": str(exc)}
        try:
            atomic_json(self.output / f"pipeline_error_rank{self.rank}.json", value)
        finally:
            print(json.dumps({"event": "lazy_vae_pipeline_error", **value}), flush=True)

    def _run(self) -> None:
        try:
            while self.position < len(self.work) and not self.stop_event.is_set():
                self._produce(*self.work[self.position])
                self.position += 1
            if self.position == len(self.work):
                self.stream.synchronize()
                self.encoder = None
                print(json.dumps({
                    "event": "lazy_vae_pipeline_complete", "rank": self.rank,
                    **self.counts,
                }), flush=True)
        except BaseException as exc:
            self._record_error(exc)

    def start(self) -> None:
        self.thread = threading.Thread(
            target=self._run, name=f"lazy-vae-rank-{self.rank}", daemon=True,
        )
        self.thread.start()

    def wait_for(self, dataset: Any, name: str, indices: list[int],
                 timeout_seconds: float) -> None:
        deadline = time.monotonic() + float(timeout_seconds)
        missing = set(map(int, indices))
        while missing:
            missing = {index for index in missing if not dataset.latent_cached(index)}
            if not missing:
                return
            if self.error is not None:
                raise RuntimeError("local lazy VAE pipeline failed") from self.error
            errors = sorted(self.output.glob("pipeline_error_rank*.json"))
            if errors:
                raise RuntimeError(f"lazy VAE pipeline failed: {errors[0].read_text().strip()}")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for pipeline latent {name}/{sorted(missing)}"
                )
            time.sleep(0.05)

    def snapshot(self) -> dict[str, int]:
        return {**self.counts, "position": self.position, "remaining": len(self.work) - self.position}

    def close(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=600)
            if self.thread.is_alive():
                raise RuntimeError("lazy VAE pipeline did not stop within 600 seconds")
        self.encoder = None


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
    run.define_metric("train/loss_by_dataset/*", step_metric="global_step")
    run.define_metric("train/raw_epe_m_by_dataset/*", step_metric="global_step")
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


def update_latest_checkpoint(checkpoint: Path, latest: Path) -> None:
    """Atomically point ``latest`` at an immutable same-filesystem checkpoint."""
    latest.parent.mkdir(parents=True, exist_ok=True)
    temporary = latest.with_name(f".{latest.name}.{os.getpid()}.link.tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(checkpoint, temporary)
        temporary.replace(latest)
    finally:
        temporary.unlink(missing_ok=True)


def launch_durable_checkpoint_replica(source: Path, destination: Path,
                                      step: int) -> bool:
    """Start a detached, best-effort SSD-to-HDD replica without blocking training."""
    log_path = source.parent / "durable_replication.log"
    command = [
        sys.executable, str(ROOT / "scripts" / "replicate_checkpoint.py"),
        "--source", str(source), "--destination", str(destination),
        "--step", str(int(step)),
    ]
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab", buffering=0) as stream:
            subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, close_fds=True, start_new_session=True,
            )
        print(json.dumps({
            "event": "durable_checkpoint_queued", "step": int(step),
            "source": str(source), "destination": str(destination),
            "log": str(log_path),
        }), flush=True)
        return True
    except Exception as error:
        print(json.dumps({
            "event": "durable_checkpoint_queue_failed", "step": int(step),
            "source": str(source), "destination": str(destination),
            "error": repr(error),
        }), file=sys.stderr, flush=True)
        return False


def launch_post_resume_checksum(marker: Path, resume: Path, step: int) -> None:
    """Hash an already strictly restored immutable checkpoint in the background."""
    marker = marker.resolve()
    value = json.loads(marker.read_text())
    marked_checkpoint = Path(value["checkpoint"]).resolve()
    if marked_checkpoint != resume.resolve():
        raise ValueError(
            f"post-resume checksum marker checkpoint mismatch: {marked_checkpoint} != {resume.resolve()}"
        )
    log_path = marker.parent / "post_resume_checksum.log"
    command = [
        "ionice", "-c", "3", "nice", "-n", "19", sys.executable,
        str(ROOT / "scripts" / "prepare_three_dataset_256_gpu14_handoff.py"),
        "--finalize-checksum-marker", str(marker),
    ]
    with log_path.open("ab", buffering=0) as stream:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            close_fds=True, start_new_session=True,
        )
    print(json.dumps({
        "event": "post_resume_checksum_started", "step": int(step),
        "pid": process.pid, "marker": str(marker), "log": str(log_path),
    }), flush=True)


def prune_periodic_checkpoints(output: Path, keep_last: int) -> list[str]:
    """Bound disk use while retaining latest.pt and the newest named milestones."""
    paths = sorted(output.glob("checkpoint-*.pt"))
    remove = paths[:-max(int(keep_last), 0)] if keep_last else paths
    removed = []
    for path in remove:
        path.unlink(missing_ok=True)
        removed.append(path.name)
    return removed


def load_model_state_for_resume(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
    allowed_missing_prefixes: tuple[str, ...] = (),
) -> list[str]:
    """Load an exact state, or one audited zero-init structural extension."""
    if not allowed_missing_prefixes:
        model.load_state_dict(state, strict=True)
        return []
    missing, unexpected = model.load_state_dict(state, strict=False)
    invalid_missing = [
        name for name in missing
        if not name.startswith(allowed_missing_prefixes)
    ]
    if unexpected or invalid_missing or not missing:
        raise RuntimeError(
            "structural checkpoint migration mismatch; "
            f"missing={missing[:8]}, invalid_missing={invalid_missing[:8]}, "
            f"unexpected={unexpected[:8]}"
        )
    return list(missing)


def load_unwrapped_model_checkpoint(
    path: Path, model: torch.nn.Module, rank: int, world: int,
    allowed_missing_prefixes: tuple[str, ...] = (),
) -> tuple[dict[str, Any] | None, dict[str, Any], list[Any]]:
    """Load rank 0 before FSDP so ``sync_module_states`` broadcasts exact weights.

    Loading a rank-0-only FULL_STATE_DICT after FSDP construction does not
    broadcast it: nonzero ranks that receive an empty state dict retain their
    construction weights. The subsequent all-reduce then trains a hybrid model.
    """
    payload = torch.load(
        path, map_location="cpu", mmap=True, weights_only=True,
    ) if rank == 0 else None
    metadata = [(
        int(payload["training_state"].get("world_size", world)),
        payload["training_state"], payload["training_state"].get("rng_states", []),
    ) if rank == 0 else None]
    dist.broadcast_object_list(metadata, src=0)
    saved_world, training_state, states = metadata[0]
    if saved_world != world:
        raise ValueError(f"exact resume world-size mismatch: checkpoint={saved_world}, current={world}")
    if len(states) != world:
        raise ValueError("checkpoint lacks one RNG state per rank")
    if rank == 0:
        missing = load_model_state_for_resume(
            model, payload["model"], allowed_missing_prefixes,
        )
        if missing:
            print(json.dumps({
                "event": "structural_model_extension_loaded",
                "fresh_parameters": len(missing),
                "allowed_prefixes": list(allowed_missing_prefixes),
            }), flush=True)
    return payload, training_state, states


def load_initial_model_weights(
    path: Path,
    model: torch.nn.Module,
    rank: int,
    expected_global_step: int = 0,
    expected_clips_seen: dict[str, int] | None = None,
    expected_schedule: dict[str, Any] | None = None,
    restore_optimizer: bool = False,
) -> dict[str, Any] | None:
    """Initialize the RGB-pyramid route from an old strict model-only state.

    With ``restore_optimizer``, existing parameters retain their saved AdamW
    moments and step while the new RGB parameters receive empty, lazy optimizer
    state.  This is a structure-aware continuation, not a strict exact resume.
    """
    if rank != 0:
        return None
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    source_state = payload.get("training_state", {})
    if expected_global_step:
        source_step = int(source_state.get("global_step", -1))
        if source_step != int(expected_global_step):
            raise RuntimeError(
                f"initial model step {source_step} != expected {expected_global_step}"
            )
        source_clips = {
            key: int(value)
            for key, value in source_state.get("clips_seen", {}).items()
        }
        expected_clips = {
            key: int(value) for key, value in (expected_clips_seen or {}).items()
        }
        if source_clips != expected_clips:
            raise RuntimeError(
                f"initial model clip counters {source_clips} != expected {expected_clips}"
            )
        source_config = payload.get("config", {})
        schedule_mismatches = {
            key: (source_config.get(key), value)
            for key, value in (expected_schedule or {}).items()
            if source_config.get(key) != value
        }
        if schedule_mismatches:
            raise RuntimeError(
                f"initial model LR schedule mismatch: {schedule_mismatches}"
            )
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    allowed_prefixes = (
        "decoder.upsampler.source_rgb_encoder.",
        "decoder.upsampler.source_fusions.",
    )
    invalid_missing = [
        name for name in missing if not name.startswith(allowed_prefixes)
    ]
    if invalid_missing or unexpected:
        raise RuntimeError(
            "initial model checkpoint mismatch; "
            f"missing={invalid_missing[:8]}, unexpected={unexpected[:8]}"
        )
    if not missing:
        raise RuntimeError("initial model checkpoint already contains the RGB pyramid")
    print(json.dumps({
        "event": "initial_model_weights_loaded",
        "checkpoint": str(path.resolve()),
        "new_parameters": len(missing),
        "optimizer": (
            "checkpoint_plus_fresh_rgb" if restore_optimizer else "fresh"
        ),
        "rng": "restored" if restore_optimizer else "fresh",
        "global_step": int(expected_global_step),
        "clips_seen": expected_clips_seen or {},
    }), flush=True)
    return payload


def extend_optimizer_state_for_rgb(
    source: dict[str, Any], current_group_names: dict[str, list[str]],
) -> tuple[dict[str, Any], list[str]]:
    """Extend a name-keyed full AdamW state with only the new RGB parameters."""
    source_groups = {
        str(group.get("name")): group for group in source["param_groups"]
    }
    if set(source_groups) != set(current_group_names):
        raise RuntimeError(
            f"optimizer groups changed: {set(source_groups)} != {set(current_group_names)}"
        )
    merged_groups = []
    added_names: list[str] = []
    allowed_prefixes = (
        "decoder.upsampler.source_rgb_encoder.",
        "decoder.upsampler.source_fusions.",
    )
    for group_name, names in current_group_names.items():
        old_group = source_groups[group_name]
        old_names = list(old_group["params"])
        old_set = set(old_names)
        current_set = set(names)
        removed = sorted(old_set - current_set)
        added = [name for name in names if name not in old_set]
        invalid_added = [
            name for name in added if not name.startswith(allowed_prefixes)
        ]
        if removed or invalid_added:
            raise RuntimeError(
                f"optimizer parameter mismatch in {group_name}: "
                f"removed={removed[:8]}, invalid_added={invalid_added[:8]}"
            )
        if added and group_name != "dense_decoder":
            raise RuntimeError(
                f"new RGB optimizer parameters unexpectedly entered {group_name}"
            )
        merged_groups.append({**old_group, "params": list(names)})
        added_names.extend(added)
    if not added_names:
        raise RuntimeError("optimizer checkpoint already contains RGB parameters")
    return {
        "state": source["state"],
        "param_groups": merged_groups,
    }, added_names


def load_extended_optimizer_checkpoint(
    payload: dict[str, Any] | None,
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    states: list[Any],
    current_group_names: dict[str, list[str]],
    rank: int,
) -> None:
    full_optimizer_state = None
    added_names: list[str] = []
    if rank == 0:
        if payload is None:
            raise RuntimeError("rank zero lacks initial optimizer checkpoint")
        full_optimizer_state, added_names = extend_optimizer_state_for_rgb(
            payload["optimizer"], current_group_names,
        )
    optimizer_state = FSDP.scatter_full_optim_state_dict(
        full_optimizer_state, model, optim=optimizer,
    )
    optimizer.load_state_dict(optimizer_state)
    restore_rng_state(states[rank])
    if rank == 0:
        print(json.dumps({
            "event": "initial_optimizer_state_loaded",
            "restored_parameters": len(payload["optimizer"]["state"]),
            "fresh_rgb_parameters": len(added_names),
            "optimizer_step": int(payload["training_state"]["global_step"]),
            "rng_states": len(states),
        }), flush=True)


def filter_optimizer_state_for_trainable(
    source: dict[str, Any], current_groups: list[dict[str, Any]],
    *, allow_fresh_non_rgb: bool = False,
    allowed_fresh_rgb_prefixes: tuple[str, ...] = (),
) -> tuple[dict[str, Any], list[str]]:
    """Restore all available moments while strictly preserving RGB history.

    RGB-only warm-up requires every current tensor to have a source state. The
    following joint phase may add Wan and decoder tensors with fresh AdamW state,
    but all RGB tensors must still restore their accumulated moments.
    """
    prefixes = (
        "decoder.upsampler.source_rgb_encoder.",
        "decoder.upsampler.source_fusions.",
    )
    names = [str(name) for group in current_groups for name in group["params"]]
    if len(names) != len(set(names)):
        raise RuntimeError("fine-tune optimizer contains duplicate parameters")
    current = set(names)
    source_names = set(source["state"])
    invalid = [name for name in names if not name.startswith(prefixes)]
    if invalid and not allow_fresh_non_rgb:
        raise RuntimeError(
            f"RGB-only optimizer unexpectedly includes non-RGB parameters: {invalid[:8]}"
        )
    rgb_names = {name for name in current if name.startswith(prefixes)}
    missing_rgb = sorted(rgb_names - source_names)
    invalid_missing_rgb = [
        name for name in missing_rgb
        if not name.startswith(allowed_fresh_rgb_prefixes)
    ]
    if invalid_missing_rgb:
        raise RuntimeError(
            f"source checkpoint lacks RGB AdamW state: {invalid_missing_rgb[:8]}"
        )
    fresh = [name for name in names if name not in source_names]
    if fresh and not allow_fresh_non_rgb:
        raise RuntimeError(f"source checkpoint lacks RGB AdamW state: {fresh[:8]}")
    fresh_rgb = [name for name in fresh if name.startswith(prefixes)]
    invalid_fresh_rgb = [
        name for name in fresh_rgb
        if not name.startswith(allowed_fresh_rgb_prefixes)
    ]
    if invalid_fresh_rgb:
        raise RuntimeError(
            f"joint phase would reset RGB AdamW state: {invalid_fresh_rgb[:8]}"
        )
    restored = [name for name in names if name in source_names]
    return ({
        "state": {name: source["state"][name] for name in restored},
        "param_groups": current_groups,
    }, fresh)


def load_finetune_optimizer_checkpoint(
    payload: dict[str, Any] | None,
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    states: list[Any],
    current_group_names: dict[str, list[str]],
    rank: int,
    *,
    allow_fresh_non_rgb: bool = False,
    allowed_fresh_rgb_prefixes: tuple[str, ...] = (),
) -> None:
    full_optimizer_state = None
    restored = 0
    fresh = 0
    if rank == 0:
        if payload is None:
            raise RuntimeError("rank zero lacks the fine-tune source checkpoint")
        current_groups = []
        for group in optimizer.param_groups:
            group_name = str(group["name"])
            values = {key: value for key, value in group.items() if key != "params"}
            values["params"] = list(current_group_names[group_name])
            current_groups.append(values)
        full_optimizer_state, fresh_names = filter_optimizer_state_for_trainable(
            payload["optimizer"], current_groups,
            allow_fresh_non_rgb=allow_fresh_non_rgb,
            allowed_fresh_rgb_prefixes=allowed_fresh_rgb_prefixes,
        )
        restored = len(full_optimizer_state["state"])
        fresh = len(fresh_names)
    optimizer_state = FSDP.scatter_full_optim_state_dict(
        full_optimizer_state, model, optim=optimizer,
    )
    optimizer.load_state_dict(optimizer_state)
    restore_rng_state(states[rank])
    if rank == 0:
        print(json.dumps({
            "event": "finetune_optimizer_state_loaded",
            "restored_trainable_parameters": restored,
            "fresh_trainable_parameters": fresh,
            "optimizer_groups": [group["name"] for group in optimizer.param_groups],
            "rng_states": len(states),
        }), flush=True)


def load_optimizer_checkpoint(payload: dict[str, Any] | None, model: FSDP,
                              optimizer: torch.optim.Optimizer, states: list[Any],
                              rank: int) -> None:
    full_optimizer_state = payload["optimizer"] if rank == 0 else None
    optimizer_state = FSDP.scatter_full_optim_state_dict(
        full_optimizer_state, model, optim=optimizer,
    )
    optimizer.load_state_dict(optimizer_state)
    restore_rng_state(states[rank])


def apply_fresh_group_warmup(
    optimizer: torch.optim.Optimizer,
    group_names: set[str],
    update_number: int,
    start_step: int,
    warmup_steps: int,
    max_scale: float = 1.0,
) -> float:
    """Ramp newly unfrozen groups without reducing the mature RGB groups."""
    if not 0.0 < max_scale <= 1.0:
        raise ValueError("fresh-group maximum LR scale must be in (0,1]")
    if warmup_steps <= 0:
        factor = max_scale
    else:
        progress = int(update_number) - int(start_step)
        if progress <= 0:
            raise ValueError("fresh-group warm-up requires an update after the phase start")
        factor = max_scale * min(1.0, progress / int(warmup_steps))
    found: set[str] = set()
    for group in optimizer.param_groups:
        name = str(group.get("name"))
        if name in group_names:
            group["lr"] = float(group["lr"]) * factor
            found.add(name)
    if found != group_names:
        raise RuntimeError(f"fresh optimizer groups missing: {sorted(group_names - found)}")
    return factor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument(
        "--finetune-from",
        help="strictly load model/RNG and selected AdamW moments into a new optimizer phase",
    )
    parser.add_argument(
        "--init-model-weights",
        help="start a new trajectory from model weights only; permits only new RGB-pyramid parameters",
    )
    parser.add_argument("--checkpoint-dir", help="fast local checkpoint directory; defaults to output-dir")
    parser.add_argument("--durable-checkpoint", help="best-effort asynchronous replica path for latest checkpoint")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--allow-two-gpu-gate", action="store_true")
    parser.add_argument(
        "--allow-four-gpu-experiment", action="store_true",
        help="permit the explicit four-rank B2/K16 experimental route; not the formal K4/K6 contract",
    )
    parser.add_argument(
        "--allow-arbitrary-world", action="store_true",
        help="permit 1..5 ranks with the (2,2) microbatch/accumulation matching precomputed latents",
    )
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument(
        "--wandb-log-after-step", type=int, default=-1,
        help="initialize/resume W&B normally, but upload metrics only after this global step",
    )
    parser.add_argument("--checkpoint-at-end", action="store_true")
    parser.add_argument(
        "--post-resume-checksum-marker",
        help="after strict model/optimizer restore, hash the immutable resume checkpoint in background",
    )
    parser.add_argument(
        "--no-checkpoint", action="store_true",
        help="capacity-gate only: skip periodic/final checkpoint materialization",
    )
    parser.add_argument("--lazy-vae-cache", action="store_true",
                        help="encode/cache all planned missing latents before constructing FSDP")
    parser.add_argument("--lazy-vae-pipeline", action="store_true",
                        help="warm a short prefix, then encode future clips beside training")
    parser.add_argument("--pipeline-lookahead-steps", type=int, default=16)
    parser.add_argument("--source-rgb-lr-multiplier", type=float)
    args = parser.parse_args()
    initialization_modes = [args.resume, args.finetune_from, args.init_model_weights]
    if sum(value is not None for value in initialization_modes) > 1:
        parser.error("--resume, --finetune-from, and --init-model-weights are mutually exclusive")
    if args.lazy_vae_cache and args.lazy_vae_pipeline:
        parser.error("--lazy-vae-cache and --lazy-vae-pipeline are mutually exclusive")
    if args.pipeline_lookahead_steps < 1:
        parser.error("--pipeline-lookahead-steps must be positive")
    for value in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        signal.signal(value, stop_signal)
    config = yaml.safe_load(Path(args.config).read_text())
    if args.source_rgb_lr_multiplier is not None:
        config["source_rgb_learning_rate_multiplier"] = float(
            args.source_rgb_lr_multiplier
        )
    rank, world, local, device = initialize_distributed()
    validate_config(
        config, world, args.allow_two_gpu_gate, args.allow_four_gpu_experiment,
        args.allow_arbitrary_world,
    )
    seed = int(config.get("seed", 20260812))
    random.seed(seed + rank); np.random.seed(seed + rank); torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    output = Path(args.output_dir).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else output
    durable_checkpoint = Path(args.durable_checkpoint).resolve() if args.durable_checkpoint else None
    initial_model_weights = (
        Path(args.init_model_weights).resolve() if args.init_model_weights else None
    )
    finetune_from = Path(args.finetune_from).resolve() if args.finetune_from else None
    if finetune_from is not None:
        if not finetune_from.is_file():
            raise FileNotFoundError(finetune_from)
        if (checkpoint_dir / "latest.pt").is_file():
            raise ValueError(
                "fine-tune initialization refuses an existing implicit resume checkpoint"
            )
        config["finetune_from"] = str(finetune_from)
    initial_model_global_step = int(config.get("initial_model_global_step", 0))
    initial_model_restore_optimizer = bool(
        config.get("initial_model_restore_optimizer", False)
    )
    initial_model_clips_seen = {
        key: int(value)
        for key, value in config.get("initial_model_clips_seen", {}).items()
    }
    if initial_model_global_step < 0:
        raise ValueError("initial_model_global_step must be non-negative")
    if initial_model_global_step and initial_model_weights is None:
        raise ValueError("initial_model_global_step requires --init-model-weights")
    if initial_model_restore_optimizer and not initial_model_global_step:
        raise ValueError(
            "initial_model_restore_optimizer requires a positive initial model step"
        )
    if initial_model_global_step and set(initial_model_clips_seen) != set(DATASET_NAMES):
        raise ValueError(
            "initial_model_clips_seen must contain all three datasets"
        )
    if initial_model_weights is not None:
        if not bool(config.get("source_rgb_pyramid", False)):
            raise ValueError("--init-model-weights is reserved for the new source RGB route")
        if not initial_model_weights.is_file():
            raise FileNotFoundError(initial_model_weights)
        implicit_resume = checkpoint_dir / "latest.pt"
        if implicit_resume.is_file():
            raise ValueError(
                f"weights-only initialization refuses existing implicit resume: {implicit_resume}"
            )
        config["initial_model_weights"] = str(initial_model_weights)
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
    planning_checkpoint = finetune_from if finetune_from is not None else resume
    if rank == 0 and finetune_from is not None:
        # A fine-tune source is immutable but its parent trajectory may continue
        # updating train_status.json. Plan from the checkpoint-bound expected
        # step, then validate the actual full checkpoint after model construction.
        metadata[0] = int(config.get("finetune_expected_global_step", -1))
        if metadata[0] < 0:
            raise ValueError("fine-tune planning requires finetune_expected_global_step")
    elif rank == 0 and planning_checkpoint.is_file():
        status_path = Path(config.get(
            "resume_status_path", planning_checkpoint.parent / "train_status.json"
        ))
        if not status_path.is_file():
            raise FileNotFoundError(
                f"checkpoint planning sidecar missing: {status_path}; "
                "refusing a second full checkpoint read"
            )
        status = json.loads(status_path.read_text())
        metadata[0] = int(status["completed_steps"])
    dist.broadcast_object_list(metadata, src=0)
    planned_start = int(metadata[0] or initial_model_global_step)
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
    if initial_model_weights is not None or finetune_from is not None:
        load_wan_pretrained = False
    if rank == 0 and not load_wan_pretrained:
        if initial_model_weights is not None:
            load_event = "wan_pretrained_load_skipped_for_initial_weights"
            load_checkpoint = initial_model_weights
        elif finetune_from is not None:
            load_event = "wan_pretrained_load_skipped_for_finetune"
            load_checkpoint = finetune_from
        else:
            load_event = "wan_pretrained_load_skipped_for_full_resume"
            load_checkpoint = resume
        print(json.dumps({
            "event": load_event, "checkpoint": str(load_checkpoint),
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
    initial_payload: dict[str, Any] | None = None
    initial_rng_states: list[Any] = []
    finetune_payload: dict[str, Any] | None = None
    finetune_state: dict[str, Any] | None = None
    finetune_rng_states: list[Any] = []
    resume_payload: dict[str, Any] | None = None
    resume_state: dict[str, Any] | None = None
    resume_rng_states: list[Any] = []
    if initial_model_weights is not None:
        initial_payload = load_initial_model_weights(
            initial_model_weights,
            model,
            rank,
            expected_global_step=initial_model_global_step,
            expected_clips_seen=initial_model_clips_seen,
            expected_schedule={
                key: config.get(key) for key in (
                    "learning_rate",
                    "geometry_learning_rate",
                    "backbone_learning_rate",
                    "warmup_steps",
                    "schedule_horizon_steps",
                    "schedule_extension_start_step",
                    "schedule_extension_horizon_steps",
                )
            },
            restore_optimizer=initial_model_restore_optimizer,
        )
        if initial_model_restore_optimizer:
            initial_metadata = [(
                int(initial_payload["training_state"].get("world_size", world)),
                initial_payload["training_state"].get("rng_states", []),
            ) if rank == 0 else None]
            dist.broadcast_object_list(initial_metadata, src=0)
            saved_world, initial_rng_states = initial_metadata[0]
            if saved_world != world:
                raise ValueError(
                    f"initial optimizer world-size mismatch: "
                    f"checkpoint={saved_world}, current={world}"
                )
            if len(initial_rng_states) != world:
                raise ValueError("initial checkpoint lacks one RNG state per rank")
    if finetune_from is not None:
        allow_new_rgb_32 = bool(config.get(
            "finetune_allow_new_source_rgb_fusion_32", False,
        ))
        finetune_payload, finetune_state, finetune_rng_states = load_unwrapped_model_checkpoint(
            finetune_from, model, rank, world,
            allowed_missing_prefixes=(
                ("decoder.upsampler.source_fusions.32.",)
                if allow_new_rgb_32 else ()
            ),
        )
        expected_step = int(config.get("finetune_expected_global_step", -1))
        if expected_step < 0 or int(finetune_state["global_step"]) != expected_step:
            raise RuntimeError(
                f"fine-tune source step {finetune_state['global_step']} != expected {expected_step}"
            )
        expected_clips = {
            key: int(value)
            for key, value in config.get("finetune_expected_clips_seen", {}).items()
        }
        source_clips = {
            key: int(value) for key, value in finetune_state["clips_seen"].items()
        }
        if expected_clips != source_clips:
            raise RuntimeError(
                f"fine-tune source clip counters {source_clips} != expected {expected_clips}"
            )
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
    if trainable_mode == "source_rgb_only":
        prefixes = (
            "decoder.upsampler.source_rgb_encoder.",
            "decoder.upsampler.source_fusions.",
        )
        invalid = [name for name in trainable_names if not name.startswith(prefixes)]
        expected_trainable = int(config.get("expected_source_rgb_trainable_parameters", -1))
        if invalid or trainable_count != expected_trainable:
            raise RuntimeError(
                "source-RGB freeze audit failed: "
                f"invalid={invalid[:8]}, count={trainable_count}, expected={expected_trainable}"
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
    if str(config.get("trainable_mode", "full")) == "lora":
        auto_wrap = wan_block_auto_wrap_policy
    else:
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
    start_step = initial_model_global_step
    clips_seen = {name: 0 for name in DATASET_NAMES}
    if initial_model_global_step:
        clips_seen.update(initial_model_clips_seen)
    if initial_model_restore_optimizer:
        load_extended_optimizer_checkpoint(
            initial_payload,
            fsdp,
            optimizer,
            initial_rng_states,
            current_group_names,
            rank,
        )
        del initial_payload
    if finetune_state is not None:
        if bool(config.get("finetune_restore_trainable_optimizer", False)):
            load_finetune_optimizer_checkpoint(
                finetune_payload,
                fsdp,
                optimizer,
                finetune_rng_states,
                current_group_names,
                rank,
                allow_fresh_non_rgb=(
                    str(config.get("trainable_mode")) == "source_rgb_plus_wan_decoder"
                ),
                allowed_fresh_rgb_prefixes=(
                    ("decoder.upsampler.source_fusions.32.",)
                    if bool(config.get("finetune_allow_new_source_rgb_fusion_32", False))
                    else ()
                ),
            )
        else:
            restore_rng_state(finetune_rng_states[rank])
        start_step = int(finetune_state["global_step"])
        clips_seen.update({
            key: int(value) for key, value in finetune_state["clips_seen"].items()
        })
        del finetune_payload
        if rank == 0:
            print(json.dumps({
                "event": "finetune_state_loaded", "step": start_step,
                "world_size": world,
                "optimizer": (
                    "filtered_trainable_state"
                    if bool(config.get("finetune_restore_trainable_optimizer", False))
                    else "fresh"
                ),
                "rng_states": len(finetune_rng_states),
            }), flush=True)
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
            if args.post_resume_checksum_marker:
                launch_post_resume_checksum(
                    Path(args.post_resume_checksum_marker), resume, start_step,
                )
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
            if str(config.get("trainable_mode")) == "source_rgb_only":
                gradient_norm = clip_optimizer_grad_norm_(
                    optimizer, float(config["gradient_clip"]), device,
                )
            else:
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
            if not args.no_checkpoint and (
                periodic or final or (args.checkpoint_at_end and completed == target_steps)
            ):
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
            "initial_model_weights": (
                str(initial_model_weights) if initial_model_weights is not None else None
            ),
            "lazy_vae_cache": bool(args.lazy_vae_cache),
            "lazy_vae_pipeline": bool(args.lazy_vae_pipeline),
            "lazy_vae_rank0": lazy_counts,
        }
        atomic_json(output / "train_status.json", result)
        print(json.dumps(result, indent=2), flush=True)
        print("THREE_DATASET_256_FSDP_OK", flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
