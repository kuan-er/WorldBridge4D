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
    wan_num_layers = int(config.get("wan_num_layers", 30))
    if wan_num_layers < 16:
        raise ValueError("wan_num_layers must include readout blocks 13,14,15")
    expected_hidden_layers = [13, 14, 15, wan_num_layers - 1]
    if list(config.get("wan_hidden_layers", [])) != expected_hidden_layers:
        raise ValueError(
            f"wan_hidden_layers must include the model's final block: {expected_hidden_layers}"
        )
    mode = str(config.get("trainable_mode", "full"))
    if mode not in {"full", "lora"}:
        raise ValueError("three-dataset training supports trainable_mode full or lora")
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
        allowed_k = (4, 6, 10, 16, 21)
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
        "geometry_prefetch_workers", accumulation * microbatch * 2,
    ))
    if not 1 <= prefetch_depth <= 16:
        raise ValueError("geometry_prefetch_depth must be in [1,16]")
    if not 1 <= prefetch_workers <= 32:
        raise ValueError("geometry_prefetch_workers must be in [1,32]")


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


def prune_periodic_checkpoints(output: Path, keep_last: int) -> list[str]:
    """Bound disk use while retaining latest.pt and the newest named milestones."""
    paths = sorted(output.glob("checkpoint-*.pt"))
    remove = paths[:-max(int(keep_last), 0)] if keep_last else paths
    removed = []
    for path in remove:
        path.unlink(missing_ok=True)
        removed.append(path.name)
    return removed


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
        "--no-checkpoint", action="store_true",
        help="capacity-gate only: skip periodic/final checkpoint materialization",
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
    if rank == 0 and resume.is_file():
        status_path = Path(config.get(
            "resume_status_path", resume.parent / "train_status.json"
        ))
        if not status_path.is_file():
            raise FileNotFoundError(
                f"resume planning sidecar missing: {status_path}; refusing a second full checkpoint read"
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
    microbatch_per_gpu = int(config["microbatch_per_gpu"])
    k = int(config["targets_per_source"])
    diagnostic_every = int(config.get("diagnostic_every_steps", 20))
    checkpoint_steps = {int(value) for value in config.get("checkpoint_steps", [])}
    checkpoint_every = int(config.get("checkpoint_every_after", 5000))
    graceful_seconds = float(config.get("graceful_stop_hours", 68)) * 3600
    started = time.perf_counter()
    completed = start_step
    prefetch_depth = int(config.get("geometry_prefetch_depth", 2))
    prefetch_workers = int(config.get(
        "geometry_prefetch_workers", accumulation * microbatch_per_gpu * 2,
    ))
    pool = ThreadPoolExecutor(
        max_workers=prefetch_workers,
        thread_name_prefix="three-dataset-geometry",
    )
    optimizer.zero_grad(set_to_none=True)
    try:
        slots_per_rank = accumulation * microbatch_per_gpu

        def timed_geometry(step_dataset, index: int, permutation: np.ndarray):
            task_started = time.perf_counter()
            value = source_with_eligible_targets(step_dataset, index, permutation)
            return value, time.perf_counter() - task_started

        def plan_step(step: int):
            """Plan one update and submit its deterministic geometry futures."""
            step_name = dataset_for_step(step, seed)
            step_dataset = datasets[step_name]
            step_plans = [deterministic_sample_plan(
                step_dataset, step_name, seed, step, slot, rank, slots_per_rank
            ) for slot in range(slots_per_rank)]
            step_futures = [pool.submit(
                timed_geometry, step_dataset, index,
                np.random.default_rng(np.random.SeedSequence([seed, step, slot, rank, 771])).permutation(21)
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
                for (index, _source, rng), future in group:
                    wait_started = time.perf_counter()
                    (source, xyz_all, valid_all), task_seconds = future.result()
                    geometry_wait_seconds += time.perf_counter() - wait_started
                    geometry_task_max_seconds = max(geometry_task_max_seconds, task_seconds)
                    targets = sample_eligible_targets(valid_all, k, rng)
                    batch_values.append((index, source, targets, xyz_all[targets], valid_all[targets]))
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
                pair_count += sum(target_counts)
                for _index, source, targets, _xyz, _valid in batch_values:
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
            clips_seen[name] += world * accumulation * microbatch_per_gpu
            elapsed = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated(device) / 2**30
            diagnostic = completed == start_step + 1 or completed % diagnostic_every == 0 or completed == target_steps
            if diagnostic:
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
                    "system/geometry_wait_seconds_max_rank": float(timing_max[0]),
                    "system/geometry_task_seconds_max_rank": float(timing_max[1]),
                    "system/latent_load_seconds_max_rank": float(timing_max[2]),
                    "system/geometry_prefetch_depth": prefetch_depth,
                    "system/geometry_prefetch_workers": prefetch_workers,
                    "system/elapsed_seconds": elapsed, "train/lr_factor": lr_factor,
                    "train/gradient_norm": float(gradient_norm),
                    **{f"sampling/source_{index}": int(value) for index, value in enumerate(source_hist.tolist())},
                    **{f"sampling/target_{index}": int(value) for index, value in enumerate(target_hist.tolist())},
                    **{f"sampling/gap_{index}": int(value) for index, value in enumerate(gap_hist.tolist())},
                }
                print(json.dumps(payload), flush=True)
                if run is not None and completed > args.wandb_log_after_step:
                    run.log(payload, step=completed)
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
