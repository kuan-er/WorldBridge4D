"""Reproducible construction helpers for real Wan H004 runs."""
from __future__ import annotations

import copy
import gc
from pathlib import Path
import random
from typing import Any, Sequence

import numpy as np
import torch

from .data import MOViSample
from .dense4d import (
    CleanLatentBackbone, DenseQueryDecoder, DenseQueryWanModel,
    FeedForwardWanBackbone, WanHiddenGeometryBackbone,
)
from .wan import WAN_LATENT_SHAPE, WanDiTMapping, WanVAEEncoder, inject_wan_lora


def precision_dtype(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported precision={name!r}")


def apply_linear_warmup(optimizer: torch.optim.Optimizer, update_number: int,
                        warmup_steps: int) -> float:
    """Scale every optimizer group linearly up to its configured base LR.

    ``update_number`` is one-based: the first optimizer update uses
    ``1 / warmup_steps`` of each group's base LR.  The base LR is stored on the
    optimizer group so the Wan and decoder groups retain their independent
    configured rates.
    """
    update_number, warmup_steps = int(update_number), int(warmup_steps)
    if update_number < 1:
        raise ValueError("update_number must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    factor = 1.0 if warmup_steps == 0 else min(1.0, update_number / warmup_steps)
    for group in optimizer.param_groups:
        base_lr = float(group.setdefault("_base_lr", group["lr"]))
        group["lr"] = base_lr * factor
    return factor


def load_text_condition(path: str | Path) -> torch.Tensor:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"native Wan text condition not found: {path}; create it with scripts/create_wan_text_conditions.py"
        )
    value = torch.load(path, map_location="cpu", weights_only=True)
    value = value.get("encoder_hidden_states", value) if isinstance(value, dict) else value
    value = torch.as_tensor(value)
    if value.shape != (1, 512, 4096):
        raise ValueError(f"Wan condition must be [1,512,4096], got {tuple(value.shape)}")
    return value


# Backwards-compatible alias for existing 128/empty-condition configurations.
load_empty_condition = load_text_condition


def encode_clean_video_latents(samples: Sequence[MOViSample], wan_root: str | Path,
                               device: torch.device | str) -> list[torch.Tensor]:
    """Frozen deterministic VAE means cached on CPU; no diffusion noise is added."""
    device = torch.device(device)
    encoder = WanVAEEncoder(Path(wan_root) / "Wan2.1_VAE.pth", device=device, dtype=torch.float32)
    latents = []
    with torch.inference_mode():
        for sample in samples:
            rgb = torch.from_numpy(sample.rgb).permute(0, 3, 1, 2)[None].to(device)
            latent = encoder(rgb)
            if tuple(latent.shape[1:]) != WAN_LATENT_SHAPE:
                raise RuntimeError(f"unexpected clean latent {tuple(latent.shape)}")
            latents.append(latent.float().cpu())
    del encoder
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return latents


def build_real_model(
    config: dict[str, Any], device: torch.device | str,
    *, load_wan_pretrained: bool = True,
) -> DenseQueryWanModel:
    device = torch.device(device)
    dtype = precision_dtype(config["precision"])
    readout = str(config.get("backbone_readout", "wan_velocity"))
    structured = readout == "wan_hidden_structured"
    latent_spatial_size = int(config.get("latent_spatial_size", int(config["image_size"]) // 8))
    wan_latent_shape = (16, 6, latent_spatial_size, latent_spatial_size)
    if readout in {"wan_velocity", "wan_hidden_structured"}:
        condition_path = config.get("empty_text_condition")
        condition = load_text_condition(condition_path) if condition_path else None
        wan_root = Path(config["wan_root"])
        wan_dit_root = Path(config.get("wan_dit_root", wan_root))
        checkpoint = Path(config.get(
            "wan_checkpoint", wan_dit_root / "diffusion_pytorch_model.safetensors",
        ))
        if not checkpoint.exists():
            sharded_index = wan_dit_root / "diffusion_pytorch_model.safetensors.index.json"
            if sharded_index.exists():
                checkpoint = sharded_index
        mapping = WanDiTMapping(
            checkpoint,
            condition=condition, device=device, dtype=dtype,
            expected_latent_shape=wan_latent_shape,
            truncate_after_block=config.get("wan_truncate_after_block"),
            load_pretrained_weights=load_wan_pretrained,
        )
        mode = str(config.get("trainable_mode", "full"))
        if mode == "lora":
            for parameter in mapping.dit.parameters():
                parameter.requires_grad_(False)
            mapping.lora_modules = inject_wan_lora(
                mapping.dit,
                rank=int(config.get("lora_rank", 16)),
                alpha=float(config.get("lora_alpha", config.get("lora_rank", 16))),
                dropout=float(config.get("lora_dropout", 0.0)),
                targets=tuple(config.get("lora_targets", (
                    "to_q", "to_k", "to_v", "to_out.0",
                    "ffn.net.0.proj", "ffn.net.2",
                ))),
            )
        if bool(config.get("gradient_checkpointing", True)):
            mapping.dit.enable_gradient_checkpointing()
        if readout == "wan_velocity":
            backbone = FeedForwardWanBackbone(mapping)
        else:
            geometry_seed = int(config.get("geometry_seed", config.get("seed", 0)))
            torch.manual_seed(geometry_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(geometry_seed)
            backbone = WanHiddenGeometryBackbone(
                mapping,
                hidden_layers=tuple(int(value) for value in config.get(
                    "wan_hidden_layers", (5, 11, 17, 23, 29)
                )),
                geometry_dim=int(config.get("geometry_dim", 128)),
                num_frames=int(config["clip_length"]),
                spatial_size=int(config.get("geometry_spatial_size", WAN_LATENT_SHAPE[-1])),
                motion_slots=int(config.get("motion_slots", 16)),
                num_heads=int(config.get("geometry_num_heads", 8)),
                use_clean_skip=bool(config.get("geometry_clean_skip", True)),
                layer_gate_temperature=float(config.get("layer_gate_temperature", 1.0)),
                layer_gate_top_k=config.get("layer_gate_top_k"),
                layer_gate_init_std=float(config.get("layer_gate_init_std", 0.0)),
                layer_gate_seed=int(config.get("layer_gate_seed", geometry_seed)),
                layer_gate_initial_logits=config.get("layer_gate_initial_logits"),
            )
    elif readout == "clean_latent":
        backbone = CleanLatentBackbone()
    else:
        raise ValueError(f"unknown backbone_readout={readout!r}")
    backbone = backbone.to(device=device, dtype=dtype)
    # Construct every decoder ablation from identical weights even when the
    # backbone path consumes a different amount of RNG during loading.
    torch.manual_seed(int(config.get("decoder_seed", config.get("seed", 0))))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(config.get("decoder_seed", config.get("seed", 0))))
    latent_shape = (
        int(config.get("geometry_dim", 128)), int(config["clip_length"]),
        int(config.get("geometry_spatial_size", WAN_LATENT_SHAPE[-1])),
        int(config.get("geometry_spatial_size", WAN_LATENT_SHAPE[-1])),
    ) if structured else wan_latent_shape
    decoder = DenseQueryDecoder(
        num_frames=int(config["clip_length"]), latent_shape=latent_shape,
        query_dim=int(config["query_dim"]), embedding_dim=int(config.get("embedding_dim", 128)),
        num_layers=int(config["num_cross_attn_layers"]), num_heads=int(config["num_heads"]),
        upsample_channels=tuple(int(x) for x in config["upsample_channels"]),
        output_size=(int(config["image_size"]), int(config["image_size"])),
        coarse_diagnostic=bool(config.get("coarse_diagnostic", False)),
        fullres_coordinates=bool(config.get("fullres_coordinates", False)),
        query_grid_size=int(config.get("query_grid_size", latent_shape[-1])),
        structured_motion_slots=int(config.get("motion_slots", 16)) if structured else 0,
        structured_local_queries=bool(config.get("structured_local_queries", True)) if structured else False,
        structured_pair_motion_queries=bool(config.get("structured_pair_motion_queries", False)) if structured else False,
        structured_pair_motion_zero_init=bool(config.get("structured_pair_motion_zero_init", False)) if structured else False,
        source_rgb_pyramid=bool(config.get("source_rgb_pyramid", False)),
        source_rgb_channels=tuple(int(value) for value in config.get(
            "source_rgb_channels", (32, 64, 128)
        )),
        source_rgb_fusion_32=bool(config.get("source_rgb_fusion_32", False)),
    ).to(device=device, dtype=dtype)
    model = DenseQueryWanModel(backbone, decoder)
    mode = str(config.get("trainable_mode", "full"))
    model.configure_trainable(
        "full" if mode == "lora" else mode,
        int(config.get("trainable_blocks", 2)),
    )
    if mode == "lora":
        for parameter in model.backbone.mapping.parameters():
            parameter.requires_grad_(False)
        for module in model.backbone.mapping.dit.modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                for parameter in module.lora_A.parameters():
                    parameter.requires_grad_(True)
                for parameter in module.lora_B.parameters():
                    parameter.requires_grad_(True)
        for parameter in model.backbone.adapter_parameters:
            parameter.requires_grad_(True)
        for parameter in model.decoder.parameters():
            parameter.requires_grad_(True)
    return model


def parameter_groups(model: DenseQueryWanModel, config: dict[str, Any]) -> list[dict[str, Any]]:
    named = list(model.named_parameters())
    names_by_id = {id(parameter): name for name, parameter in named}
    rgb_prefixes = (
        "decoder.upsampler.source_rgb_encoder.",
        "decoder.upsampler.source_fusions.",
    )
    separate_rgb = bool(config.get("source_rgb_separate_optimizer_group", False))
    rgb = [
        parameter for name, parameter in named
        if separate_rgb and parameter.requires_grad and name.startswith(rgb_prefixes)
    ]
    rgb_ids = {id(parameter) for parameter in rgb}
    backbone = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    adapter = [parameter for parameter in getattr(model.backbone, "adapter_parameters", [])
               if parameter.requires_grad]
    adapter_ids = {id(parameter) for parameter in adapter}
    wan = [parameter for parameter in backbone if id(parameter) not in adapter_ids]
    decoder = [
        parameter for parameter in model.decoder.parameters()
        if parameter.requires_grad and id(parameter) not in rgb_ids
    ]
    groups = []
    if wan:
        groups.append({"params": wan, "lr": float(config.get("backbone_learning_rate", config["learning_rate"])),
                       "name": "wan_backbone"})
    if adapter:
        groups.append({"params": adapter, "lr": float(config.get("geometry_learning_rate", config["learning_rate"])),
                       "name": "geometry_adapter"})
    if decoder:
        groups.append({"params": decoder, "lr": float(config["learning_rate"]), "name": "dense_decoder"})
    if rgb:
        multiplier = float(config.get("source_rgb_learning_rate_multiplier", 1.0))
        if not np.isfinite(multiplier) or multiplier <= 0:
            raise ValueError("source RGB learning-rate multiplier must be finite and positive")
        rgb_lr = float(config["learning_rate"]) * multiplier
        decay = [parameter for parameter in rgb if parameter.ndim > 1]
        no_decay = [parameter for parameter in rgb if parameter.ndim <= 1]
        if decay:
            groups.append({
                "params": decay, "lr": rgb_lr,
                "weight_decay": float(config["weight_decay"]),
                "name": "source_rgb_decay",
            })
        if no_decay:
            groups.append({
                "params": no_decay, "lr": rgb_lr, "weight_decay": 0.0,
                "name": "source_rgb_no_decay",
            })
        grouped_rgb = {id(parameter) for group in groups[-2:] for parameter in group["params"]} \
            if decay and no_decay else {id(parameter) for parameter in (decay + no_decay)}
        if grouped_rgb != rgb_ids:
            missing = [names_by_id[value] for value in rgb_ids - grouped_rgb]
            raise RuntimeError(f"source RGB optimizer grouping lost parameters: {missing[:8]}")
    if not groups:
        raise ValueError("model has no trainable parameters")
    return groups


def optimizer_trainable_count(groups: list[dict[str, Any]]) -> int:
    return sum(parameter.numel() for group in groups for parameter in group["params"])


def capture_rng_state(numpy_generator: np.random.Generator | None = None,
                      include_cuda: bool = True) -> dict[str, Any]:
    """Capture restart-safe Python, NumPy, and Torch RNG state."""
    numpy_global = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy_global": {
            "bit_generator": numpy_global[0],
            # PyTorch 2.4/2.5 cannot serialize TypedStorage(torch.uint32).
            # int64 is weights_only-safe and restore_rng_state casts back to
            # NumPy's required uint32 representation exactly.
            "state": torch.from_numpy(numpy_global[1].astype(np.int64, copy=True)),
            "position": int(numpy_global[2]),
            "has_gauss": int(numpy_global[3]),
            "cached_gaussian": float(numpy_global[4]),
        },
        "torch_cpu": torch.get_rng_state(),
    }
    if numpy_generator is not None:
        state["numpy_generator"] = copy.deepcopy(numpy_generator.bit_generator.state)
    if include_cuda and torch.cuda.is_available():
        # Dense4D is single-device; storing only the active device keeps exact
        # resume portable across different CUDA_VISIBLE_DEVICES layouts.
        state["torch_cuda"] = torch.cuda.get_rng_state()
    return state


def restore_rng_state(state: dict[str, Any], numpy_generator: np.random.Generator | None = None) -> None:
    """Restore state captured by :func:`capture_rng_state`."""
    required = {"python", "numpy_global", "torch_cpu"}
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"checkpoint RNG state is missing {missing}")
    random.setstate(state["python"])
    numpy_global = state["numpy_global"]
    np.random.set_state((
        str(numpy_global["bit_generator"]),
        torch.as_tensor(numpy_global["state"]).cpu().numpy().astype(np.uint32, copy=False),
        int(numpy_global["position"]), int(numpy_global["has_gauss"]),
        float(numpy_global["cached_gaussian"]),
    ))
    if numpy_generator is not None:
        if "numpy_generator" not in state:
            raise ValueError("checkpoint RNG state has no local NumPy generator")
        numpy_generator.bit_generator.state = copy.deepcopy(state["numpy_generator"])
    torch.set_rng_state(torch.as_tensor(state["torch_cpu"], dtype=torch.uint8).cpu())
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state(torch.as_tensor(cuda_state, dtype=torch.uint8).cpu())


def save_checkpoint(path: str | Path, model: DenseQueryWanModel, config: dict[str, Any],
                    coordinate_mean: np.ndarray, coordinate_scale: np.ndarray,
                    extra: dict[str, Any] | None = None,
                    optimizer: torch.optim.Optimizer | None = None,
                    training_state: dict[str, Any] | None = None) -> Path:
    """Atomically save model state and optional exact-resume state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": 2,
        "model": model.state_dict(), "config": config,
        # Plain lists keep weights_only=True checkpoint validation safe on
        # PyTorch 2.6+; no NumPy reconstruction globals are needed.
        "coordinate_mean": np.asarray(coordinate_mean, np.float32).tolist(),
        "coordinate_scale": np.asarray(coordinate_scale, np.float32).tolist(),
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if training_state is not None:
        payload["training_state"] = training_state
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path
