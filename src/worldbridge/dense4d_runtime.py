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
from .wan import WAN_LATENT_SHAPE, WanDiTMapping, WanVAEEncoder


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


def load_empty_condition(path: str | Path) -> torch.Tensor:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"native Wan empty condition not found: {path}; create it with scripts/create_wan_empty_text_condition.py"
        )
    value = torch.load(path, map_location="cpu", weights_only=True)
    value = value.get("encoder_hidden_states", value) if isinstance(value, dict) else value
    value = torch.as_tensor(value)
    if value.shape != (1, 512, 4096):
        raise ValueError(f"empty Wan condition must be [1,512,4096], got {tuple(value.shape)}")
    return value


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


def build_real_model(config: dict[str, Any], device: torch.device | str) -> DenseQueryWanModel:
    device = torch.device(device)
    dtype = precision_dtype(config["precision"])
    readout = str(config.get("backbone_readout", "wan_velocity"))
    structured = readout == "wan_hidden_structured"
    if readout in {"wan_velocity", "wan_hidden_structured"}:
        condition = load_empty_condition(config["empty_text_condition"])
        mapping = WanDiTMapping(
            Path(config["wan_root"]) / "diffusion_pytorch_model.safetensors",
            condition=condition, device=device, dtype=dtype,
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
    ) if structured else WAN_LATENT_SHAPE
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
    ).to(device=device, dtype=dtype)
    model = DenseQueryWanModel(backbone, decoder)
    model.configure_trainable(str(config.get("trainable_mode", "full")), int(config.get("trainable_blocks", 2)))
    return model


def parameter_groups(model: DenseQueryWanModel, config: dict[str, Any]) -> list[dict[str, Any]]:
    backbone = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    adapter = [parameter for parameter in getattr(model.backbone, "adapter_parameters", [])
               if parameter.requires_grad]
    adapter_ids = {id(parameter) for parameter in adapter}
    wan = [parameter for parameter in backbone if id(parameter) not in adapter_ids]
    decoder = [parameter for parameter in model.decoder.parameters() if parameter.requires_grad]
    groups = []
    if wan:
        groups.append({"params": wan, "lr": float(config.get("backbone_learning_rate", config["learning_rate"])),
                       "name": "wan_backbone"})
    if adapter:
        groups.append({"params": adapter, "lr": float(config.get("geometry_learning_rate", config["learning_rate"])),
                       "name": "geometry_adapter"})
    if decoder:
        groups.append({"params": decoder, "lr": float(config["learning_rate"]), "name": "dense_decoder"})
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
