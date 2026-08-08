"""Reproducible construction helpers for real Wan H004 runs."""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .data import MOViSample
from .dense4d import CleanLatentBackbone, DenseQueryDecoder, DenseQueryWanModel, FeedForwardWanBackbone
from .wan import WAN_LATENT_SHAPE, WanDiTMapping, WanVAEEncoder


def precision_dtype(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported precision={name!r}")


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
    if readout == "wan_velocity":
        condition = load_empty_condition(config["empty_text_condition"])
        mapping = WanDiTMapping(
            Path(config["wan_root"]) / "diffusion_pytorch_model.safetensors",
            condition=condition, device=device, dtype=dtype,
        )
        if bool(config.get("gradient_checkpointing", True)):
            mapping.dit.enable_gradient_checkpointing()
        backbone = FeedForwardWanBackbone(mapping)
    elif readout == "clean_latent":
        backbone = CleanLatentBackbone()
    else:
        raise ValueError(f"unknown backbone_readout={readout!r}")
    # Construct every decoder ablation from identical weights even when the
    # backbone path consumes a different amount of RNG during loading.
    torch.manual_seed(int(config.get("decoder_seed", config.get("seed", 0))))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(config.get("decoder_seed", config.get("seed", 0))))
    decoder = DenseQueryDecoder(
        num_frames=int(config["clip_length"]), latent_shape=WAN_LATENT_SHAPE,
        query_dim=int(config["query_dim"]), embedding_dim=int(config.get("embedding_dim", 128)),
        num_layers=int(config["num_cross_attn_layers"]), num_heads=int(config["num_heads"]),
        upsample_channels=tuple(int(x) for x in config["upsample_channels"]),
        output_size=(int(config["image_size"]), int(config["image_size"])),
        coarse_diagnostic=bool(config.get("coarse_diagnostic", False)),
        fullres_coordinates=bool(config.get("fullres_coordinates", False)),
        query_grid_size=int(config.get("query_grid_size", WAN_LATENT_SHAPE[-1])),
        rope_mode=str(config.get("rope_mode", "2d")),
        visibility_head=bool(config.get("visibility_head", False)),
    ).to(device=device, dtype=dtype)
    model = DenseQueryWanModel(backbone, decoder)
    model.configure_trainable(str(config.get("trainable_mode", "full")), int(config.get("trainable_blocks", 2)))
    return model


def parameter_groups(model: DenseQueryWanModel, config: dict[str, Any]) -> list[dict[str, Any]]:
    backbone = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    decoder = [parameter for parameter in model.decoder.parameters() if parameter.requires_grad]
    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": float(config.get("backbone_learning_rate", config["learning_rate"])),
                       "name": "wan_backbone"})
    if decoder:
        groups.append({"params": decoder, "lr": float(config["learning_rate"]), "name": "dense_decoder"})
    if not groups:
        raise ValueError("model has no trainable parameters")
    return groups


def optimizer_trainable_count(groups: list[dict[str, Any]]) -> int:
    return sum(parameter.numel() for group in groups for parameter in group["params"])


def save_checkpoint(path: str | Path, model: DenseQueryWanModel, config: dict[str, Any],
                    coordinate_mean: np.ndarray, coordinate_scale: np.ndarray,
                    extra: dict[str, Any] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(), "config": config,
        # Plain lists keep weights_only=True checkpoint validation safe on
        # PyTorch 2.6+; no NumPy reconstruction globals are needed.
        "coordinate_mean": np.asarray(coordinate_mean, np.float32).tolist(),
        "coordinate_scale": np.asarray(coordinate_scale, np.float32).tolist(),
        "extra": extra or {},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path
