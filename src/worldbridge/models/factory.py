"""Reproducible construction of real Wan-backed models."""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Sequence

import torch

from ..data.movif import MOViSample
from .backbones import CleanLatentBackbone, FeedForwardWanBackbone, WanHiddenGeometryBackbone
from .decoder import DenseQueryDecoder
from .wan import WAN_LATENT_SHAPE, WanDiTMapping, WanVAEEncoder, inject_wan_lora
from .worldbridge import DenseQueryWanModel

def precision_dtype(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported precision={name!r}")

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
