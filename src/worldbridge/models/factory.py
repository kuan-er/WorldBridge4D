"""Reproducible construction of real Wan-backed models."""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Sequence

import torch

from ..data.movif import MOViSample
from .backbones import CleanLatentBackbone, FeedForwardWanBackbone, WanHiddenGeometryBackbone
from .decoder import DenseQueryDecoder
from .wan import WAN_LATENT_SHAPE, WanDiTMapping, WanVAEEncoder
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
            f"native Wan text condition not found: {path}; create it with scripts/prepare_data.py text-conditions"
        )
    value = torch.load(path, map_location="cpu", weights_only=True)
    value = value.get("encoder_hidden_states", value) if isinstance(value, dict) else value
    value = torch.as_tensor(value)
    if value.shape != (1, 512, 4096):
        raise ValueError(f"Wan condition must be [1,512,4096], got {tuple(value.shape)}")
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
            native_512=bool(config.get('native_kubric512_b1_a4_k15', False) or config.get('native_kubric512_b1_a4_k9', False) or config.get('native_kubric512_b1_a4_k5', False) or config.get('native_kubric512_full', False)),
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
                native_512=bool(config.get('native_kubric512_b1_a4_k15', False) or config.get('native_kubric512_b1_a4_k9', False) or config.get('native_kubric512_b1_a4_k5', False) or config.get('native_kubric512_full', False)),
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
        pre_attention_rgb_query=bool(config.get("pre_attention_rgb_query", False)),
        native_512=bool(config.get('native_kubric512_b1_a4_k15', False) or config.get('native_kubric512_b1_a4_k9', False) or config.get('native_kubric512_b1_a4_k5', False) or config.get('native_kubric512_full', False)),
    ).to(device=device, dtype=dtype)
    camera_head = None
    camera_cfg = config.get('camera_supervision')
    if camera_cfg is not None:
        if not structured:
            raise ValueError('camera queries require structured physical-frame features')
        from .camera import SourceConditionedCameraHead
        torch.manual_seed(int(camera_cfg['seed']))
        camera_head = SourceConditionedCameraHead(
            input_dim=int(config['geometry_dim']), dim=int(camera_cfg['dim']),
            num_heads=int(camera_cfg['num_heads']), num_frames=int(config['clip_length']),
            memory_grid=int(camera_cfg['memory_grid']),
        ).to(device=device, dtype=dtype)
    model = DenseQueryWanModel(backbone, decoder, camera_head)
    model.configure_trainable(str(config.get("trainable_mode", "full")))
    return model
