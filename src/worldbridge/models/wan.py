"""Strict Wan2.1 VAE/DiT weight adapters.

The adapters fail on unexpected native shapes; no temporal interpolation or
reshape is hidden in this module.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

WAN_LATENT_SHAPE = (16, 6, 16, 16)
WAN_LATENT_SHAPE_256 = (16, 6, 32, 32)


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


WAN_TIMESTEP_SCALE = 1000.0


def _wan_stats(module: nn.Module, name: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    values = getattr(module.config, name, None)
    if values is None or len(values) != WAN_LATENT_SHAPE[0]:
        raise ValueError(f"WAN VAE config has invalid {name}: {values}")
    return torch.tensor(values, device=device, dtype=dtype).reshape(1, -1, 1, 1, 1)


def rgb_to_wan_input(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB to WAN's [B,3,T,H,W] float input without changing T."""
    if rgb.ndim != 5:
        raise ValueError(f"RGB video must be [B,T,3,H,W], got {tuple(rgb.shape)}")
    if rgb.shape[2] != 3:
        raise ValueError(f"RGB channel dimension must be 3, got {rgb.shape[2]}")
    x = rgb if rgb.is_floating_point() else rgb.float() / 255.0
    if float(x.detach().amin()) < -1e-4 or float(x.detach().amax()) > 1.0001:
        raise ValueError("RGB input must be uint8 or floating point in [0,1]")
    return (x * 2.0 - 1.0).permute(0, 2, 1, 3, 4).contiguous()


class WanVAEEncoder(nn.Module):
    """Frozen native Wan-VAE encoder returning the diffusion-normalized latent."""

    def __init__(self, checkpoint: str | Path, device: torch.device | str = "cpu",
                 dtype: torch.dtype = torch.float32, expected_shape: tuple[int, int, int, int] = WAN_LATENT_SHAPE):
        super().__init__()
        self.checkpoint = str(checkpoint)
        self.expected_shape = tuple(expected_shape)
        self.device = torch.device(device)
        self.compute_dtype = dtype
        self.vae = self._load(self.checkpoint, self.device, dtype)
        freeze_module(self.vae)

    @staticmethod
    def _load(checkpoint: str, device: torch.device, dtype: torch.dtype) -> nn.Module:
        try:
            from diffusers import AutoencoderKLWan
        except ImportError as exc:
            raise RuntimeError("WAN VAE requires diffusers>=0.36") from exc
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"WAN VAE checkpoint not found: {path}")
        # The supplied Wan2.1 directory contains the original VAE .pth, not a
        # diffusers VAE directory. from_single_file performs the official key
        # conversion and avoids a guessed model or a silent resize.
        config = {
            "base_dim": 96, "decoder_base_dim": 96, "z_dim": 16,
            "dim_mult": [1, 2, 4, 4], "num_res_blocks": 2,
            "attn_scales": [], "temperal_downsample": [False, True, True],
            "latents_mean": [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                             0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921],
            "latents_std": [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160],
            "scale_factor_temporal": 4, "scale_factor_spatial": 8,
        }
        from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers, load_single_file_checkpoint
        checkpoint_state = load_single_file_checkpoint(str(path))
        vae = AutoencoderKLWan(**config)
        converted = convert_wan_vae_to_diffusers(checkpoint_state)
        missing, unexpected = vae.load_state_dict(converted, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"WAN VAE checkpoint conversion mismatch; missing={missing[:8]}, unexpected={unexpected[:8]}")
        vae.to(device=device, dtype=dtype)
        return vae

    @property
    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        x = rgb_to_wan_input(rgb).to(device=self.device, dtype=self.compute_dtype)
        posterior = self.vae.encode(x, return_dict=True).latent_dist
        # Mean is deterministic; sampling would make cache endpoints differ
        # between runs. These are WAN's own channel statistics, not a project
        # normalization and not a temporal operation.
        raw = posterior.mean
        mean = _wan_stats(self.vae, "latents_mean", raw.device, raw.dtype)
        std = _wan_stats(self.vae, "latents_std", raw.device, raw.dtype)
        latent = (raw - mean) / std
        actual = tuple(latent.shape[1:])
        if actual != self.expected_shape:
            raise RuntimeError(
                "WAN VAE latent shape mismatch: actual "
                f"[B,{','.join(map(str, actual))}] != expected [B,{','.join(map(str, self.expected_shape))}]. "
                "For T=21, the native causal WAN temporal factor 4 yields 6; "
                "the adapter will not reshape, pool, crop, or interpolate it."
            )
        return latent


class WanDiTMapping(nn.Module):
    """Native Wan DiT used as F_theta(Y, tau)."""

    def __init__(self, checkpoint: str | Path,
                 device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32,
                 timestep_scale: float = WAN_TIMESTEP_SCALE,
                 expected_latent_shape: tuple[int, int, int, int] = WAN_LATENT_SHAPE,
                 load_pretrained_weights: bool = True, native_512: bool = False):
        super().__init__()
        self.checkpoint = str(checkpoint)
        self.timestep_scale = float(timestep_scale)
        self.expected_latent_shape = tuple(int(value) for value in expected_latent_shape)
        self.native_512 = bool(native_512)
        if self.native_512 and self.expected_latent_shape != (16, 6, 32, 32):
            raise ValueError('native512 mapping retains the original256 base contract')
        if len(self.expected_latent_shape) != 4 or self.expected_latent_shape[:2] != (16, 6):
            raise ValueError(f"unsupported Wan latent contract: {self.expected_latent_shape}")
        self.dit = self._load(
            self.checkpoint, torch.device(device), dtype,
            load_pretrained_weights=bool(load_pretrained_weights),
        )

    @staticmethod
    def _architecture(path: Path) -> dict[str, Any]:
        config_path = path.parent / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"WAN architecture config not found: {config_path}")
        native = json.loads(config_path.read_text())
        hidden_dim = int(native.get("dim", 0))
        num_heads = int(native.get("num_heads", 0))
        num_layers = int(native.get("num_layers", 0))
        if hidden_dim <= 0 or num_heads <= 0 or hidden_dim % num_heads:
            raise ValueError(f"invalid native WAN architecture in {config_path}: {native}")
        return {
            "patch_size": (1, 2, 2),
            "num_attention_heads": num_heads,
            "attention_head_dim": hidden_dim // num_heads,
            "in_channels": int(native.get("in_dim", 16)),
            "out_channels": int(native.get("out_dim", 16)),
            "text_dim": int(native.get("text_dim", 4096)),
            "freq_dim": int(native.get("freq_dim", 256)),
            "ffn_dim": int(native.get("ffn_dim", 0)),
            "num_layers": num_layers,
            "cross_attn_norm": True,
            "qk_norm": "rms_norm_across_heads",
            "eps": float(native.get("eps", 1e-6)),
            "rope_max_seq_len": 1024,
        }

    @staticmethod
    def _convert_native_key(key: str) -> str:
        if "model.diffusion_model." in key:
            key = key.replace("model.diffusion_model.", "")
        replacements = (
            ("time_embedding.0", "condition_embedder.time_embedder.linear_1"),
            ("time_embedding.2", "condition_embedder.time_embedder.linear_2"),
            ("text_embedding.0", "condition_embedder.text_embedder.linear_1"),
            ("text_embedding.2", "condition_embedder.text_embedder.linear_2"),
            ("time_projection.1", "condition_embedder.time_proj"),
            ("cross_attn", "attn2"), ("self_attn", "attn1"),
            (".o.", ".to_out.0."), (".q.", ".to_q."),
            (".k.", ".to_k."), (".v.", ".to_v."),
            ("head.modulation", "scale_shift_table"),
            ("head.head", "proj_out"), ("modulation", "scale_shift_table"),
            ("ffn.0", "ffn.net.0.proj"), ("ffn.2", "ffn.net.2"),
            ("norm2", "norm__placeholder"), ("norm3", "norm2"),
            ("norm__placeholder", "norm3"),
        )
        for old, new in replacements:
            key = key.replace(old, new)
        return key

    @classmethod
    def _load_sharded(cls, model: nn.Module, index_path: Path) -> None:
        """Load only tensors retained by the instantiated model."""
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError("sharded WAN loading requires safetensors") from exc
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"invalid WAN shard index: {index_path}")
        expected = set(model.state_dict())
        by_file: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for native_key, filename in weight_map.items():
            converted_key = cls._convert_native_key(native_key)
            if converted_key in expected:
                by_file[str(filename)].append((native_key, converted_key))
        loaded: set[str] = set()
        for filename in sorted(by_file):
            shard_path = index_path.parent / filename
            if not shard_path.exists():
                raise FileNotFoundError(f"WAN checkpoint shard not found: {shard_path}")
            with safe_open(shard_path, framework="pt", device="cpu") as shard:
                converted = {
                    converted_key: shard.get_tensor(native_key)
                    for native_key, converted_key in by_file[filename]
                }
            _, unexpected = model.load_state_dict(converted, strict=False)
            if unexpected:
                raise RuntimeError(
                    f"unexpected converted WAN keys in {filename}: {unexpected[:8]}"
                )
            loaded.update(converted)
            del converted
        missing = sorted(expected - loaded)
        if missing:
            raise RuntimeError(
                f"sharded WAN checkpoint is missing retained keys: {missing[:8]}"
            )

    @staticmethod
    def _new_model(config: dict[str, Any]) -> nn.Module:
        try:
            from diffusers import WanTransformer3DModel
        except ImportError as exc:
            raise RuntimeError("WAN DiT requires diffusers>=0.36") from exc
        return WanTransformer3DModel(**config)

    @classmethod
    def _load(cls, checkpoint: str, device: torch.device, dtype: torch.dtype,
              load_pretrained_weights: bool = True) -> nn.Module:
        path = Path(checkpoint)
        if load_pretrained_weights and not path.exists():
            raise FileNotFoundError(f"WAN DiT checkpoint not found: {path}")
        config = cls._architecture(path)
        model = cls._new_model(config)
        if load_pretrained_weights:
            if path.name.endswith(".safetensors.index.json"):
                cls._load_sharded(model, path)
            else:
                from diffusers.loaders.single_file_utils import (
                    convert_wan_transformer_to_diffusers, load_single_file_checkpoint,
                )
                checkpoint_state = load_single_file_checkpoint(str(path))
                converted = convert_wan_transformer_to_diffusers(checkpoint_state)
                missing, unexpected = model.load_state_dict(converted, strict=False)
                if missing or unexpected:
                    raise RuntimeError(
                        "WAN DiT checkpoint conversion mismatch; "
                        f"missing={missing[:8]}, unexpected={unexpected[:8]}"
                    )
        model.to(device=device, dtype=dtype)
        return model

    def _condition(self, batch: int, device: torch.device, dtype: torch.dtype,
                   encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        condition = torch.as_tensor(encoder_hidden_states)
        if condition.numel() == 0:
            raise ValueError('WAN condition must be a native UMT5 tensor, not an empty placeholder')
        if condition.ndim == 2:
            condition = condition[None]
        expected_text_dim = int(getattr(self.dit.config, "text_dim", 4096))
        if condition.ndim != 3 or condition.shape[1:] != (512, expected_text_dim):
            raise ValueError(
                f"WAN condition must be [1 or B,512,{expected_text_dim}], got {tuple(condition.shape)}"
            )
        if condition.shape[0] not in (1, batch):
            raise ValueError("WAN condition batch dimension does not match latent batch")
        return condition.expand(batch, -1, -1).to(device=device, dtype=dtype)

    def _inputs(self, latent: torch.Tensor, tau: torch.Tensor,
                encoder_hidden_states: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expected_shape = getattr(self, "expected_latent_shape", WAN_LATENT_SHAPE)
        allowed_shapes = {expected_shape}
        if getattr(self, 'native_512', False):
            allowed_shapes.add((16, 6, 64, 64))
        if latent.ndim != 5 or tuple(latent.shape[1:]) not in allowed_shapes:
            expected = ','.join(map(str, expected_shape))
            raise ValueError(f"WAN mapper input must be [B,{expected}], got {tuple(latent.shape)}")
        tau = torch.as_tensor(tau, device=latent.device, dtype=latent.dtype).flatten()
        if tau.shape != (latent.shape[0],):
            raise ValueError(f"tau must be [B]={latent.shape[0]}, got {tuple(tau.shape)}")
        if not torch.isfinite(tau).all() or (tau < 0).any() or (tau > 1).any():
            raise ValueError("external flow tau must be in [0,1]")
        dit_dtype = next(self.dit.parameters()).dtype
        hidden_states = latent.to(dtype=dit_dtype)
        timestep = (tau * self.timestep_scale).to(dtype=dit_dtype)
        condition = self._condition(
            latent.shape[0], latent.device, dit_dtype, encoder_hidden_states
        )
        return hidden_states, timestep, condition

    def forward(self, latent: torch.Tensor, tau: torch.Tensor,
                encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, timestep, condition = self._inputs(latent, tau, encoder_hidden_states)
        output = self.dit(hidden_states, timestep=timestep,
                          encoder_hidden_states=condition, return_dict=True).sample
        output = output.to(dtype=latent.dtype)
        if output.shape != latent.shape:
            raise RuntimeError(f"WAN DiT output shape {tuple(output.shape)} != input {tuple(latent.shape)}")
        return output

    def forward_hidden_layers(
        self,
        latent: torch.Tensor,
        tau: torch.Tensor,
        layers: tuple[int, ...],
        encoder_hidden_states: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[int, int, int]]:
        """Run Wan through selected transformer blocks without its RF output head.

        ``layers`` contains zero-based block indices. Returned tensors preserve
        Wan's patch-token order ``(latent_time, row, column)`` and never pass
        through ``norm_out`` or the 16-channel ``proj_out``.
        """
        layers = tuple(int(index) for index in layers)
        if not layers or tuple(sorted(set(layers))) != layers:
            raise ValueError("hidden layers must be a non-empty, sorted, unique tuple")
        if layers[0] < 0 or layers[-1] >= len(self.dit.blocks):
            raise ValueError(f"hidden layer outside [0,{len(self.dit.blocks) - 1}]: {layers}")

        hidden_states, timestep, condition = self._inputs(latent, tau, encoder_hidden_states)
        batch, _, frames, height, width = hidden_states.shape
        patch_t, patch_h, patch_w = map(int, self.dit.config.patch_size)
        grid_shape = (frames // patch_t, height // patch_h, width // patch_w)
        rotary_emb = self.dit.rope(hidden_states)
        hidden_states = self.dit.patch_embedding(hidden_states).flatten(2).transpose(1, 2)
        _, timestep_proj, condition, _ = self.dit.condition_embedder(timestep, condition)
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        selected: list[torch.Tensor] = []
        wanted = set(layers)
        for index, block in enumerate(self.dit.blocks):
            if torch.is_grad_enabled() and self.dit.gradient_checkpointing:
                hidden_states = self.dit._gradient_checkpointing_func(
                    block, hidden_states, condition, timestep_proj, rotary_emb
                )
            else:
                hidden_states = block(hidden_states, condition, timestep_proj, rotary_emb)
            if index in wanted:
                selected.append(hidden_states)
            if index == layers[-1]:
                break

        expected_tokens = grid_shape[0] * grid_shape[1] * grid_shape[2]
        if len(selected) != len(layers) or any(value.shape != (batch, expected_tokens, hidden_states.shape[-1])
                                                  for value in selected):
            raise RuntimeError("Wan hidden-state extraction produced an unexpected shape")
        return tuple(selected), grid_shape

    @property
    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]
