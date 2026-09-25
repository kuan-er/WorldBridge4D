"""Wan-backed feed-forward geometry representations."""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from ..decoder.blocks import _group_count
from ..outputs import StructuredZ4D
from ..wan import WAN_LATENT_SHAPE, WanDiTMapping

class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(channels), channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(channels), channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.conv1(F.silu(self.norm1(value)))
        value = self.conv2(F.silu(self.norm2(value)))
        return value + residual


def _temporal_interpolation_logits(output_frames: int, native_frames: int) -> torch.Tensor:
    """Logits initialized to linear interpolation over Wan's six causal times."""
    native_positions = torch.linspace(0, output_frames - 1, native_frames)
    weights = torch.zeros(output_frames, native_frames)
    for frame in range(output_frames):
        right = int(torch.searchsorted(native_positions, torch.tensor(float(frame))).clamp(max=native_frames - 1))
        left = max(0, right - 1)
        if left == right:
            weights[frame, left] = 1.0
        else:
            alpha = (frame - float(native_positions[left])) / float(native_positions[right] - native_positions[left])
            weights[frame, left] = 1.0 - alpha
            weights[frame, right] = alpha
    return weights.clamp_min(1e-4).log()


class WanHiddenGeometryBackbone(nn.Module):
    """Selected pre-output Wan states -> frame-aligned dense planes and motion slots."""

    raw_velocity_convention = "not_used"
    z4d_transform = "selected_hidden_states_to_structured_geometry"

    def __init__(self, mapping: WanDiTMapping, hidden_layers: Sequence[int] = (5, 11, 17, 23, 29),
                 geometry_dim: int = 128, num_frames: int = 21, spatial_size: int = 16,
                 motion_slots: int = 16, num_heads: int = 8, use_clean_skip: bool = True,
                 layer_gate_temperature: float = 1.0, layer_gate_top_k: int | None = None,
                 layer_gate_initial_logits: Sequence[float] | None = None,
                 native_512: bool = False):
        super().__init__()
        self.mapping = mapping
        self.hidden_layers = tuple(int(index) for index in hidden_layers)
        self.geometry_dim = int(geometry_dim)
        self.num_frames = int(num_frames)
        self.spatial_size = int(spatial_size)
        self.native_512 = bool(native_512)
        if self.native_512 and self.spatial_size != 32:
            raise ValueError('native512 geometry requires original32 base grid')
        self.motion_slots = int(motion_slots)
        self.use_clean_skip = bool(use_clean_skip)
        self.layer_gate_temperature = float(layer_gate_temperature)
        self.layer_gate_top_k = len(self.hidden_layers) if layer_gate_top_k is None else int(layer_gate_top_k)
        if self.geometry_dim % int(num_heads):
            raise ValueError("geometry_dim must be divisible by geometry attention heads")
        if self.layer_gate_temperature <= 0:
            raise ValueError("layer gate temperature must be positive")
        if not 1 <= self.layer_gate_top_k <= len(self.hidden_layers):
            raise ValueError("layer gate top-k must be within the selected hidden layers")
        hidden_dim = int(mapping.dit.config.num_attention_heads * mapping.dit.config.attention_head_dim)
        native_frames = WAN_LATENT_SHAPE[1] // int(mapping.dit.config.patch_size[0])
        self.native_frames = native_frames
        self.layer_projections = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, self.geometry_dim))
            for _ in self.hidden_layers
        ])
        if layer_gate_initial_logits is None:
            raise ValueError("layer_gate_initial_logits is required by the supported profile")
        initial_layer_logits = torch.as_tensor(layer_gate_initial_logits, dtype=torch.float32).clone()
        if initial_layer_logits.shape != (len(self.hidden_layers),):
            raise ValueError("layer_gate_initial_logits must match hidden_layers")
        if not torch.isfinite(initial_layer_logits).all():
            raise ValueError("layer gate initial logits must be finite")
        self.layer_logits = nn.Parameter(initial_layer_logits)
        self.clean_projection = nn.Conv3d(WAN_LATENT_SHAPE[0], self.geometry_dim, 1) \
            if self.use_clean_skip else None
        self.temporal_logits = nn.Parameter(_temporal_interpolation_logits(self.num_frames, native_frames))
        self.dense_frame_embedding = nn.Parameter(torch.randn(self.num_frames, self.geometry_dim) / math.sqrt(self.geometry_dim))
        self.native_frame_embedding = nn.Parameter(torch.randn(native_frames, self.geometry_dim) / math.sqrt(self.geometry_dim))
        self.dense_refine = ResidualBlock3D(self.geometry_dim)
        if self.motion_slots < 0:
            raise ValueError("motion_slots cannot be negative")
        if self.motion_slots:
            self.motion_slot_embedding = nn.Parameter(
                torch.randn(self.motion_slots, self.geometry_dim) / math.sqrt(self.geometry_dim)
            )
            self.motion_frame_embedding = nn.Parameter(
                torch.randn(self.num_frames, self.geometry_dim) / math.sqrt(self.geometry_dim)
            )
            self.motion_attention = nn.MultiheadAttention(self.geometry_dim, int(num_heads), batch_first=True)
            self.motion_norm = nn.LayerNorm(self.geometry_dim)
        else:
            # A true dense-only control should not retain unused slot parameters.
            self.register_parameter("motion_slot_embedding", None)
            self.register_parameter("motion_frame_embedding", None)
            self.motion_attention = None
            self.motion_norm = None

        # These modules only define Wan's RF output parameterization and are
        # deliberately outside the H005 forward graph.
        for name in ("norm_out", "proj_out"):
            for parameter in getattr(mapping.dit, name).parameters():
                parameter.requires_grad_(False)
        mapping.dit.scale_shift_table.requires_grad_(False)

    @property
    def dit(self) -> nn.Module:
        return self.mapping.dit

    @property
    def adapter_parameters(self) -> list[nn.Parameter]:
        mapping_ids = {id(parameter) for parameter in self.mapping.parameters()}
        return [parameter for parameter in self.parameters() if id(parameter) not in mapping_ids]

    @property
    def bypassed_parameters(self) -> list[nn.Parameter]:
        return [
            *self.mapping.dit.norm_out.parameters(),
            *self.mapping.dit.proj_out.parameters(),
            self.mapping.dit.scale_shift_table,
        ]

    def soft_layer_weights(self) -> torch.Tensor:
        """Differentiable dense gates used for entropy regularization and selection."""
        return (self.layer_logits.float() / self.layer_gate_temperature).softmax(dim=0)

    def layer_weights(self) -> torch.Tensor:
        """Return dense softmax or straight-through top-k fusion gates."""
        soft = self.soft_layer_weights()
        if self.layer_gate_top_k == len(self.hidden_layers):
            return soft
        indices = soft.topk(self.layer_gate_top_k).indices
        mask = torch.zeros_like(soft).scatter_(0, indices, 1.0)
        hard = soft * mask
        hard = hard / hard.sum().clamp_min(1e-12)
        if self.training:
            return hard.detach() - soft.detach() + soft
        return hard

    def layer_gate_entropy(self) -> torch.Tensor:
        weights = self.soft_layer_weights()
        return -(weights * weights.clamp_min(1e-12).log()).sum()

    def forward(self, clean_video_latent: torch.Tensor,
                encoder_hidden_states: torch.Tensor) -> StructuredZ4D:
        flow_time = torch.zeros(clean_video_latent.shape[0], device=clean_video_latent.device,
                                dtype=clean_video_latent.dtype)
        hidden_layers, grid_shape = self.mapping.forward_hidden_layers(
            clean_video_latent, flow_time, self.hidden_layers, encoder_hidden_states
        )
        if grid_shape[0] != self.native_frames:
            raise RuntimeError(f"Wan hidden temporal grid {grid_shape[0]} != {self.native_frames}")
        projected = torch.stack([
            projection(hidden) for projection, hidden in zip(self.layer_projections, hidden_layers)
        ], dim=0)
        weights = self.layer_weights().to(dtype=projected.dtype).reshape(-1, 1, 1, 1)
        fused = (projected * weights).sum(dim=0)
        batch = fused.shape[0]
        native_time, native_height, native_width = grid_shape

        native_tokens = fused.reshape(batch, native_time, native_height, native_width, self.geometry_dim)
        native_tokens = native_tokens + self.native_frame_embedding[None, :, None, None, :]
        if self.motion_slots:
            motion_query = (
                self.motion_frame_embedding[:, None, :] + self.motion_slot_embedding[None, :, :]
            ).reshape(1, self.num_frames * self.motion_slots, self.geometry_dim).expand(batch, -1, -1)
            motion, _ = self.motion_attention(
                motion_query, native_tokens.reshape(batch, -1, self.geometry_dim),
                native_tokens.reshape(batch, -1, self.geometry_dim), need_weights=False,
            )
            motion = self.motion_norm(motion + motion_query).reshape(
                batch, self.num_frames, self.motion_slots, self.geometry_dim
            )
        else:
            motion = fused.new_empty(batch, self.num_frames, 0, self.geometry_dim)

        dense_native = fused.reshape(
            batch, native_time, native_height, native_width, self.geometry_dim
        ).permute(0, 4, 1, 2, 3)
        spatial_size = self.spatial_size
        if self.native_512:
            if tuple(clean_video_latent.shape[-2:]) not in {(32, 32), (64, 64)}:
                raise ValueError('native geometry supports only256/512 square inputs')
            spatial_size = clean_video_latent.shape[-1]
        dense_native = F.interpolate(
            dense_native, size=(native_time, spatial_size, spatial_size),
            mode="trilinear", align_corners=False,
        )
        if self.clean_projection is not None:
            clean = self.clean_projection(clean_video_latent.to(dtype=dense_native.dtype))
            if clean.shape[-3:] != dense_native.shape[-3:]:
                clean = F.interpolate(clean, size=dense_native.shape[-3:], mode="trilinear", align_corners=False)
            dense_native = dense_native + clean
        temporal_weights = self.temporal_logits.softmax(dim=-1).to(dtype=dense_native.dtype)
        dense = torch.einsum("qn,bcnhw->bcqhw", temporal_weights, dense_native)
        dense = dense + self.dense_frame_embedding.T[None, :, :, None, None].to(dtype=dense.dtype)
        dense = self.dense_refine(dense)
        result = StructuredZ4D(dense=dense, motion=motion)
        result.validate()
        return result
