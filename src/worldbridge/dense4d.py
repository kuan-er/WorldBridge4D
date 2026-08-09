"""Dense `(source,target)` XYZ queries over a feed-forward Wan final latent.

The physical frame indices ``source``/``target`` are intentionally distinct from
Wan's rectified-flow time.  The latter is always exactly zero in this module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .wan import WAN_LATENT_SHAPE, WanDiTMapping


def flatten_z4d(z4d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten ``[B,C,T,H,W]`` in `(time,row,column)` order.

    Returns memory ``[B,T*H*W,C]`` and integer spatial coordinates
    ``[T*H*W,2]`` in ``(u,v)`` order.  Temporal replicas deliberately share
    coordinates because v1 applies no temporal RoPE.
    """
    if z4d.ndim != 5:
        raise ValueError(f"Z4D must be [B,C,T,H,W], got {tuple(z4d.shape)}")
    batch, channels, latent_time, height, width = z4d.shape
    memory = z4d.permute(0, 2, 3, 4, 1).reshape(batch, latent_time * height * width, channels)
    v, u = torch.meshgrid(
        torch.arange(height, device=z4d.device),
        torch.arange(width, device=z4d.device),
        indexing="ij",
    )
    spatial = torch.stack((u.reshape(-1), v.reshape(-1)), dim=-1)
    coordinates = spatial.repeat(latent_time, 1)
    return memory, coordinates


def unflatten_z4d(memory: torch.Tensor, latent_time: int, height: int, width: int) -> torch.Tensor:
    """Inverse of :func:`flatten_z4d`, used to assert memory ordering."""
    if memory.ndim != 3:
        raise ValueError(f"memory must be [B,N,C], got {tuple(memory.shape)}")
    expected = int(latent_time) * int(height) * int(width)
    if memory.shape[1] != expected:
        raise ValueError(f"memory tokens {memory.shape[1]} != {expected}")
    batch, _, channels = memory.shape
    return memory.reshape(batch, latent_time, height, width, channels).permute(0, 4, 1, 2, 3).contiguous()


@dataclass
class StructuredZ4D:
    """Decoder-facing geometry latent with local planes and motion slots.

    ``dense`` is ``[B,C,T,H,W]`` and is aligned to physical source frames.
    ``motion`` is ``[B,T,M,C]``; a fixed slot index is shared across time.
    """

    dense: torch.Tensor
    motion: torch.Tensor

    def validate(self) -> None:
        if self.dense.ndim != 5:
            raise ValueError(f"dense Z4D must be [B,C,T,H,W], got {tuple(self.dense.shape)}")
        if self.motion.ndim != 4:
            raise ValueError(f"motion Z4D must be [B,T,M,C], got {tuple(self.motion.shape)}")
        batch, channels, frames, _, _ = self.dense.shape
        if self.motion.shape[0] != batch or self.motion.shape[1] != frames or self.motion.shape[-1] != channels:
            raise ValueError("dense and motion Z4D batch/time/channel dimensions differ")

    @property
    def shape(self) -> torch.Size:
        """Dense shape compatibility for existing metric and logging code."""
        return self.dense.shape

    def shapes(self) -> dict[str, list[int]]:
        self.validate()
        return {"dense": list(self.dense.shape), "motion": list(self.motion.shape)}


def flatten_structured_z4d(z4d: StructuredZ4D) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten dense planes and append non-spatial motion tokens."""
    z4d.validate()
    dense_memory, dense_coordinates = flatten_z4d(z4d.dense)
    batch, frames, slots, channels = z4d.motion.shape
    motion_memory = z4d.motion.reshape(batch, frames * slots, channels)
    # Motion slots have no pixel location. Keeping them at the grid center
    # avoids assigning a false object position while retaining 2D-RoPE for the
    # dense memory. Physical time and slot identity are embedded in the values.
    center = dense_coordinates.float().mean(dim=0, keepdim=True)
    motion_coordinates = center.expand(frames * slots, -1)
    return torch.cat((dense_memory, motion_memory), dim=1), torch.cat(
        (dense_coordinates.to(dtype=center.dtype), motion_coordinates), dim=0
    )


class RotaryEmbedding2D(nn.Module):
    """Spatial-only 2D RoPE with half of each head assigned to each axis."""

    def __init__(self, head_dim: int, theta: float = 10_000.0):
        super().__init__()
        self.head_dim = int(head_dim)
        if self.head_dim % 4:
            raise ValueError(f"2D RoPE requires head_dim divisible by 4, got {head_dim}")
        self.axis_dim = self.head_dim // 2
        inv_freq = theta ** (-torch.arange(0, self.axis_dim, 2, dtype=torch.float32) / self.axis_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _axis(self, values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # values: [...,N,axis_dim], positions: [N]
        pairs = values.float().unflatten(-1, (-1, 2))
        angles = positions.float()[:, None] * self.inv_freq[None, :]
        shape = (1,) * (pairs.ndim - 3) + angles.shape
        cos = angles.cos().reshape(shape)
        sin = angles.sin().reshape(shape)
        first, second = pairs.unbind(-1)
        rotated = torch.stack((first * cos - second * sin, first * sin + second * cos), dim=-1)
        return rotated.flatten(-2).to(dtype=values.dtype)

    def forward(self, values: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != self.head_dim:
            raise ValueError(f"last dimension {values.shape[-1]} != RoPE head_dim {self.head_dim}")
        if coordinates.ndim != 2 or coordinates.shape != (values.shape[-2], 2):
            raise ValueError(f"coordinates must be [N,2] for N={values.shape[-2]}, got {tuple(coordinates.shape)}")
        x_axis, y_axis = values.split(self.axis_dim, dim=-1)
        coordinates = coordinates.to(device=values.device)
        return torch.cat((self._axis(x_axis, coordinates[:, 0]), self._axis(y_axis, coordinates[:, 1])), dim=-1)


class DenseCrossAttention(nn.Module):
    """Cross-attention from independent dense pair queries into full Z4D memory."""

    def __init__(self, query_dim: int, memory_dim: int, num_heads: int):
        super().__init__()
        self.query_dim = int(query_dim)
        self.memory_dim = int(memory_dim)
        self.num_heads = int(num_heads)
        if self.query_dim % self.num_heads:
            raise ValueError("query_dim must be divisible by num_heads")
        self.head_dim = self.query_dim // self.num_heads
        self.rope = RotaryEmbedding2D(self.head_dim)
        self.query_norm = nn.LayerNorm(self.query_dim)
        self.memory_norm = nn.LayerNorm(self.memory_dim)
        self.to_q = nn.Linear(self.query_dim, self.query_dim)
        self.to_k = nn.Linear(self.memory_dim, self.query_dim)
        self.to_v = nn.Linear(self.memory_dim, self.query_dim)
        self.to_out = nn.Linear(self.query_dim, self.query_dim)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        query_coordinates: torch.Tensor,
        memory_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        if query.ndim != 4:
            raise ValueError(f"query must be [B,K,Nq,C], got {tuple(query.shape)}")
        if memory.ndim != 3 or memory.shape[0] != query.shape[0]:
            raise ValueError("memory must be [B,Nm,Cm] with the same batch")
        batch, pairs, num_query, _ = query.shape
        num_memory = memory.shape[1]
        q = self.to_q(self.query_norm(query)).reshape(
            batch, pairs, num_query, self.num_heads, self.head_dim
        ).permute(0, 1, 3, 2, 4)
        k = self.to_k(self.memory_norm(memory)).reshape(
            batch, num_memory, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        v = self.to_v(self.memory_norm(memory)).reshape(
            batch, num_memory, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        q = self.rope(q, query_coordinates)
        k = self.rope(k, memory_coordinates)
        # Pair queries are independent. Expanding K/V is only a batch view; no
        # attention or normalization mixes pair indices.
        k = k[:, None].expand(-1, pairs, -1, -1, -1).reshape(
            batch * pairs, self.num_heads, num_memory, self.head_dim
        )
        v = v[:, None].expand(-1, pairs, -1, -1, -1).reshape(
            batch * pairs, self.num_heads, num_memory, self.head_dim
        )
        q = q.reshape(batch * pairs, self.num_heads, num_query, self.head_dim)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attended = attended.reshape(batch, pairs, self.num_heads, num_query, self.head_dim)
        attended = attended.permute(0, 1, 3, 2, 4).reshape(batch, pairs, num_query, self.query_dim)
        return self.to_out(attended)


class CrossAttentionBlock(nn.Module):
    """Pre-LN cross-attention and Pre-LN FFN, without query self-attention."""

    def __init__(self, query_dim: int, memory_dim: int, num_heads: int, ffn_ratio: float = 4.0):
        super().__init__()
        self.cross_attention = DenseCrossAttention(query_dim, memory_dim, num_heads)
        self.ffn_norm = nn.LayerNorm(query_dim)
        hidden = int(round(query_dim * ffn_ratio))
        self.ffn = nn.Sequential(nn.Linear(query_dim, hidden), nn.GELU(), nn.Linear(hidden, query_dim))

    def forward(self, query: torch.Tensor, memory: torch.Tensor,
                query_coordinates: torch.Tensor, memory_coordinates: torch.Tensor) -> torch.Tensor:
        query = query + self.cross_attention(query, memory, query_coordinates, memory_coordinates)
        return query + self.ffn(self.ffn_norm(query))


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return x + residual


class DenseUpsampler2D(nn.Module):
    """Bilinear 16->32->64->128 XYZ decoder (never transposed convolution)."""

    def __init__(self, query_dim: int = 256, channels: Sequence[int] = (256, 128, 64, 32),
                 latent_size: tuple[int, int] = (16, 16), output_size: tuple[int, int] = (128, 128),
                 fullres_coordinates: bool = False):
        super().__init__()
        channels = tuple(int(x) for x in channels)
        if len(channels) < 2:
            raise ValueError("upsample channels must contain a projection and at least one stage")
        factor = 2 ** (len(channels) - 1)
        if tuple(x * factor for x in latent_size) != tuple(output_size):
            raise ValueError(f"{latent_size} with {len(channels)-1} x2 stages does not produce {output_size}")
        self.latent_size = tuple(latent_size)
        self.output_size = tuple(output_size)
        self.fullres_coordinates = bool(fullres_coordinates)
        self.projection = nn.Conv2d(query_dim, channels[0], 3, padding=1)
        self.blocks = nn.ModuleList([
            ResidualBlock2D(in_channel, out_channel)
            for in_channel, out_channel in zip(channels[:-1], channels[1:])
        ])
        self.xyz = nn.Conv2d(channels[-1] + (2 if self.fullres_coordinates else 0), 3, 3, padding=1)
        if self.fullres_coordinates:
            v, u = torch.meshgrid(
                torch.linspace(-1.0, 1.0, self.output_size[0]),
                torch.linspace(-1.0, 1.0, self.output_size[1]),
                indexing="ij",
            )
            self.register_buffer(
                "fullres_uv",
                torch.stack((u, v), dim=0).unsqueeze(0),
                persistent=False,
            )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        x = self.projection(feature)
        for block in self.blocks:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
            x = block(x)
        if self.fullres_coordinates:
            x = torch.cat((x, self.fullres_uv.expand(x.shape[0], -1, -1, -1).to(dtype=x.dtype)), dim=1)
        x = self.xyz(x)
        if x.shape[-2:] != self.output_size:
            raise RuntimeError(f"upsampler output {tuple(x.shape[-2:])} != {self.output_size}")
        return x


@dataclass
class DenseQueryOutput:
    normalized_xyz: torch.Tensor
    low_resolution_feature: torch.Tensor
    coarse_normalized_xyz: torch.Tensor | None = None


class DenseQueryDecoder(nn.Module):
    """Map global native Z4D and K dense `(s,t)` queries to K XYZ maps."""

    def __init__(self, num_frames: int = 21, latent_shape: tuple[int, int, int, int] = WAN_LATENT_SHAPE,
                 query_dim: int = 256, embedding_dim: int = 128, num_layers: int = 2,
                 num_heads: int = 8, upsample_channels: Sequence[int] = (256, 128, 64, 32),
                 output_size: tuple[int, int] = (128, 128), coarse_diagnostic: bool = False,
                 fullres_coordinates: bool = False, query_grid_size: int | None = None,
                 structured_motion_slots: int = 0, structured_local_queries: bool = False):
        super().__init__()
        channels, latent_time, latent_height, latent_width = map(int, latent_shape)
        self.num_frames = int(num_frames)
        self.latent_shape = (channels, latent_time, latent_height, latent_width)
        self.query_dim = int(query_dim)
        self.source_embedding = nn.Embedding(self.num_frames, embedding_dim)
        self.target_embedding = nn.Embedding(self.num_frames, embedding_dim)
        self.query_mlp = nn.Sequential(
            nn.Linear(2 * embedding_dim, query_dim), nn.SiLU(), nn.Linear(query_dim, query_dim)
        )
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(query_dim, channels, num_heads) for _ in range(int(num_layers))
        ])
        self.structured_motion_slots = int(structured_motion_slots)
        self.structured_local_queries = bool(structured_local_queries)
        if self.structured_local_queries and self.structured_motion_slots <= 0:
            raise ValueError("structured local queries require at least one motion slot")
        self.source_local_projection = nn.Conv2d(channels, query_dim, 1) \
            if self.structured_local_queries else None
        query_grid_size = int(query_grid_size or latent_height)
        if query_grid_size < latent_height:
            raise ValueError("query_grid_size cannot be smaller than the Wan latent grid")
        self.query_grid_shape = (query_grid_size, query_grid_size)
        v, u = torch.meshgrid(
            torch.linspace(0, latent_height - 1, query_grid_size),
            torch.linspace(0, latent_width - 1, query_grid_size),
            indexing="ij",
        )
        self.register_buffer("query_coordinates", torch.stack((u.reshape(-1), v.reshape(-1)), dim=-1), persistent=False)
        self.upsampler = DenseUpsampler2D(
            query_dim, upsample_channels, self.query_grid_shape, output_size,
            fullres_coordinates=fullres_coordinates,
        )
        self.coarse_head = nn.Conv2d(query_dim, 3, 1) if coarse_diagnostic else None

    def query_content(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        source = torch.as_tensor(source, dtype=torch.long, device=self.source_embedding.weight.device)
        target = torch.as_tensor(target, dtype=torch.long, device=self.target_embedding.weight.device)
        if source.shape != target.shape:
            raise ValueError("source and target shapes differ")
        if source.ndim == 1:
            source, target = source[None], target[None]
        if source.ndim != 2:
            raise ValueError("source and target must be [B,K] or [K]")
        if (source < 0).any() or (source >= self.num_frames).any():
            raise ValueError("source index outside clip")
        if (target < 0).any() or (target >= self.num_frames).any():
            raise ValueError("target index outside clip")
        return self.query_mlp(torch.cat((self.source_embedding(source), self.target_embedding(target)), dim=-1))

    def _structured_source_query(self, z4d: StructuredZ4D, source: torch.Tensor, pairs: int) -> torch.Tensor:
        if self.source_local_projection is None:
            raise RuntimeError("structured local projection was not constructed")
        batch, channels, frames, height, width = z4d.dense.shape
        source = torch.as_tensor(source, device=z4d.dense.device, dtype=torch.long)
        if source.ndim == 1:
            source = source[None]
        if source.shape[0] == 1:
            source = source.expand(batch, -1)
        if source.shape != (batch, pairs) or (source < 0).any() or (source >= frames).any():
            raise ValueError("structured source indices do not match dense Z4D")
        by_time = z4d.dense.permute(0, 2, 1, 3, 4)
        local = by_time[torch.arange(batch, device=z4d.dense.device)[:, None], source]
        local = self.source_local_projection(local.reshape(batch * pairs, channels, height, width))
        if local.shape[-2:] != self.query_grid_shape:
            local = F.interpolate(local, size=self.query_grid_shape, mode="bilinear", align_corners=False)
        return local.flatten(2).transpose(1, 2).reshape(batch, pairs, -1, self.query_dim)

    def forward(self, z4d: torch.Tensor | StructuredZ4D, source: torch.Tensor,
                target: torch.Tensor) -> DenseQueryOutput:
        structured = isinstance(z4d, StructuredZ4D)
        dense = z4d.dense if structured else z4d
        if dense.ndim != 5 or tuple(dense.shape[1:]) != self.latent_shape:
            raise ValueError(f"Z4D dense tensor must be [B,{','.join(map(str, self.latent_shape))}], got {tuple(dense.shape)}")
        if structured:
            z4d.validate()
            if z4d.motion.shape[2] != self.structured_motion_slots:
                raise ValueError(
                    f"motion slots {z4d.motion.shape[2]} != decoder slots {self.structured_motion_slots}"
                )
        content = self.query_content(source, target)
        if content.shape[0] not in (1, dense.shape[0]):
            raise ValueError("query batch does not match Z4D batch")
        content = content.expand(dense.shape[0], -1, -1)
        num_query = self.query_coordinates.shape[0]
        query = content[:, :, None, :].expand(-1, -1, num_query, -1)
        if structured and self.structured_local_queries:
            query = query + self._structured_source_query(z4d, source, content.shape[1])
        memory, memory_coordinates = flatten_structured_z4d(z4d) if structured else flatten_z4d(z4d)
        for block in self.blocks:
            query = block(query, memory, self.query_coordinates, memory_coordinates)
        batch, pairs, _, _ = query.shape
        query_height, query_width = self.query_grid_shape
        feature = query.reshape(batch * pairs, query_height, query_width, self.query_dim).permute(0, 3, 1, 2)
        coarse = self.coarse_head(feature).reshape(batch, pairs, 3, query_height, query_width) \
            if self.coarse_head is not None else None
        xyz = self.upsampler(feature).reshape(batch, pairs, 3, *self.upsampler.output_size)
        feature = feature.reshape(batch, pairs, self.query_dim, query_height, query_width)
        return DenseQueryOutput(xyz, feature, coarse)


class CleanLatentBackbone(nn.Module):
    """Control readout returning the frozen clean VAE latent unchanged."""

    raw_velocity_convention = "not_applicable"
    z4d_transform = "clean_latent_identity"

    def forward(self, clean_video_latent: torch.Tensor) -> torch.Tensor:
        return clean_video_latent


class FeedForwardWanBackbone(nn.Module):
    """Clean latent -> Wan at exact RF time zero -> negative raw velocity."""

    raw_velocity_convention = "epsilon_minus_clean"
    z4d_transform = "negative_raw_velocity"

    def __init__(self, mapping: WanDiTMapping):
        super().__init__()
        self.mapping = mapping

    @property
    def dit(self) -> nn.Module:
        return self.mapping.dit

    def forward(self, clean_video_latent: torch.Tensor) -> torch.Tensor:
        flow_time = torch.zeros(clean_video_latent.shape[0], device=clean_video_latent.device,
                                dtype=clean_video_latent.dtype)
        raw_velocity = self.mapping(clean_video_latent, flow_time)
        z4d = -raw_velocity
        if z4d.shape != clean_video_latent.shape:
            raise RuntimeError(f"Z4D {tuple(z4d.shape)} != clean latent {tuple(clean_video_latent.shape)}")
        return z4d


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
                 layer_gate_temperature: float = 1.0, layer_gate_top_k: int | None = None):
        super().__init__()
        self.mapping = mapping
        self.hidden_layers = tuple(int(index) for index in hidden_layers)
        self.geometry_dim = int(geometry_dim)
        self.num_frames = int(num_frames)
        self.spatial_size = int(spatial_size)
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
        self.layer_logits = nn.Parameter(torch.zeros(len(self.hidden_layers)))
        self.clean_projection = nn.Conv3d(WAN_LATENT_SHAPE[0], self.geometry_dim, 1) \
            if self.use_clean_skip else None
        self.temporal_logits = nn.Parameter(_temporal_interpolation_logits(self.num_frames, native_frames))
        self.dense_frame_embedding = nn.Parameter(torch.randn(self.num_frames, self.geometry_dim) / math.sqrt(self.geometry_dim))
        self.native_frame_embedding = nn.Parameter(torch.randn(native_frames, self.geometry_dim) / math.sqrt(self.geometry_dim))
        self.dense_refine = ResidualBlock3D(self.geometry_dim)
        self.motion_slot_embedding = nn.Parameter(
            torch.randn(self.motion_slots, self.geometry_dim) / math.sqrt(self.geometry_dim)
        )
        self.motion_frame_embedding = nn.Parameter(
            torch.randn(self.num_frames, self.geometry_dim) / math.sqrt(self.geometry_dim)
        )
        self.motion_attention = nn.MultiheadAttention(self.geometry_dim, int(num_heads), batch_first=True)
        self.motion_norm = nn.LayerNorm(self.geometry_dim)

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

    def forward(self, clean_video_latent: torch.Tensor) -> StructuredZ4D:
        flow_time = torch.zeros(clean_video_latent.shape[0], device=clean_video_latent.device,
                                dtype=clean_video_latent.dtype)
        hidden_layers, grid_shape = self.mapping.forward_hidden_layers(
            clean_video_latent, flow_time, self.hidden_layers
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

        dense_native = fused.reshape(
            batch, native_time, native_height, native_width, self.geometry_dim
        ).permute(0, 4, 1, 2, 3)
        dense_native = F.interpolate(
            dense_native, size=(native_time, self.spatial_size, self.spatial_size),
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


class DenseQueryWanModel(nn.Module):
    def __init__(self, backbone: nn.Module, decoder: DenseQueryDecoder):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder

    def forward(self, clean_video_latent: torch.Tensor, source: torch.Tensor,
                target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | StructuredZ4D, DenseQueryOutput]:
        z4d = self.backbone(clean_video_latent)
        output = self.decoder(z4d, source, target)
        return output.normalized_xyz, z4d, output

    def configure_trainable(self, mode: str = "full", last_blocks: int = 2) -> None:
        mode = str(mode)
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        bypassed = list(getattr(self.backbone, "bypassed_parameters", []))
        for parameter in bypassed:
            parameter.requires_grad_(False)
        if mode == "full":
            return
        if mode == "decoder_only":
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            return
        if mode == "geometry_adapter":
            adapter = list(getattr(self.backbone, "adapter_parameters", []))
            if not adapter:
                raise ValueError("geometry_adapter mode requires a structured geometry backbone")
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            for parameter in adapter:
                parameter.requires_grad_(True)
            return
        if mode == "last_blocks":
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            dit = getattr(self.backbone, "dit", None)
            if dit is None or not hasattr(dit, "blocks"):
                raise ValueError("last_blocks mode requires a Wan-like backbone.dit.blocks")
            for parameter in getattr(self.backbone, "adapter_parameters", []):
                parameter.requires_grad_(True)
            for block in dit.blocks[-int(last_blocks):]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
            if not bypassed:
                for name in ("norm_out", "proj_out", "scale_shift_table"):
                    module_or_parameter = getattr(dit, name, None)
                    if isinstance(module_or_parameter, nn.Parameter):
                        module_or_parameter.requires_grad_(True)
                    elif isinstance(module_or_parameter, nn.Module):
                        for parameter in module_or_parameter.parameters():
                            parameter.requires_grad_(True)
            return
        raise ValueError(f"unknown trainable_mode={mode!r}")


def masked_pair_smooth_l1(prediction: torch.Tensor, target: torch.Tensor,
                          validity: torch.Tensor, beta: float = 0.05) -> torch.Tensor:
    """Exact pair-wise validity-masked SmoothL1 average from the H004 spec."""
    if prediction.shape != target.shape or prediction.ndim != 5 or prediction.shape[2] != 3:
        raise ValueError("prediction/target must match [B,K,3,H,W]")
    if validity.shape != prediction.shape[:2] + prediction.shape[-2:]:
        raise ValueError("validity must be [B,K,H,W]")
    error = F.smooth_l1_loss(prediction, target, beta=beta, reduction="none").sum(dim=2)
    mask = validity.to(dtype=error.dtype)
    per_pair = (error * mask).sum(dim=(-2, -1)) / mask.sum(dim=(-2, -1)).clamp_min(1.0)
    return per_pair.mean()


def verify_flow_velocity_algebra(device: torch.device | str = "cpu") -> dict[str, float | str]:
    """Tie the Wan sign decision to the scheduler's executable RF equations."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    device = torch.device(device)
    clean = torch.tensor([[[[2.0, -1.0]]]], device=device)
    noise = torch.tensor([[[[-3.0, 4.0]]]], device=device)
    raw_velocity = noise - clean
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
    scheduler.set_timesteps(sigmas=[1.0], device=device)
    timestep = scheduler.timesteps[0]
    recovered = scheduler.step(raw_velocity, timestep, noise, return_dict=False)[0]
    maximum_error = float((recovered - clean).abs().max())
    if maximum_error > 1e-6:
        raise AssertionError(f"Wan RF velocity algebra failed: {maximum_error}")
    return {
        "forward_path": "x_sigma=(1-sigma)*clean+sigma*noise",
        "raw_velocity": "noise-clean",
        "perception_readout": "negative_raw_velocity",
        "maximum_clean_recovery_error": maximum_error,
    }
