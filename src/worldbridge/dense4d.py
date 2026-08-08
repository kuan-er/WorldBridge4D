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


class RotaryEmbedding3D(nn.Module):
    """3D RoPE with 8 rotary dimensions each for time/u/v and 8 passthrough."""

    def __init__(self, head_dim: int, theta: float = 10_000.0):
        super().__init__()
        self.head_dim = int(head_dim)
        self.axis_dim = 8
        self.rotary_dim = 3 * self.axis_dim
        if self.head_dim != 32:
            raise ValueError(f"H004 E3 fixes head_dim=32, got {head_dim}")
        inv_freq = theta ** (-torch.arange(0, self.axis_dim, 2, dtype=torch.float32) / self.axis_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rotate_axis(self, values: torch.Tensor, positions: torch.Tensor, axis: int) -> torch.Tensor:
        pairs = values.float().unflatten(-1, (self.axis_dim // 2, 2))
        angles = positions[..., axis, None, None].float() * self.inv_freq
        cos, sin = angles.cos(), angles.sin()
        first, second = pairs.unbind(-1)
        rotated = torch.stack((first * cos - second * sin, first * sin + second * cos), dim=-1)
        return rotated.flatten(-2).to(dtype=values.dtype)

    def forward(self, values: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != self.head_dim:
            raise ValueError(f"last dimension {values.shape[-1]} != RoPE head_dim {self.head_dim}")
        if coordinates.shape[-1] != 3 or coordinates.shape[:-1] != values.shape[:-2]:
            raise ValueError(
                f"3D coordinates must match values prefix/N {tuple(values.shape[:-2])}, got {tuple(coordinates.shape)}"
            )
        chunks = values.split(self.axis_dim, dim=-1)
        rotated = [self._rotate_axis(chunks[axis], coordinates, axis) for axis in range(3)]
        return torch.cat((*rotated, chunks[3]), dim=-1)


class DenseCrossAttention(nn.Module):
    """Cross-attention from independent dense pair queries into full Z4D memory."""

    def __init__(self, query_dim: int, memory_dim: int, num_heads: int, rope_mode: str = "2d"):
        super().__init__()
        self.query_dim = int(query_dim)
        self.memory_dim = int(memory_dim)
        self.num_heads = int(num_heads)
        self.rope_mode = str(rope_mode)
        if self.query_dim % self.num_heads:
            raise ValueError("query_dim must be divisible by num_heads")
        self.head_dim = self.query_dim // self.num_heads
        if self.rope_mode == "2d":
            self.rope = RotaryEmbedding2D(self.head_dim)
        elif self.rope_mode == "3d":
            self.rope = RotaryEmbedding3D(self.head_dim)
        else:
            raise ValueError(f"unknown rope_mode={self.rope_mode!r}")
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
        if self.rope_mode == "2d":
            q = self.rope(q, query_coordinates)
            k = self.rope(k, memory_coordinates)
        else:
            q = self.rope(q.transpose(2, 3), query_coordinates).transpose(2, 3)
            k = self.rope(k.transpose(1, 2), memory_coordinates).transpose(1, 2)
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

    def __init__(self, query_dim: int, memory_dim: int, num_heads: int, ffn_ratio: float = 4.0,
                 rope_mode: str = "2d"):
        super().__init__()
        self.cross_attention = DenseCrossAttention(query_dim, memory_dim, num_heads, rope_mode=rope_mode)
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

    def forward_features(self, feature: torch.Tensor) -> torch.Tensor:
        x = self.projection(feature)
        for block in self.blocks:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
            x = block(x)
        if self.fullres_coordinates:
            x = torch.cat((x, self.fullres_uv.expand(x.shape[0], -1, -1, -1).to(dtype=x.dtype)), dim=1)
        return x

    def forward_with_features(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.forward_features(feature)
        xyz = self.xyz(x)
        if xyz.shape[-2:] != self.output_size:
            raise RuntimeError(f"upsampler output {tuple(xyz.shape[-2:])} != {self.output_size}")
        return xyz, x

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.forward_with_features(feature)[0]


@dataclass
class DenseQueryOutput:
    normalized_xyz: torch.Tensor
    low_resolution_feature: torch.Tensor
    coarse_normalized_xyz: torch.Tensor | None = None
    visibility_logits: torch.Tensor | None = None


class FixedChannelWhitening(nn.Module):
    """Fixed per-channel standardization for a latent covariate-shift control."""

    def __init__(self, channels: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(int(channels)), persistent=True)
        self.register_buffer("scale", torch.ones(int(channels)), persistent=True)

    def set_stats(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        mean = torch.as_tensor(mean, dtype=self.mean.dtype, device=self.mean.device).flatten()
        scale = torch.as_tensor(scale, dtype=self.scale.dtype, device=self.scale.device).flatten()
        if mean.shape != self.mean.shape or scale.shape != self.scale.shape:
            raise ValueError("latent whitening statistics must match the latent channel count")
        self.mean.copy_(mean)
        self.scale.copy_(scale.clamp_min(1e-6))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        shape = (1, -1, 1, 1, 1)
        return (latent - self.mean.to(dtype=latent.dtype).view(shape)) / self.scale.to(dtype=latent.dtype).view(shape)


class ChannelAffineAdapter(nn.Module):
    """Learnable channel-wise affine bridge, initialized as the identity."""

    def __init__(self, channels: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(int(channels)))
        self.bias = nn.Parameter(torch.zeros(int(channels)))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        shape = (1, -1, 1, 1, 1)
        return latent * self.scale.to(dtype=latent.dtype).view(shape) + self.bias.to(dtype=latent.dtype).view(shape)


class ConvLatentAdapter(nn.Module):
    """Small nonlinear 1x1x1 latent remapping for the D3 control."""

    def __init__(self, channels: int):
        super().__init__()
        channels = int(channels)
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, 1), nn.GELU(), nn.Conv3d(channels, channels, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return latent + self.net(latent)


def make_latent_adapter(kind: str, channels: int) -> nn.Module:
    kind = str(kind).lower()
    if kind in {"none", "identity"}:
        return nn.Identity()
    if kind in {"fixed_whiten", "whiten"}:
        return FixedChannelWhitening(channels)
    if kind in {"channel_affine", "affine"}:
        return ChannelAffineAdapter(channels)
    if kind in {"conv1x1", "nonlinear"}:
        return ConvLatentAdapter(channels)
    raise ValueError(f"unknown latent_adapter={kind!r}")


class DenseQueryDecoder(nn.Module):
    """Map global native Z4D and K dense `(s,t)` queries to K XYZ maps."""

    def __init__(self, num_frames: int = 21, latent_shape: tuple[int, int, int, int] = WAN_LATENT_SHAPE,
                 query_dim: int = 256, embedding_dim: int = 128, num_layers: int = 2,
                 num_heads: int = 8, upsample_channels: Sequence[int] = (256, 128, 64, 32),
                 output_size: tuple[int, int] = (128, 128), coarse_diagnostic: bool = False,
                 fullres_coordinates: bool = False, query_grid_size: int | None = None,
                 rope_mode: str = "2d", visibility_head: bool = False,
                 latent_adapter: str = "none"):
        super().__init__()
        channels, latent_time, latent_height, latent_width = map(int, latent_shape)
        self.num_frames = int(num_frames)
        self.latent_shape = (channels, latent_time, latent_height, latent_width)
        self.query_dim = int(query_dim)
        self.rope_mode = str(rope_mode)
        self.visibility_head_enabled = bool(visibility_head)
        self.source_embedding = nn.Embedding(self.num_frames, embedding_dim)
        self.target_embedding = nn.Embedding(self.num_frames, embedding_dim)
        self.query_mlp = nn.Sequential(
            nn.Linear(2 * embedding_dim, query_dim), nn.SiLU(), nn.Linear(query_dim, query_dim)
        )
        num_layers = int(num_layers)
        if num_layers < 1:
            raise ValueError("decoder requires at least one cross-attention block")
        # Construct the two common screening blocks before every variant-only
        # module.  E6's extra blocks are appended only after the common
        # upsampler is initialized, so overlapping B0/E3/E5/E6 weights are
        # bit-identical under the shared decoder seed.
        common_layers = min(num_layers, 2)
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(query_dim, channels, num_heads, rope_mode=self.rope_mode)
            for _ in range(common_layers)
        ])
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
        self.visibility_head = nn.Conv2d(int(upsample_channels[-1]), 1, 3, padding=1) \
            if self.visibility_head_enabled else None
        if num_layers > common_layers:
            self.blocks.extend([
                CrossAttentionBlock(query_dim, channels, num_heads, rope_mode=self.rope_mode)
                for _ in range(num_layers - common_layers)
            ])
        # Construct latent adapters last so B0/E3/E5/E6 common decoder weights
        # remain bit-identical under decoder_seed=424242.  D2 is identity at
        # initialization; D1 receives fixed statistics before training starts.
        self.latent_adapter_kind = str(latent_adapter).lower()
        self.latent_adapter = make_latent_adapter(self.latent_adapter_kind, channels)

    def set_latent_stats(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        if not hasattr(self.latent_adapter, "set_stats"):
            raise ValueError("latent statistics are only supported by fixed_whiten")
        self.latent_adapter.set_stats(mean, scale)

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

    def forward(self, z4d: torch.Tensor, source: torch.Tensor, target: torch.Tensor) -> DenseQueryOutput:
        if z4d.ndim != 5 or tuple(z4d.shape[1:]) != self.latent_shape:
            raise ValueError(f"Z4D must be [B,{','.join(map(str, self.latent_shape))}], got {tuple(z4d.shape)}")
        z4d = self.latent_adapter(z4d)
        content = self.query_content(source, target)
        if content.shape[0] not in (1, z4d.shape[0]):
            raise ValueError("query batch does not match Z4D batch")
        content = content.expand(z4d.shape[0], -1, -1)
        num_query = self.query_coordinates.shape[0]
        query = content[:, :, None, :].expand(-1, -1, num_query, -1)
        memory, memory_spatial_coordinates = flatten_z4d(z4d)
        if self.rope_mode == "3d":
            latent_time = self.latent_shape[1]
            spatial_tokens = self.latent_shape[2] * self.latent_shape[3]
            memory_time = torch.arange(latent_time, device=z4d.device, dtype=z4d.dtype) \
                .repeat_interleave(spatial_tokens)
            memory_coordinates = torch.cat(
                (memory_time[:, None], memory_spatial_coordinates.to(dtype=z4d.dtype)), dim=-1
            )[None].expand(z4d.shape[0], -1, -1)
            target_for_coordinates = torch.as_tensor(target, device=z4d.device, dtype=z4d.dtype)
            if target_for_coordinates.ndim == 1:
                target_for_coordinates = target_for_coordinates[None]
            if target_for_coordinates.shape[0] == 1 and z4d.shape[0] != 1:
                target_for_coordinates = target_for_coordinates.expand(z4d.shape[0], -1)
            query_spatial = self.query_coordinates.to(device=z4d.device, dtype=z4d.dtype)
            query_spatial = query_spatial[None, None].expand(z4d.shape[0], target_for_coordinates.shape[1], -1, -1)
            query_time = target_for_coordinates[..., None, None] * ((latent_time - 1) / max(self.num_frames - 1, 1))
            query_time = query_time.expand(-1, -1, query_spatial.shape[2], 1)
            query_coordinates = torch.cat((query_time, query_spatial), dim=-1)
        else:
            memory_coordinates = memory_spatial_coordinates
            query_coordinates = self.query_coordinates
        for block in self.blocks:
            query = block(query, memory, query_coordinates, memory_coordinates)
        batch, pairs, _, _ = query.shape
        query_height, query_width = self.query_grid_shape
        feature = query.reshape(batch * pairs, query_height, query_width, self.query_dim).permute(0, 3, 1, 2)
        coarse = self.coarse_head(feature).reshape(batch, pairs, 3, query_height, query_width) \
            if self.coarse_head is not None else None
        xyz, fullres_feature = self.upsampler.forward_with_features(feature)
        xyz = xyz.reshape(batch, pairs, 3, *self.upsampler.output_size)
        visibility_logits = None
        if self.visibility_head is not None:
            visibility_logits = self.visibility_head(fullres_feature).reshape(
                batch, pairs, 1, *self.upsampler.output_size
            )
        feature = feature.reshape(batch, pairs, self.query_dim, query_height, query_width)
        return DenseQueryOutput(xyz, feature, coarse, visibility_logits)


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


class DenseQueryWanModel(nn.Module):
    def __init__(self, backbone: nn.Module, decoder: DenseQueryDecoder):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder

    def forward(self, clean_video_latent: torch.Tensor, source: torch.Tensor,
                target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, DenseQueryOutput]:
        z4d = self.backbone(clean_video_latent)
        output = self.decoder(z4d, source, target)
        return output.normalized_xyz, z4d, output

    def configure_trainable(self, mode: str = "full", last_blocks: int = 2) -> None:
        mode = str(mode)
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        if mode == "full":
            return
        if mode == "decoder_only":
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            return
        if mode == "last_blocks":
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            dit = getattr(self.backbone, "dit", None)
            if dit is None or not hasattr(dit, "blocks"):
                raise ValueError("last_blocks mode requires a Wan-like backbone.dit.blocks")
            for block in dit.blocks[-int(last_blocks):]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
            for name in ("norm_out", "proj_out", "scale_shift_table"):
                module_or_parameter = getattr(dit, name, None)
                if isinstance(module_or_parameter, nn.Parameter):
                    module_or_parameter.requires_grad_(True)
                elif isinstance(module_or_parameter, nn.Module):
                    for parameter in module_or_parameter.parameters():
                        parameter.requires_grad_(True)
            return
        raise ValueError(f"unknown trainable_mode={mode!r}")


def masked_visibility_bce(logits: torch.Tensor, target_visible: torch.Tensor,
                          validity: torch.Tensor, source: torch.Tensor, target: torch.Tensor,
                          pos_weight: float | torch.Tensor) -> torch.Tensor:
    """Off-diagonal M-target BCE masked only by A; predictions never affect XYZ."""
    if logits.ndim == 5 and logits.shape[2] == 1:
        logits = logits[:, :, 0]
    if logits.ndim != 4 or target_visible.shape != logits.shape or validity.shape != logits.shape:
        raise ValueError("visibility tensors must all be [B,K,H,W]")
    if source.shape != target.shape or source.shape != logits.shape[:2]:
        raise ValueError("source/target must be [B,K]")
    off_diagonal = (source != target).to(device=logits.device)[:, :, None, None]
    mask = validity.to(device=logits.device, dtype=logits.dtype) * off_diagonal
    if not bool(mask.any()):
        return logits.sum() * 0.0
    weight = torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(
        logits, target_visible.to(device=logits.device, dtype=logits.dtype),
        pos_weight=weight, reduction="none",
    )
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


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
