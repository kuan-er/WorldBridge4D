"""Independent dense source-target query decoder."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from ..outputs import (
    DenseQueryOutput, StructuredZ4D, flatten_structured_z4d, flatten_z4d,
)
from ..wan import WAN_LATENT_SHAPE
from .blocks import CrossAttentionBlock, _group_count
from .upsampler import DenseUpsampler2D

class DenseQueryDecoder(nn.Module):
    """Map global native Z4D and K dense `(s,t)` queries to K XYZ maps."""

    def __init__(self, num_frames: int = 21, latent_shape: tuple[int, int, int, int] = WAN_LATENT_SHAPE,
                 query_dim: int = 256, embedding_dim: int = 128, num_layers: int = 2,
                 num_heads: int = 8, upsample_channels: Sequence[int] = (256, 128, 64, 32),
                 output_size: tuple[int, int] = (128, 128), coarse_diagnostic: bool = False,
                 fullres_coordinates: bool = False, query_grid_size: int | None = None,
                 structured_motion_slots: int = 0, structured_local_queries: bool = False,
                 structured_pair_motion_queries: bool = False,
                 structured_pair_motion_zero_init: bool = False,
                 source_rgb_pyramid: bool = False,
                 source_rgb_channels: Sequence[int] = (32, 64, 128),
                 source_rgb_fusion_32: bool = False,
                 pre_attention_rgb_query: bool = False):
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
        self.structured_pair_motion_queries = bool(structured_pair_motion_queries)
        self.structured_pair_motion_zero_init = bool(structured_pair_motion_zero_init)
        if self.structured_motion_slots < 0:
            raise ValueError("structured motion slot count cannot be negative")
        if self.structured_pair_motion_queries and self.structured_motion_slots == 0:
            raise ValueError("pair-conditioned motion queries require motion slots")
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
            source_rgb_pyramid=source_rgb_pyramid,
            source_rgb_channels=source_rgb_channels,
            source_rgb_fusion_32=source_rgb_fusion_32,
        )
        self.coarse_head = nn.Conv2d(query_dim, 3, 1) if coarse_diagnostic else None
        # Constructed after all baseline modules so enabling this ablation does
        # not shift the seeded initialization of any shared decoder parameter.
        self.motion_pair_projection = nn.Sequential(
            nn.LayerNorm(3 * channels), nn.Linear(3 * channels, query_dim),
            nn.SiLU(), nn.Linear(query_dim, query_dim),
        ) if self.structured_pair_motion_queries else None
        if self.motion_pair_projection is not None and self.structured_pair_motion_zero_init:
            nn.init.zeros_(self.motion_pair_projection[-1].weight)
            nn.init.zeros_(self.motion_pair_projection[-1].bias)
        self.pre_attention_rgb_query = bool(pre_attention_rgb_query)
        if self.pre_attention_rgb_query:
            if not source_rgb_pyramid:
                raise ValueError("pre-attention RGB query requires the source RGB pyramid")
            if self.query_grid_shape != (32, 32):
                raise ValueError("pre-attention RGB query requires a 32x32 query grid")
            rgb_channels = int(tuple(source_rgb_channels)[-1])
            self.query_rgb_projection = nn.Sequential(
                nn.GroupNorm(_group_count(rgb_channels), rgb_channels, affine=False),
                nn.SiLU(),
                nn.Conv2d(rgb_channels, self.query_dim, 1, bias=False),
            )
            # The migrated step-100k function is initially exact. Unlike the
            # upsampler fusions this is an unconditional additive query
            # residual: the projection itself learns when RGB should matter.
            nn.init.zeros_(self.query_rgb_projection[-1].weight)
        else:
            self.query_rgb_projection = None

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
        batch_indices = torch.arange(batch, device=z4d.dense.device)
        same_source_per_clip = bool(torch.all(source == source[:, :1]))
        if same_source_per_clip:
            # Canonical source-all-target supervision repeats one source for all
            # K targets. Project its local plane once, then let autograd sum the
            # gradients through the expanded pair view instead of recomputing
            # the identical 1x1 convolution K times.
            local = by_time[batch_indices, source[:, 0]]
            local = self.source_local_projection(local)
            if local.shape[-2:] != self.query_grid_shape:
                local = F.interpolate(local, size=self.query_grid_shape, mode="bilinear", align_corners=False)
            local = local.flatten(2).transpose(1, 2)[:, None]
            return local.expand(-1, pairs, -1, -1)
        local = by_time[batch_indices[:, None], source]
        local = self.source_local_projection(local.reshape(batch * pairs, channels, height, width))
        if local.shape[-2:] != self.query_grid_shape:
            local = F.interpolate(local, size=self.query_grid_shape, mode="bilinear", align_corners=False)
        return local.flatten(2).transpose(1, 2).reshape(batch, pairs, -1, self.query_dim)

    def _structured_pair_motion_query(
        self, z4d: StructuredZ4D, source: torch.Tensor, target: torch.Tensor, pairs: int,
    ) -> torch.Tensor:
        if self.motion_pair_projection is None:
            raise RuntimeError("pair-conditioned motion projection was not constructed")
        batch, frames, slots, channels = z4d.motion.shape
        if slots == 0:
            raise ValueError("pair-conditioned motion query received no slots")
        source = torch.as_tensor(source, device=z4d.motion.device, dtype=torch.long)
        target = torch.as_tensor(target, device=z4d.motion.device, dtype=torch.long)
        if source.ndim == 1:
            source, target = source[None], target[None]
        if source.shape[0] == 1:
            source, target = source.expand(batch, -1), target.expand(batch, -1)
        if source.shape != (batch, pairs) or target.shape != (batch, pairs):
            raise ValueError("structured pair indices do not match motion Z4D")
        batch_indices = torch.arange(batch, device=z4d.motion.device)[:, None]
        source_motion = z4d.motion[batch_indices, source].mean(dim=2)
        target_motion = z4d.motion[batch_indices, target].mean(dim=2)
        pair_motion = torch.cat(
            (source_motion, target_motion, target_motion - source_motion), dim=-1
        )
        return self.motion_pair_projection(pair_motion)

    def encode_source_rgb(self, source_rgb: torch.Tensor) -> dict[int, torch.Tensor]:
        return self.upsampler.encode_source_rgb(source_rgb)

    def forward(self, z4d: torch.Tensor | StructuredZ4D, source: torch.Tensor,
                target: torch.Tensor, source_rgb: torch.Tensor | None = None,
                source_pyramid: dict[int, torch.Tensor] | None = None
                ) -> DenseQueryOutput:
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
        if source_rgb is not None and source_pyramid is not None:
            raise ValueError("pass source_rgb or source_pyramid, not both")
        if self.pre_attention_rgb_query:
            if source_pyramid is None:
                if source_rgb is None:
                    raise ValueError("pre-attention RGB query requires source appearance")
                source_pyramid = self.encode_source_rgb(source_rgb)
                source_rgb = None
            if 32 not in source_pyramid:
                raise ValueError("source RGB pyramid lacks the 32px feature")
        content = self.query_content(source, target)
        if content.shape[0] not in (1, dense.shape[0]):
            raise ValueError("query batch does not match Z4D batch")
        content = content.expand(dense.shape[0], -1, -1)
        num_query = self.query_coordinates.shape[0]
        query = content[:, :, None, :].expand(-1, -1, num_query, -1)
        if structured and self.structured_local_queries:
            query = query + self._structured_source_query(z4d, source, content.shape[1])
        if structured and self.structured_pair_motion_queries:
            pair_motion = self._structured_pair_motion_query(
                z4d, source, target, content.shape[1]
            )
            query = query + pair_motion[:, :, None, :]
        if self.query_rgb_projection is not None:
            rgb_query = self.query_rgb_projection(source_pyramid[32])
            if rgb_query.shape != (
                dense.shape[0], self.query_dim, *self.query_grid_shape,
            ):
                raise RuntimeError(
                    f"RGB query projection has unexpected shape {tuple(rgb_query.shape)}"
                )
            rgb_query = rgb_query.flatten(2).transpose(1, 2)[:, None]
            query = query + rgb_query.expand(-1, content.shape[1], -1, -1)
        memory, memory_coordinates = flatten_structured_z4d(z4d) if structured else flatten_z4d(z4d)
        for block in self.blocks:
            query = block(query, memory, self.query_coordinates, memory_coordinates)
        batch, pairs, _, _ = query.shape
        query_height, query_width = self.query_grid_shape
        feature = query.reshape(batch * pairs, query_height, query_width, self.query_dim).permute(0, 3, 1, 2)
        coarse = self.coarse_head(feature).reshape(batch, pairs, 3, query_height, query_width) \
            if self.coarse_head is not None else None
        xyz = self.upsampler(
            feature, source_rgb=source_rgb, source_pyramid=source_pyramid,
            batch=batch, pairs=pairs,
        ).reshape(batch, pairs, 3, *self.upsampler.output_size)
        feature = feature.reshape(batch, pairs, self.query_dim, query_height, query_width)
        return DenseQueryOutput(xyz, feature, coarse)
