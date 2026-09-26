"""Dense 2D residual upsampler."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .blocks import ResidualBlock2D
from .source_rgb import GatedSourceFusion, SourceRGBPyramid

class DenseUpsampler2D(nn.Module):
    """Bilinear residual XYZ decoder with optional source-RGB pyramid fusion."""

    def __init__(self, query_dim: int = 256, channels: Sequence[int] = (256, 128, 64, 32),
                 latent_size: tuple[int, int] = (16, 16), output_size: tuple[int, int] = (128, 128),
                 source_rgb_pyramid: bool = False,
                 source_rgb_channels: Sequence[int] = (32, 64, 128),
                 source_rgb_fusion_32: bool = False, native_512: bool = False):
        super().__init__()
        channels = tuple(int(x) for x in channels)
        if len(channels) < 2:
            raise ValueError("upsample channels must contain a projection and at least one stage")
        factor = 2 ** (len(channels) - 1)
        if tuple(x * factor for x in latent_size) != tuple(output_size):
            raise ValueError(f"{latent_size} with {len(channels)-1} x2 stages does not produce {output_size}")
        self.native_512 = bool(native_512)
        if self.native_512 and (latent_size != (32, 32) or output_size != (256, 256)
                                or not source_rgb_pyramid):
            raise ValueError('native512 extends the checkpoint-stable RGB32-to256 decoder only')
        self.latent_size = tuple(latent_size)
        self.output_size = tuple(output_size)
        self.source_rgb_pyramid_enabled = bool(source_rgb_pyramid)
        self.source_rgb_fusion_32 = bool(source_rgb_fusion_32)
        self.projection = nn.Conv2d(query_dim, channels[0], 3, padding=1)
        self.blocks = nn.ModuleList([
            ResidualBlock2D(in_channel, out_channel)
            for in_channel, out_channel in zip(channels[:-1], channels[1:])
        ])
        self.source_rgb_encoder: SourceRGBPyramid | None = None
        self.source_fusions = nn.ModuleDict()
        self.stage_scales = tuple(
            int(self.latent_size[0] * 2 ** (index + 1)) for index in range(len(self.blocks))
        )
        if self.source_rgb_pyramid_enabled:
            if self.latent_size != (32, 32) or self.output_size != (256, 256) \
                    or self.stage_scales != (64, 128, 256):
                raise ValueError("source RGB pyramid requires the 32->64->128->256 decoder")
            self.source_rgb_encoder = SourceRGBPyramid(source_rgb_channels, native_512=self.native_512)
            if self.source_rgb_fusion_32:
                self.source_fusions["32"] = GatedSourceFusion(
                    self.source_rgb_encoder.channels_by_scale[32], channels[0],
                )
            for scale, tracking_channels in zip(self.stage_scales, channels[1:]):
                self.source_fusions[str(scale)] = GatedSourceFusion(
                    self.source_rgb_encoder.channels_by_scale[scale], tracking_channels,
                )
        self.xyz = nn.Conv2d(channels[-1], 3, 3, padding=1)

    def encode_source_rgb(self, source_rgb: torch.Tensor) -> dict[int, torch.Tensor]:
        if not self.source_rgb_pyramid_enabled or self.source_rgb_encoder is None:
            raise RuntimeError("source RGB pyramid is disabled")
        return self.source_rgb_encoder(source_rgb)

    def forward(self, feature: torch.Tensor, source_rgb: torch.Tensor | None = None,
                source_pyramid: dict[int, torch.Tensor] | None = None,
                batch: int | None = None, pairs: int | None = None) -> torch.Tensor:
        output_size = self.output_size
        if self.native_512:
            if tuple(feature.shape[-2:]) not in {(32, 32), (64, 64)}:
                raise ValueError('native decoder accepts only32/64 query grids')
            output_size = tuple(int(v) * 8 for v in feature.shape[-2:])
        if source_rgb is not None and source_pyramid is not None:
            raise ValueError("pass source_rgb or source_pyramid, not both")
        if self.source_rgb_pyramid_enabled:
            if batch is None or pairs is None:
                raise ValueError("source RGB pyramid requires batch and pairs")
            if source_pyramid is None:
                if source_rgb is None:
                    raise ValueError("source RGB pyramid requires source_rgb or source_pyramid")
                source_pyramid = self.encode_source_rgb(source_rgb)
        elif source_rgb is not None or source_pyramid is not None:
            raise ValueError("source appearance was provided but the RGB pyramid is disabled")
        if self.native_512 and source_pyramid is not None:
            for stage in (32, 64, 128, 256):
                expected_hw = tuple(v * stage // 256 for v in output_size)
                if tuple(source_pyramid[stage].shape[-2:]) != expected_hw:
                    raise ValueError('native RGB stage and decoder grids are misaligned')
        x = self.projection(feature)
        if source_pyramid is not None and self.source_rgb_fusion_32:
            x = self.source_fusions["32"](
                x, source_pyramid[32], int(batch), int(pairs),
            )
        for scale, block in zip(self.stage_scales, self.blocks):
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
            x = block(x)
            if source_pyramid is not None:
                x = self.source_fusions[str(scale)](
                    x, source_pyramid[scale], int(batch), int(pairs),
                )
        x = self.xyz(x)
        if x.shape[-2:] != output_size:
            raise RuntimeError(f"upsampler output {tuple(x.shape[-2:])} != {output_size}")
        return x
