"""Source-RGB pyramid and gated appearance fusion."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .blocks import ResidualBlock2D, _group_count

class SourceRGBPyramid(nn.Module):
    """Lightweight source-only appearance pyramid for 256px dense tracking."""

    def __init__(self, channels: Sequence[int] = (32, 64, 128), native_512: bool = False):
        super().__init__()
        channels = tuple(int(value) for value in channels)
        if len(channels) != 3 or any(value < 1 for value in channels):
            raise ValueError("source RGB pyramid channels must contain three positive values")
        self.native_512 = bool(native_512)
        high, middle, low = channels
        self.channels_by_scale = {256: high, 128: middle, 64: low, 32: low}
        self.stem = nn.Sequential(
            nn.Conv2d(3, high, 3, padding=1),
            nn.GroupNorm(_group_count(high), high),
            nn.SiLU(),
            ResidualBlock2D(high, high),
        )
        self.down_128 = nn.Sequential(
            nn.Conv2d(high, middle, 3, stride=2, padding=1),
            ResidualBlock2D(middle, middle),
        )
        self.down_64 = nn.Sequential(
            nn.Conv2d(middle, low, 3, stride=2, padding=1),
            ResidualBlock2D(low, low),
        )

    def forward(self, source_rgb: torch.Tensor) -> dict[int, torch.Tensor]:
        allowed = {(3, 256, 256), (3, 512, 512)} if self.native_512 else {(3, 256, 256)}
        if source_rgb.ndim != 4 or tuple(source_rgb.shape[1:]) not in allowed:
            raise ValueError(
                f"source RGB must be [B,3,256,256], got {tuple(source_rgb.shape)}"
            )
        feature_256 = self.stem(source_rgb)
        feature_128 = self.down_128(feature_256)
        feature_64 = self.down_64(feature_128)
        # Reuse the mature encoder and derive the new coarse identity scale
        # without introducing another high-resolution activation path.
        feature_32 = F.avg_pool2d(feature_64, kernel_size=2, stride=2)
        # Keys are checkpoint-stable STAGE labels. At512 these tensors have
        # spatial sizes64/128/256/512, not downsampled256 appearance.
        return {32: feature_32, 64: feature_64, 128: feature_128, 256: feature_256}


class GatedSourceFusion(nn.Module):
    """Inject one K-shared source appearance scale into pair-specific tracking."""

    def __init__(self, source_channels: int, tracking_channels: int):
        super().__init__()
        source_channels = int(source_channels)
        tracking_channels = int(tracking_channels)
        self.source_projection = nn.Sequential(
            nn.GroupNorm(_group_count(source_channels), source_channels),
            nn.Conv2d(source_channels, tracking_channels, 1),
        )
        # A scalar spatial gate is pair/target dependent while avoiding another
        # full C-channel activation at 256px for large K.
        self.gate = nn.Conv2d(tracking_channels, 1, 1)
        # Exact zero makes a migrated baseline checkpoint functionally
        # unchanged before the source branch starts learning.
        # FSDP rejects scalar parameters; keep one element in a 1D tensor.
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, tracking: torch.Tensor, source: torch.Tensor,
                batch: int, pairs: int) -> torch.Tensor:
        batch, pairs = int(batch), int(pairs)
        if tracking.ndim != 4 or tracking.shape[0] != batch * pairs:
            raise ValueError("tracking feature does not match batch*pair dimensions")
        if source.ndim != 4 or source.shape[0] != batch \
                or source.shape[-2:] != tracking.shape[-2:]:
            raise ValueError("source feature does not match tracking batch/spatial dimensions")
        source = self.source_projection(source)
        if source.shape[1] != tracking.shape[1]:
            raise RuntimeError("projected source and tracking channels differ")
        source = source[:, None].expand(-1, pairs, -1, -1, -1).reshape_as(tracking)
        gate = torch.sigmoid(self.gate(tracking))
        return tracking + self.alpha.to(dtype=tracking.dtype) * gate * source
