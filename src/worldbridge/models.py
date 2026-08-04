"""Matched Compact/Full deterministic 4D latent autoencoders."""
from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


def time_features(length: int, dim: int, device, dtype) -> torch.Tensor:
    if dim <= 0:
        return torch.empty(length, 0, device=device, dtype=dtype)
    half = max(1, dim // 2)
    t = torch.linspace(0.0, 1.0, length, device=device, dtype=dtype)[:, None]
    freq = torch.pow(2.0, torch.arange(half, device=device, dtype=dtype))[None, :] * math.pi
    out = torch.cat([torch.sin(t * freq), torch.cos(t * freq)], dim=-1)
    return out[:, :dim]


class Residual2DBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        return F.gelu(x + self.net(x))


class ReconstructionEncoder(nn.Module):
    """Shared-weight 2D residual CNN for pointmap reconstruction features."""
    def __init__(self, out_channels: int = 32, hidden: int = 32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(4, hidden, 5, stride=2, padding=2), nn.GELU())
        self.block1 = Residual2DBlock(hidden)
        self.down = nn.Sequential(nn.Conv2d(hidden, hidden, 3, stride=2, padding=1), nn.GELU())
        self.block2 = Residual2DBlock(hidden)
        self.out = nn.Conv2d(hidden, out_channels, 3, stride=2, padding=1)

    def forward(self, pointmaps: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        x = torch.cat([pointmaps, valid[:, None].to(pointmaps.dtype)], dim=1)
        return self.out(self.block2(self.down(self.block1(self.stem(x)))))


class TrajectoryEncoder(nn.Module):
    """Small temporal MLP/1D-CNN with explicit visibility, validity and gamma(t)."""
    def __init__(self, out_channels: int = 32, hidden: int = 48, time_dim: int = 8):
        super().__init__()
        self.time_dim = time_dim
        self.in_proj = nn.Sequential(nn.Linear(3 + 2 + time_dim, hidden), nn.GELU())
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.out = nn.Linear(hidden, out_channels)

    def forward(self, values: torch.Tensor, visible: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # values [N,T,3], masks [N,T]. Masking is input information, not a loss mask.
        n, t, _ = values.shape
        gamma = time_features(t, self.time_dim, values.device, values.dtype)[None].expand(n, -1, -1)
        x = torch.cat([values, visible[..., None].to(values.dtype), valid[..., None].to(values.dtype), gamma], dim=-1)
        x = self.in_proj(x).transpose(1, 2)
        x = self.temporal(x).transpose(1, 2)
        weights = valid.to(x.dtype)[..., None]
        pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.out(pooled)


class ContextResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(channels, channels, 3, padding=1), nn.GELU(),
                                 nn.Conv3d(channels, channels, 3, padding=1))

    def forward(self, x):
        return F.gelu(x + self.net(x))


class ContextEncoder3D(nn.Module):
    """Same 3D residual context structure for both methods; target size is exact."""
    def __init__(self, in_channels: int, hidden: int = 48, latent_channels: int = 16):
        super().__init__()
        self.input = nn.Conv3d(in_channels, hidden, 3, padding=1)
        self.block1 = ContextResidualBlock(hidden)
        self.block2 = ContextResidualBlock(hidden)
        self.output = nn.Conv3d(hidden, latent_channels, 1)

    def forward(self, x: torch.Tensor, target_shape: tuple[int, int, int]) -> torch.Tensor:
        x = self.block2(self.block1(F.gelu(self.input(x))))
        x = F.adaptive_avg_pool3d(x, target_shape)
        return self.output(x)


class FourierCoordinates(nn.Module):
    def __init__(self, frequencies: int = 4):
        super().__init__()
        self.frequencies = frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [...,D], values in [0,1]. Keep raw coordinates alongside Fourier bands.
        bands = [x]
        for i in range(self.frequencies):
            scale = (2.0 ** i) * math.pi
            bands.extend([torch.sin(scale * x), torch.cos(scale * x)])
        return torch.cat(bands, dim=-1)


class QueryDecoder(nn.Module):
    """Differentiable trilinear latent sampling followed by a shared MLP form."""
    def __init__(self, latent_channels: int = 16, hidden: int = 96, frequencies: int = 4):
        super().__init__()
        self.coord = FourierCoordinates(frequencies)
        # p=(u,v), source time and target time. Each scalar is encoded explicitly.
        coord_dim = (2 + 2 * frequencies * 2) + (1 + 2 * frequencies) * 2
        self.mlp = nn.Sequential(nn.Linear(latent_channels + coord_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))

    @staticmethod
    def sample_latent(z4d: torch.Tensor, source_time: torch.Tensor, uv01: torch.Tensor, source_length: int) -> torch.Tensor:
        # z4d [B,C,Tz,Hz,Wz], grid_sample uses normalized (x=width,y=height,z=time).
        b, _, tz, hz, wz = z4d.shape
        s = source_time.to(z4d.dtype)
        grid_t = 2.0 * s / max(source_length - 1, 1) - 1.0
        grid = torch.stack([2.0 * uv01[..., 0] - 1.0, 2.0 * uv01[..., 1] - 1.0, grid_t], dim=-1)
        sampled = F.grid_sample(z4d, grid[:, :, None, None, :], mode="bilinear", padding_mode="border", align_corners=True)
        return sampled[:, :, :, 0, 0].transpose(1, 2)

    def forward(self, z4d: torch.Tensor, source_time: torch.Tensor, uv01: torch.Tensor,
                target_time: torch.Tensor, source_length: int) -> torch.Tensor:
        latent = self.sample_latent(z4d, source_time, uv01, source_length)
        p = self.coord(uv01)
        s = self.coord((source_time / max(source_length - 1, 1))[..., None])
        t = self.coord((target_time / max(source_length - 1, 1))[..., None])
        return self.mlp(torch.cat([latent, p, s, t], dim=-1))


class WorldLatentModel(nn.Module):
    """Compact or Full model with exact ``[B,Cz,Tz,Hz,Wz]`` output."""
    def __init__(self, mode: Literal["compact", "full"], *, latent_channels: int = 16,
                 latent_time: int = 6, latent_height: int = 16, latent_width: int = 16,
                 trajectory_channels: int = 32, trajectory_hidden: int = 48,
                 trajectory_time_dim: int = 8, reconstruction_channels: int = 32,
                 context_hidden: int = 48, decoder_hidden: int = 96):
        super().__init__()
        if mode not in ("compact", "full"):
            raise ValueError(mode)
        self.mode = mode
        self.latent_shape = (latent_time, latent_height, latent_width)
        self.traj_encoder = TrajectoryEncoder(trajectory_channels, trajectory_hidden, trajectory_time_dim)
        self.context = ContextEncoder3D(trajectory_channels, context_hidden, latent_channels)
        self.decoder = QueryDecoder(latent_channels, decoder_hidden)
        self.coord_mean = nn.Parameter(torch.zeros(3), requires_grad=False)
        self.coord_scale = nn.Parameter(torch.ones(3), requires_grad=False)
        if mode == "compact":
            self.reconstruction = ReconstructionEncoder(reconstruction_channels, reconstruction_channels)
            self.fuser = nn.Sequential(nn.Linear(reconstruction_channels + trajectory_channels + 1, trajectory_channels), nn.GELU(),
                                       nn.Linear(trajectory_channels, trajectory_channels))

    def set_coordinate_stats(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        self.coord_mean.copy_(mean.detach().to(self.coord_mean))
        self.coord_scale.copy_(scale.detach().to(self.coord_scale).clamp_min(1e-4))

    def normalize_coordinates(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.coord_mean) / self.coord_scale

    def denormalize_coordinates(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.coord_scale + self.coord_mean

    def encode_trajectory(self, values: torch.Tensor, visible: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return self.traj_encoder(self.normalize_coordinates(values), visible, valid)

    def encode_compact(self, pointmaps: torch.Tensor, point_valid: torch.Tensor,
                       anchor_values: torch.Tensor, anchor_visible: torch.Tensor,
                       anchor_valid: torch.Tensor) -> torch.Tensor:
        b, t, _, h, w = pointmaps.shape
        rec = self.reconstruction(pointmaps.reshape(b*t, 3, h, w), point_valid.reshape(b*t, h, w))
        _, cr, hh, ww = rec.shape
        rec = rec.reshape(b, t, cr, hh, ww).permute(0, 1, 3, 4, 2)
        q = self.encode_trajectory(anchor_values.permute(0, 2, 3, 1, 4).reshape(b*h*w, t, 3),
                                   anchor_visible.permute(0, 2, 3, 1).reshape(b*h*w, t),
                                   anchor_valid.permute(0, 2, 3, 1).reshape(b*h*w, t))
        q = q.reshape(b, h, w, -1).permute(0, 3, 1, 2)
        q = F.adaptive_avg_pool2d(q, (hh, ww)).permute(0, 2, 3, 1)
        qt = q[:, None].expand(-1, t, -1, -1, -1).clone()
        qt[:, 1:] = 0.0
        anchor = torch.zeros(b, t, hh, ww, 1, device=pointmaps.device, dtype=pointmaps.dtype)
        anchor[:, 0] = 1.0
        fused = self.fuser(torch.cat([rec, qt, anchor], dim=-1))
        return self.context(fused.permute(0, 4, 1, 2, 3), self.latent_shape)

    def encode_full(self, trajectory_features: torch.Tensor) -> torch.Tensor:
        # trajectory_features [B,source,H,W,Ct], source is the latent time axis.
        return self.context(trajectory_features.permute(0, 4, 1, 2, 3), self.latent_shape)

    def decode_queries(self, z4d: torch.Tensor, source_time: torch.Tensor, uv01: torch.Tensor,
                       target_time: torch.Tensor) -> torch.Tensor:
        return self.decoder(z4d, source_time, uv01, target_time, z4d.shape[2] if False else self._source_length)

    def forward(self, *, pointmaps=None, point_valid=None, anchor_values=None, anchor_visible=None,
                anchor_valid=None, trajectory_features=None, source_time=None, uv01=None, target_time=None,
                source_length: int | None = None):
        if source_length is None:
            raise ValueError("source_length is required for coordinate sampling")
        self._source_length = int(source_length)
        if self.mode == "compact":
            z = self.encode_compact(pointmaps, point_valid, anchor_values, anchor_visible, anchor_valid)
        else:
            z = self.encode_full(trajectory_features)
        if source_time is None:
            return z
        pred = self.decoder(z, source_time, uv01, target_time, source_length)
        return z, self.denormalize_coordinates(pred)
