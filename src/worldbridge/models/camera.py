"""Decoder-native camera readout: pose token plus per-frame ray field.

H033 replaced the previous standalone attention head. Two deliberate choices:

* Pose comes from one camera token appended to the decoder's own pair-query
  sequence, so it attends to exactly the same full-resolution structured memory
  the XYZ head uses, and its readout is a plain MLP + linear heads (the 4RC
  ``camera token -> MLP`` shape) instead of a separate memory/attention stack.
* Intrinsics come from a per-frame unit ray field read off the diagonal pair's
  pixel tokens, supervised by GT rays. K is decoded from the ray field, so the
  representation is strictly more general than two FoV numbers.

Protocol: -Z forward, +Y up. Relative pose maps target camera coordinates into
source camera coordinates (T_source<-target). No external code or weights.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Safe XYZW quaternion conversion, computed in FP32."""
    q = F.normalize(q.float(), dim=-1, eps=1e-8)
    x, y, z, w = q.unbind(-1)
    return torch.stack((
        1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y),
    ), -1).reshape(*q.shape[:-1], 3, 3)


@dataclass
class CameraOutput:
    """Pose per ``(source,target)`` pair plus a per-frame ray field.

    ``rays`` and ``ray_frames`` describe the diagonal pairs actually present in
    the batch; each row is one frame's own camera-frame ray field.
    """

    rotation: torch.Tensor                    # [B,P,3,3] T_source<-target
    translation: torch.Tensor                 # [B,P,3] physical pointmap units (metres)
    rays: torch.Tensor | None = None          # [B,3,G,G] unit directions, own camera frame
    ray_frames: torch.Tensor | None = None    # [B] frame index behind each ray map

    def decode_pinhole(self, height: int, width: int) -> dict[str, torch.Tensor]:
        """Least-squares skew-free pinhole K per ray map (no centring assumed).

        GT rays follow ``x=(u-cx)/fx, y=-(v-cy)/fy, z=-1``. Dividing the
        predicted unit ray by ``-d_z`` recovers that unnormalised form, so
        ``p_x = u/fx - cx/fx`` and ``p_y = -v/fy + cy/fy`` are two independent
        linear least-squares problems in ``(u, v)``.
        """
        if self.rays is None:
            raise RuntimeError("camera output carries no ray field")
        rays = self.rays.float()
        batch, channels, grid_h, grid_w = rays.shape
        if channels != 3 or grid_h < 2 or grid_w < 2:
            raise ValueError(f"ray field must be [B,3,G,G] with G>=2, got {tuple(rays.shape)}")
        forward = rays[:, 2:3]
        # Rays must look forward (-Z). Invalid rows are masked out of the fit.
        valid = forward < -1e-6
        denominator = (-forward).clamp_min(1e-6)
        p_x = rays[:, 0:1] / denominator
        p_y = rays[:, 1:2] / denominator
        weight = forward.abs() * valid
        scale_y, scale_x = height / grid_h, width / grid_w
        ys = (torch.arange(grid_h, device=rays.device, dtype=rays.dtype) + 0.5) * scale_y - 0.5
        xs = (torch.arange(grid_w, device=rays.device, dtype=rays.dtype) + 0.5) * scale_x - 0.5
        u = xs.reshape(1, 1, 1, grid_w).expand(batch, 1, grid_h, grid_w)
        v = ys.reshape(1, 1, grid_h, 1).expand(batch, 1, grid_h, grid_w)

        def weighted_fit(coordinate: torch.Tensor, target: torch.Tensor):
            count = weight.sum((-2, -1)).clamp_min(1e-6)
            mean_c = (weight * coordinate).sum((-2, -1)) / count
            mean_t = (weight * target).sum((-2, -1)) / count
            centred_c = coordinate - mean_c[..., None, None]
            covariance = (weight * centred_c * (target - mean_t[..., None, None])).sum((-2, -1))
            variance = (weight * centred_c.square()).sum((-2, -1))
            slope = covariance / variance.clamp_min(1e-12)
            return slope.squeeze(-1), (mean_t - slope * mean_c).squeeze(-1), count.squeeze(-1)

        slope_x, intercept_x, count = weighted_fit(u, p_x)
        slope_y, intercept_y, _ = weighted_fit(v, p_y)
        # x: p_x = u/fx - cx/fx  ->  slope=1/fx, intercept=-cx/fx
        fx = 1.0 / slope_x.clamp_min(1e-12)
        cx = -intercept_x * fx
        # y: p_y = -v/fy + cy/fy ->  slope=-1/fy, intercept=cy/fy
        fy = -1.0 / slope_y.clamp_max(-1e-12)
        cy = -intercept_y / slope_y.clamp_max(-1e-12)
        # Reject degenerate fits (a nearly flat ray field has no identifiable
        # focal length); they must not pollute the decoded-intrinsics metrics.
        valid_map = ((valid.reshape(batch, -1).sum(-1) > 1)
                     & (fx > 1.0) & (fx < 1e6) & (fy > 1.0) & (fy < 1e6))
        scale = torch.as_tensor([width, height], device=rays.device, dtype=rays.dtype)
        return {
            "focal_x": fx, "focal_y": fy, "principal_x": cx, "principal_y": cy,
            "count": count, "valid": valid_map,
            "fov": 2 * torch.atan(scale / (2 * torch.stack((fx, fy), -1).clamp_min(1e-6))),
        }


class CameraQueryHead(nn.Module):
    """One sequence token -> small MLP -> translation + quaternion (4RC shape)."""

    def __init__(self, query_dim: int, hidden: int = 256):
        super().__init__()
        self.query_dim = int(query_dim)
        self.hidden = int(hidden)
        self.token = nn.Parameter(torch.zeros(1, 1, 1, self.query_dim))
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.query_dim),
            nn.Linear(self.query_dim, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, self.hidden), nn.ReLU(),
        )
        self.translation = nn.Linear(self.hidden, 3)
        self.quaternion = nn.Linear(self.hidden, 4)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.shape[-1] != self.query_dim:
            raise ValueError(f"camera token dim {features.shape[-1]} != {self.query_dim}")
        hidden = self.mlp(features)
        translation = self.translation(hidden).float()
        quaternion = self.quaternion(hidden).float() + features.new_tensor([0, 0, 0, 1])
        return translation, quaternion_to_matrix(quaternion)


class RayFieldHead(nn.Module):
    """Pixel token + pixel position -> unit ray direction in the frame's camera."""

    def __init__(self, query_dim: int, hidden: int = 256):
        super().__init__()
        self.query_dim = int(query_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.query_dim + 2, int(hidden)), nn.SiLU(),
            nn.Linear(int(hidden), int(hidden)), nn.SiLU(),
            nn.Linear(int(hidden), 3),
        )

    def forward(self, features: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        """features [B,Q,D] (or [Q,D]), coordinates [Q,2] in [-1,1] -> [B,3,G,G]."""
        if features.ndim == 2:
            features = features[None]
        if coordinates.shape != (features.shape[-2], 2):
            raise ValueError(f"ray coordinates must be [Q,2] for Q={features.shape[-2]}")
        grid = int(round(features.shape[-2] ** 0.5))
        if grid * grid != features.shape[-2]:
            raise ValueError("ray field requires a square query grid")
        position = coordinates.to(device=features.device, dtype=features.dtype)[None]
        position = position.expand(features.shape[0], -1, -1)
        raw = self.mlp(torch.cat((features, position), dim=-1))
        # [B,Q,3] -> [B,3,G,G]: transpose, never a bare reshape, otherwise the
        # token and channel axes are silently interleaved.
        return F.normalize(raw, dim=-1).transpose(1, 2).reshape(features.shape[0], 3, grid, grid)


def normalised_grid_coordinates(grid: int, device, dtype) -> torch.Tensor:
    """Pixel-centre coordinates of a square query grid, mapped to [-1,1]."""
    axis = (torch.arange(grid, device=device, dtype=dtype) + 0.5) * (2.0 / grid) - 1.0
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)
