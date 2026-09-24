"""Lightweight source-conditioned cameras; protocol is -Z forward, +Y up.

Inspired by 4RC's camera-token -> MLP -> translation/quaternion/FOV design.
No external implementation or weights are imported. Relative pose maps target
camera coordinates into source camera coordinates (T_source<-target).
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from .outputs import StructuredZ4D


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
    rotation: torch.Tensor     # [B,T,3,3], T_source<-target
    translation: torch.Tensor  # [B,T,3], physical pointmap units
    fov: torch.Tensor          # [B,T,2], horizontal/vertical, source-independent

    def intrinsics(self, height: int, width: int) -> torch.Tensor:
        # Use the existing integer-pixel-center protocol, not 4RC's half-pixel
        # shifted W/2 convention. Actual GT principal points remain untouched.
        K = self.fov.new_zeros((*self.fov.shape[:-1], 3, 3))
        K[..., 0, 0] = (width / 2) / torch.tan(self.fov[..., 0] / 2)
        K[..., 1, 1] = (height / 2) / torch.tan(self.fov[..., 1] / 2)
        K[..., 0, 2], K[..., 1, 2] = (width-1)/2, (height-1)/2
        K[..., 2, 2] = 1
        return K


class SourceConditionedCameraHead(nn.Module):
    def __init__(self, input_dim: int, dim: int = 256, num_heads: int = 8,
                 num_frames: int = 21, memory_grid: int = 16):
        super().__init__()
        self.num_frames, self.memory_grid = num_frames, memory_grid
        self.memory_projection = nn.Linear(input_dim, dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.spatial_projection = nn.Linear(2, dim, bias=False)
        self.memory_time = nn.Embedding(num_frames, dim)
        self.source_time = nn.Embedding(num_frames, dim)
        self.target_time = nn.Embedding(num_frames, dim)
        self.relative_time = nn.Embedding(2*num_frames-1, dim)
        self.camera_query = nn.Parameter(torch.zeros(1, 1, dim))
        self.intrinsic_query = nn.Parameter(torch.zeros(1, 1, dim))
        self.query_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0)
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_mlp = nn.Sequential(nn.Linear(dim, 2*dim), nn.GELU(), nn.Linear(2*dim, dim))
        self.temporal = nn.TransformerEncoderLayer(dim, num_heads, dim_feedforward=2*dim,
                                                  dropout=0, batch_first=True, norm_first=True)
        self.pose_mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU(),
                                      nn.Linear(dim, dim), nn.ReLU())
        self.translation = nn.Linear(dim, 3)
        self.quaternion = nn.Linear(dim, 4)
        self.fov_mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 2))
        for head in (self.translation, self.quaternion, self.fov_mlp[-1]):
            nn.init.normal_(head.weight, std=0.001)
            nn.init.zeros_(head.bias)
        # Start near unit rotation and ~1 rad FOV, not a degenerate quaternion.
        nn.init.constant_(self.fov_mlp[-1].bias, math.log((1.0-0.01)/(math.pi-0.01-1.0)))

    def forward(self, z4d: StructuredZ4D, source: torch.Tensor) -> CameraOutput:
        dense = z4d.dense
        b, c, t, h, w = dense.shape
        if t != self.num_frames or source.shape != (b,):
            raise ValueError('camera source must be [B] and dense time must equal21')
        # Keep all21 physical frames; only pool spatially for bounded memory.
        ph, pw = min(h, self.memory_grid), min(w, self.memory_grid)
        memory = F.adaptive_avg_pool2d(dense.permute(0,2,1,3,4).reshape(b*t,c,h,w), (ph,pw))
        memory = self.memory_projection(memory.flatten(2).transpose(1,2)).reshape(b,t,ph*pw,-1)
        yy, xx = torch.meshgrid(torch.linspace(-1,1,ph,device=dense.device,dtype=dense.dtype),
                                torch.linspace(-1,1,pw,device=dense.device,dtype=dense.dtype), indexing='ij')
        spatial = self.spatial_projection(torch.stack((xx,yy),-1).reshape(ph*pw,2))
        times = torch.arange(t, device=dense.device)
        memory = self.memory_norm(memory + spatial[None,None] + self.memory_time(times)[None,:,None])
        memory = memory.flatten(1,2)
        q = (self.camera_query + self.source_time(source)[:,None]
             + self.target_time(times)[None] + self.relative_time(times[None]-source[:,None]+t-1))
        # Cross attention does not mix queries. Intrinsic queries contain t,
        # never s: changing reference camera cannot change any predicted K_t.
        iq = (self.intrinsic_query + self.target_time(times)[None]).expand(b,-1,-1)
        q = torch.cat((q, iq), dim=1)
        nq = self.query_norm(q)
        q = q + self.cross_attention(nq, memory, memory, need_weights=False)[0]
        q = q + self.cross_mlp(self.cross_norm(q))
        feat = self.pose_mlp(self.temporal(q[:,:t]))
        translation = self.translation(feat).float()
        raw_q = self.quaternion(feat).float() + feat.new_tensor([0,0,0,1], dtype=torch.float32)
        rotation = quaternion_to_matrix(raw_q)
        diagonal = times[None] == source[:,None]
        rotation = torch.where(diagonal[...,None,None], torch.eye(3,device=dense.device), rotation)
        translation = torch.where(diagonal[...,None], torch.zeros_like(translation), translation)
        fov = 0.01 + (math.pi-0.02)*torch.sigmoid(self.fov_mlp(q[:,t:]).float())
        return CameraOutput(rotation, translation, fov)
