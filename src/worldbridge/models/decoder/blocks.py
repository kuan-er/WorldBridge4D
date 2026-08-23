"""Reusable attention and residual decoder blocks."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

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
