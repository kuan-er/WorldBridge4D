"""Typed model outputs and structured latent helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .camera import CameraOutput

import torch

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
    include_motion: bool = True

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
    if not z4d.include_motion or slots == 0:
        return dense_memory, dense_coordinates
    motion_memory = z4d.motion.reshape(batch, frames * slots, channels)
    # Motion slots have no pixel location. Keeping them at the grid center
    # avoids assigning a false object position while retaining 2D-RoPE for the
    # dense memory. Physical time and slot identity are embedded in the values.
    center = dense_coordinates.float().mean(dim=0, keepdim=True)
    motion_coordinates = center.expand(frames * slots, -1)
    return torch.cat((dense_memory, motion_memory), dim=1), torch.cat(
        (dense_coordinates.to(dtype=center.dtype), motion_coordinates), dim=0
    )

@dataclass
class DenseQueryOutput:
    normalized_xyz: torch.Tensor
    low_resolution_feature: torch.Tensor
    camera: CameraOutput | None = None
