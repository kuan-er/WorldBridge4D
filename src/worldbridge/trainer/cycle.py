"""Differentiable forward/backward pixel-cycle reprojection losses."""
from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F


def source_xyz_to_target_uv(
    source_xyz: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    camera_positions: torch.Tensor,
    camera_rotations: torch.Tensor,
    focal_length: torch.Tensor,
    sensor_width: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform source-camera XYZ to target-camera UV.

    ``source_xyz`` is ``[B,3,H,W]`` in the source camera frame. Camera
    rotations use the same convention as :class:`CameraModel`: the rows are
    the camera-to-world rotation basis. The returned UV is ``[B,H,W,2]`` in
    pixel coordinates (u, v), and depth is positive in front of the camera.
    """
    if source_xyz.ndim != 4 or source_xyz.shape[1] != 3:
        raise ValueError("source_xyz must be [B,3,H,W]")
    batch = source_xyz.shape[0]
    if source.shape != (batch,) or target.shape != (batch,):
        raise ValueError("source and target must both be [B]")
    if camera_positions.shape[:2] != (batch, 21) or camera_positions.shape[-1] != 3:
        raise ValueError("camera_positions must be [B,21,3]")
    if camera_rotations.shape[:2] != (batch, 21) or camera_rotations.shape[-2:] != (3, 3):
        raise ValueError("camera_rotations must be [B,21,3,3]")

    index = torch.arange(batch, device=source_xyz.device)
    points = source_xyz.permute(0, 2, 3, 1)
    source_rotation = camera_rotations[index, source]
    source_position = camera_positions[index, source]
    world = torch.einsum("bhwj,bij->bhwi", points, source_rotation) + source_position[:, None, None]
    target_rotation = camera_rotations[index, target]
    target_position = camera_positions[index, target]
    target_camera = torch.einsum(
        "bhwj,bji->bhwi", world - target_position[:, None, None], target_rotation,
    )
    depth = -target_camera[..., 2]
    safe_depth = depth.clamp_min(1e-6)
    width = source_xyz.shape[-1]
    height = source_xyz.shape[-2]
    # Legacy Kubric metadata supplies one normalized focal length; external
    # datasets may provide independent x/y focal lengths after crop/resize.
    if focal_length.ndim == 1:
        fx = focal_length / sensor_width * float(width)
        fy = fx
    elif focal_length.ndim == 3 and focal_length.shape[-1] == 2:
        if focal_length.shape[:2] != (batch, 21):
            raise ValueError("per-frame focal_length must be [B,21,2]")
        fx = focal_length[..., 0] / sensor_width * float(width)
        fy = focal_length[..., 1] / sensor_width * float(width)
        fx = fx[index, target]
        fy = fy[index, target]
    else:
        raise ValueError(
            "focal_length must be [B] or [B,21,2], "
            f"got {tuple(focal_length.shape)}"
        )
    cx = (float(width) - 1.0) / 2.0
    cy = (float(height) - 1.0) / 2.0
    uv = torch.stack((
        fx[:, None, None] * target_camera[..., 0] / safe_depth + cx,
        cy - fy[:, None, None] * target_camera[..., 1] / safe_depth,
    ), dim=-1)
    return uv, depth


def bilinear_sample_xyz(xyz_map: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Sample ``[B,3,H,W]`` XYZ maps at pixel-space ``[B,H,W,2]`` UV."""
    if xyz_map.ndim != 4 or xyz_map.shape[1] != 3:
        raise ValueError("xyz_map must be [B,3,H,W]")
    if uv.shape != (xyz_map.shape[0], xyz_map.shape[2], xyz_map.shape[3], 2):
        raise ValueError("uv must match xyz_map spatial dimensions")
    height, width = xyz_map.shape[-2:]
    grid = uv.to(dtype=xyz_map.dtype).clone()
    grid_x = grid[..., 0] / max(width - 1, 1) * 2.0 - 1.0
    grid_y = grid[..., 1] / max(height - 1, 1) * 2.0 - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)
    return F.grid_sample(
        xyz_map, grid, mode="bilinear", padding_mode="zeros", align_corners=True,
    )


def pixel_cycle_loss(
    forward_xyz: torch.Tensor,
    reverse_xyz: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    source_valid: torch.Tensor,
    target_valid: torch.Tensor,
    target_visible: torch.Tensor,
    camera_positions: torch.Tensor,
    camera_rotations: torch.Tensor,
    focal_length: torch.Tensor,
    sensor_width: torch.Tensor,
    *,
    huber_delta: float = 0.01,
    image_size: int = 256,
    pixel_stride: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute one bidirectional-compatible A->B->A pixel cycle.

    The forward map is in A coordinates and the reverse map is in B
    coordinates. Only points that are valid/visible in both endpoint frames
    are supervised; out-of-view predictions are masked. The returned values
    are (loss, valid-point count, mean pixel error), with the latter two
    detached for logging.
    """
    if forward_xyz.shape != reverse_xyz.shape or forward_xyz.ndim != 4:
        raise ValueError("forward_xyz and reverse_xyz must match [B,3,H,W]")
    batch, _, height, width = forward_xyz.shape
    expected = (batch, height, width)
    for name, value in (
        ("source_valid", source_valid), ("target_valid", target_valid),
        ("target_visible", target_visible),
    ):
        if value.shape != expected:
            raise ValueError(f"{name} must be [B,H,W]")
    if image_size != width or image_size != height:
        raise ValueError("cycle currently requires square maps at image_size")
    stride = int(pixel_stride)
    if stride < 1:
        raise ValueError("pixel_stride must be positive")

    forward_uv, forward_depth = source_xyz_to_target_uv(
        forward_xyz, source, target, camera_positions, camera_rotations,
        focal_length, sensor_width,
    )
    sampled_reverse_xyz = bilinear_sample_xyz(reverse_xyz, forward_uv)
    cycle_uv, cycle_depth = source_xyz_to_target_uv(
        sampled_reverse_xyz, target, source, camera_positions, camera_rotations,
        focal_length, sensor_width,
    )
    values = torch.arange(0, image_size, stride, device=forward_xyz.device)
    vv, uu = torch.meshgrid(values, values, indexing="ij")
    source_uv = torch.stack((uu, vv), dim=-1).to(dtype=cycle_uv.dtype)
    source_uv = source_uv.unsqueeze(0).expand(batch, -1, -1, -1)
    if stride != 1:
        source_valid = source_valid[:, ::stride, ::stride]
        target_valid = target_valid[:, ::stride, ::stride]
        target_visible = target_visible[:, ::stride, ::stride]
        forward_uv = forward_uv[:, ::stride, ::stride]
        cycle_uv = cycle_uv[:, ::stride, ::stride]
        forward_depth = forward_depth[:, ::stride, ::stride]
        cycle_depth = cycle_depth[:, ::stride, ::stride]
    inside_forward = (
        (forward_uv[..., 0] >= 0.0) & (forward_uv[..., 0] <= image_size - 1)
        & (forward_uv[..., 1] >= 0.0) & (forward_uv[..., 1] <= image_size - 1)
    )
    inside_cycle = (
        (cycle_uv[..., 0] >= 0.0) & (cycle_uv[..., 0] <= image_size - 1)
        & (cycle_uv[..., 1] >= 0.0) & (cycle_uv[..., 1] <= image_size - 1)
    )
    sampled_forward_xyz = forward_xyz[..., ::stride, ::stride]
    sampled_reverse_xyz = sampled_reverse_xyz[..., ::stride, ::stride]
    finite = (
        torch.isfinite(sampled_forward_xyz).all(dim=1)
        & torch.isfinite(sampled_reverse_xyz).all(dim=1)
        & torch.isfinite(forward_uv).all(dim=-1)
        & torch.isfinite(cycle_uv).all(dim=-1)
    )
    mask = (
        source_valid & target_valid & target_visible
        & (forward_depth > 0.0) & (cycle_depth > 0.0)
        & inside_forward & inside_cycle & finite
    )
    if not bool(mask.any()):
        zero = forward_xyz.sum() * 0.0
        return zero, mask.sum().detach(), zero.detach()
    normalized_error = (cycle_uv - source_uv) / float(image_size)
    loss_map = F.huber_loss(
        normalized_error, torch.zeros_like(normalized_error),
        delta=float(huber_delta), reduction="none",
    ).sum(dim=-1)
    loss = loss_map[mask].mean()
    pixel_error = torch.linalg.vector_norm(cycle_uv - source_uv, dim=-1)[mask].mean()
    return loss, mask.sum().detach(), pixel_error.detach()


def camera_batch(cameras: list[Mapping[str, object]], device: torch.device,
                 dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    """Convert cached Kubric camera metadata to device tensors."""
    if not cameras or any(camera is None for camera in cameras):
        raise ValueError("all cycle samples must provide camera metadata")
    positions = torch.from_numpy(np.stack([
        camera["positions"] for camera in cameras
    ])).to(device=device, dtype=dtype)
    rotations = torch.from_numpy(np.stack([
        camera["rotations"] for camera in cameras
    ])).to(device=device, dtype=dtype)
    focal = torch.as_tensor(
        np.stack([np.asarray(camera["focal_length"], dtype=np.float32) for camera in cameras]),
        device=device, dtype=dtype,
    )
    sensor = torch.as_tensor(
        np.stack([np.asarray(camera["sensor_width"], dtype=np.float32) for camera in cameras]),
        device=device, dtype=dtype,
    )
    return positions, rotations, focal, sensor
