"""H033 camera supervision: pair pose token plus per-frame ray field.

Pose is read from the decoder's camera token for the sampled ``(source,target)``
pairs, so the loss is defined on exactly those pairs. Intrinsics are read as a
per-frame unit ray field off the diagonal pairs, supervised by GT rays and
decoded to a skew-free pinhole K for metrics.

Carried over unchanged from the H032 audit:

* GT ray construction always uses real principal points; predicted intrinsics
  never build a target.
* Invalid GT poses (non-rigid PO annotation bases) mask only their own pose
  labels; clip, XYZ, GT rays and ray-field supervision continue.
* The diagonal pointmap is additionally pulled towards the GT source rays.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from ..models.camera import CameraOutput


def validate_supervision_camera(camera, height: int, width: int, *,
                                require_shared_intrinsics: bool = False) -> dict[str, float]:
    K = np.asarray(camera['intrinsics'], dtype=np.float64)
    R = np.asarray(camera['rotations'], dtype=np.float64)
    p = np.asarray(camera['positions'], dtype=np.float64)
    if K.shape != (21,3,3) or R.shape != (21,3,3) or p.shape != (21,3):
        raise ValueError('camera metadata requires21 K/R/position entries')
    if not all(np.isfinite(x).all() for x in (K,R,p)):
        raise ValueError('nonfinite GT camera')
    orth_errors = np.max(np.abs(R.transpose(0,2,1) @ R - np.eye(3)), axis=(1,2))
    det_errors = np.abs(np.linalg.det(R)-1)
    orth_error, det_error = float(orth_errors.max()), float(det_errors.max())
    # Preserve the .003 near-SO3 threshold. Some PO zoom clips have genuinely
    # nonrigid annotation bases (~5% shear), not rounding noise. Mark only
    # their pose labels invalid; keep clip, XYZ, true K and ray supervision.
    invalid_pose_frames = int(np.count_nonzero((orth_errors > 0.003) | (det_errors > 0.003)))
    if (K[:,0,0] <= 0).any() or (K[:,1,1] <= 0).any():
        raise ValueError('GT focal must be positive')
    if not np.allclose(K[:,2], [0,0,1]) or not np.allclose(K[:,0,1],0) or not np.allclose(K[:,1,0],0):
        raise ValueError('camera supervision requires zero-skew pinhole K')
    focal_drift = float(np.max(np.abs(K[:,:2,:2]-K[0,:2,:2]) / np.maximum(np.abs(K[0,:2,:2]),1)))
    principal_drift = float(np.max(np.abs(K[:,:2,2]-K[0,:2,2])))
    principal_offset = float(np.max(np.abs(K[:,:2,2]-[(width-1)/2,(height-1)/2])))
    if require_shared_intrinsics and (focal_drift > 1e-5 or principal_drift > 1e-4):
        raise ValueError(f'clip-shared intrinsics violated: {focal_drift=}, {principal_drift=}')
    # Allow <=1px convention differences, never replace actual GT principal points.
    if principal_offset > 1.0:
        raise ValueError(f'centered-camera assumption violated: {principal_offset=}')
    return dict(focal_relative_drift=focal_drift, principal_drift_px=principal_drift,
                principal_center_offset_px=principal_offset, rotation_orthogonality_error=orth_error,
                rotation_determinant_error=det_error, invalid_pose_frames=invalid_pose_frames)


def supervision_camera_batch(cameras, device):
    return tuple(torch.as_tensor(np.stack([c[key] for c in cameras]), device=device, dtype=torch.float32)
                 for key in ('intrinsics','rotations','positions'))


def relative_pose_gt(rotations: torch.Tensor, positions: torch.Tensor, source: torch.Tensor):
    batch = torch.arange(len(source), device=source.device)
    Rs, ps = rotations[batch,source], positions[batch,source]
    # Invert the actual annotation basis, not its transpose (PO is rounded).
    R = torch.linalg.solve(Rs[:,None], rotations)
    p = torch.linalg.solve(Rs[:,None], (positions-ps[:,None])[...,None])[...,0]
    u, _, vh = torch.linalg.svd(R)
    sign = torch.linalg.det(u @ vh)
    correction = torch.ones_like(sign)[...,None].expand(*sign.shape,3).clone()
    correction[...,2] = sign
    R = (u * correction[...,None,:]) @ vh
    return R, p


def pixel_grid_coordinates(grid: int, height: int, width: int, device, dtype):
    """Pixel centres represented by a square query grid of decoder tokens."""
    scale_y, scale_x = height / grid, width / grid
    ys = (torch.arange(grid, device=device, dtype=dtype) + 0.5) * scale_y - 0.5
    xs = (torch.arange(grid, device=device, dtype=dtype) + 0.5) * scale_x - 0.5
    return ys, xs


def unit_rays_at(K: torch.Tensor, ys: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    """GT unit rays at explicit pixel centres; K is [B,3,3], ys/xs are [G]."""
    if ys.numel() != xs.numel():
        raise ValueError('unit_rays_at requires a square square grid of pixel centres')
    grid = int(ys.numel())
    batch = int(K.shape[0])
    dx = ((xs.reshape(1, 1, grid) - K[:, 0, 2].reshape(batch, 1, 1))
          / K[:, 0, 0].reshape(batch, 1, 1)).expand(batch, grid, grid)
    dy = (-(ys.reshape(1, grid, 1) - K[:, 1, 2].reshape(batch, 1, 1))
          / K[:, 1, 1].reshape(batch, 1, 1)).expand(batch, grid, grid)
    dz = -torch.ones_like(dx)
    return F.normalize(torch.stack((dx, dy, dz), dim=1), dim=1)


def unit_source_rays(K: torch.Tensor, height: int, width: int):
    yy, xx = torch.meshgrid(torch.arange(height, device=K.device, dtype=K.dtype),
                            torch.arange(width, device=K.device, dtype=K.dtype), indexing='ij')
    x = (xx[None]-K[:,0,2,None,None]) / K[:,0,0,None,None]
    y = -(yy[None]-K[:,1,2,None,None]) / K[:,1,1,None,None]
    return F.normalize(torch.stack((x,y,-torch.ones_like(x)),dim=1), dim=1)


def diagonal_ray_loss(prediction, valid, source, target, K, mean, scale, beta=0.05):
    """Only diagonal, physical coordinates, fixed scalar scale; mask invalids first."""
    diagonal = source == target
    if not torch.all(diagonal.sum(1) >= 1):
        raise ValueError('every source must have a diagonal target')
    idx = diagonal.long().argmax(1)
    batch = torch.arange(len(idx), device=prediction.device)
    mask = valid[batch,idx]
    if not torch.all(mask.flatten(1).any(1)):
        raise ValueError('every source must have valid diagonal GT')
    xyz = prediction[batch,idx].float()
    mean = torch.as_tensor(mean,device=xyz.device,dtype=xyz.dtype).view(1,3,1,1)
    scale = torch.as_tensor(scale,device=xyz.device,dtype=xyz.dtype).view(1,3,1,1)
    xyz = torch.where(mask[:,None], xyz*scale+mean, torch.zeros_like(xyz))
    rays = unit_source_rays(K[batch,source[:,0]], *xyz.shape[-2:])
    distance = (xyz*rays).sum(1)
    perp = xyz - distance[:,None]*rays
    sigma = scale.square().mean().sqrt()
    loss_map = F.smooth_l1_loss(perp/sigma, torch.zeros_like(perp), beta=beta, reduction='none').mean(1)
    counts = mask.sum((-2,-1))
    ray = (loss_map.sum((-2,-1)) / counts).mean()
    front_map = torch.where(mask, F.relu(1e-4-distance)/sigma, torch.zeros_like(distance))
    front = (front_map.sum((-2,-1))/counts).mean()
    deviation = (perp.detach().norm(dim=1).sum((-2,-1))/counts).mean()
    return ray, front, deviation


def ray_field_loss(output: CameraOutput, K: torch.Tensor, height: int, width: int):
    """Angular loss of the predicted per-frame ray field against GT rays."""
    if output.rays is None or output.ray_frames is None:
        raise RuntimeError('H033 ray supervision requires a predicted ray field')
    rays = output.rays.float()
    batch, channels, grid, grid_w = rays.shape
    if channels != 3 or grid != grid_w:
        raise ValueError(f'ray field must be [B,3,G,G], got {tuple(rays.shape)}')
    batch_index = torch.arange(batch, device=rays.device)
    frames = output.ray_frames.to(device=rays.device, dtype=torch.long)
    ys, xs = pixel_grid_coordinates(grid, height, width, rays.device, rays.dtype)
    gt = unit_rays_at(K[batch_index, frames].to(rays.dtype), ys, xs)
    cosine = (rays * gt).sum(1).clamp(-1.0, 1.0)
    field = (1.0 - cosine).mean()
    with torch.no_grad():
        angle = torch.rad2deg(torch.acos(cosine)).mean()
    return field, angle


def decoded_intrinsics_metrics(output: CameraOutput, K: torch.Tensor, height: int, width: int):
    """Decode K from the predicted ray field and compare with the real GT K."""
    with torch.no_grad():
        return _decoded_intrinsics_metrics(output, K, height, width)


def _decoded_intrinsics_metrics(output: CameraOutput, K: torch.Tensor, height: int, width: int):
    decoded = output.decode_pinhole(height, width)
    valid = decoded['valid']
    batch = output.rays.shape[0]
    batch_index = torch.arange(batch, device=output.rays.device)
    frames = output.ray_frames.to(device=output.rays.device, dtype=torch.long)
    gt_focal = torch.stack((K[batch_index, frames, 0, 0], K[batch_index, frames, 1, 1]), dim=-1).float()
    gt_principal = torch.stack((K[batch_index, frames, 0, 2], K[batch_index, frames, 1, 2]), dim=-1).float()
    predicted_focal = torch.stack((decoded['focal_x'], decoded['focal_y']), dim=-1)
    predicted_principal = torch.stack((decoded['principal_x'], decoded['principal_y']), dim=-1)
    gt_fov = 2 * torch.atan(torch.as_tensor([width, height], device=gt_focal.device, dtype=gt_focal.dtype)
                            / (2 * gt_focal.clamp_min(1e-6)))
    if not bool(valid.any()):
        zeros = torch.zeros((), device=gt_focal.device)
        return dict(fov=zeros, focal_relative_error=zeros, principal_offset_px=zeros,
                    decoded_valid_fraction=zeros)
    relation = torch.tan(gt_fov[valid] / 2) / torch.tan(decoded['fov'][valid] / 2)
    return dict(
        fov=F.smooth_l1_loss(decoded['fov'][valid], gt_fov[valid], beta=0.05),
        focal_relative_error=(relation - 1).abs().mean(),
        principal_offset_px=(predicted_principal[valid] - gt_principal[valid]).abs().mean(),
        decoded_valid_fraction=valid.float().mean(),
    )


def camera_supervision_losses(output: CameraOutput, source, target, K, rotations, positions,
                              scale, height, width, cfg, dataset=None):
    source = torch.as_tensor(source, device=target.device, dtype=torch.long)
    target = torch.as_tensor(target, device=target.device, dtype=torch.long)
    if source.ndim == 1:
        source = source[:, None].expand(-1, target.shape[1])
    # The training contract samples all K pairs of one shared source per batch
    # item, so GT relative poses can be built once per source and gathered.
    if not bool((source == source[:, :1]).all()):
        raise ValueError('camera pose supervision requires one shared source per batch item')
    reference = source[:, 0]
    orth = (rotations.transpose(-1,-2) @ rotations - torch.eye(3,device=rotations.device)).abs().amax((-2,-1))
    det_error = (torch.linalg.det(rotations)-1).abs()
    valid_pose = (orth <= 0.003) & (det_error <= 0.003)
    batch_index = torch.arange(len(reference), device=reference.device)
    # Never invert an invalid source matrix (possibly singular); placeholders
    # below are used ONLY for masked pose labels, not for geometry or rays.
    safe_R = torch.where(valid_pose[...,None,None],rotations,torch.eye(3,device=rotations.device))
    safe_p = torch.where(valid_pose[...,None],positions,torch.zeros_like(positions))
    gt_R_all, gt_p_all = relative_pose_gt(safe_R, safe_p, reference)
    gt_R = gt_R_all[batch_index[:,None], target]
    gt_p = gt_p_all[batch_index[:,None], target]
    mask = ((target != source) & valid_pose[batch_index[:,None], target]
            & valid_pose[batch_index[:,None], reference[:,None]])
    count = mask.sum().clamp_min(1)
    rot = ((output.rotation.float()-gt_R).square().sum((-2,-1))*mask).sum()/count/8
    sigma = torch.as_tensor(
        float(cfg['pose_translation_scale'][dataset]) if isinstance(cfg['pose_translation_scale'], dict)
        else float(cfg['pose_translation_scale']),
        device=source.device, dtype=torch.float32,
    )
    trans_map = F.smooth_l1_loss(output.translation.float()/sigma, gt_p/sigma, beta=0.05, reduction='none').mean(-1)
    trans = (trans_map*mask).sum()/count
    ray, ray_angle = ray_field_loss(output, K, height, width)
    with torch.no_grad():
        rel = output.rotation.float().transpose(-1,-2) @ gt_R
        cosine = ((rel.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)
        angle = (torch.rad2deg(torch.acos(cosine))*mask).sum()/count
        terr = ((output.translation.float()-gt_p).norm(dim=-1)*mask).sum()/count
    losses = dict(pose_rotation=rot, pose_translation=trans,
                  rotation_deg=angle, translation_m=terr, ray_field=ray, ray_angle_deg=ray_angle,
                  pose_valid_pairs=mask.sum().float(),
                  pose_valid_fraction=mask.sum().float()/mask.numel())
    losses.update(decoded_intrinsics_metrics(output, K, height, width))
    return losses


def camera_ray_objective(prediction, target_xyz, valid, source, target, output,
                         cameras, mean, scale, cfg, beta=0.05, dataset=None):
    if output is None:
        raise RuntimeError('camera-enabled training did not return camera output')
    K, R, p = cameras
    diag = source == target
    safe_pred = torch.where(valid[:,:,None], prediction.float(), torch.zeros_like(prediction, dtype=torch.float32))
    safe_gt = torch.where(valid[:,:,None], target_xyz.float(), torch.zeros_like(target_xyz, dtype=torch.float32))
    errors = F.smooth_l1_loss(safe_pred,safe_gt,beta=beta,reduction='none').sum(2)
    counts = valid.sum((-2,-1))
    per_pair = errors.sum((-2,-1))/counts.clamp_min(1)
    if not (diag & (counts>0)).any() or not (~diag & (counts>0)).any():
        raise ValueError('balanced XYZ requires valid diagonal and non-diagonal pairs')
    diag_xyz = per_pair[diag & (counts>0)].mean()
    off_xyz = per_pair[~diag & (counts>0)].mean()
    ray, front, deviation = diagonal_ray_loss(prediction,valid,source,target,K,mean,scale,beta)
    losses = camera_supervision_losses(output, source, target, K, R, p, scale,
                                      *prediction.shape[-2:], cfg, dataset)
    losses.update(diagonal_xyz=diag_xyz, offdiagonal_xyz=off_xyz, diagonal_ray=ray,
                  front=front, ray_deviation_m=deviation)
    weights = cfg['loss_weights']
    total = sum(float(weights[name])*losses[name] for name in weights)
    return total, losses
