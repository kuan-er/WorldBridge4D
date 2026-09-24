"""GT camera supervision and source-diagonal ray loss (no predicted-K coupling)."""
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
        raise ValueError('camera head requires zero skew pinhole K')
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


def unit_source_rays(K: torch.Tensor, height: int, width: int):
    yy, xx = torch.meshgrid(torch.arange(height, device=K.device, dtype=K.dtype),
                            torch.arange(width, device=K.device, dtype=K.dtype), indexing='ij')
    x = (xx[None]-K[:,0,2,None,None]) / K[:,0,0,None,None]
    y = -(yy[None]-K[:,1,2,None,None]) / K[:,1,1,None,None]
    return F.normalize(torch.stack((x,y,-torch.ones_like(x)),dim=1), dim=1)


def diagonal_ray_loss(prediction, valid, source, target, K, mean, scale, beta=0.05):
    """Only diagonal, physical coordinates, fixed scalar scale; mask invalids first."""
    diagonal = source == target
    if not torch.all(diagonal.sum(1) == 1):
        raise ValueError('every source must have exactly one diagonal target')
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


def camera_supervision_losses(output: CameraOutput, source, K, rotations, positions, scale, height, width):
    orth = (rotations.transpose(-1,-2) @ rotations - torch.eye(3,device=rotations.device)).abs().amax((-2,-1))
    det_error = (torch.linalg.det(rotations)-1).abs()
    valid_pose = (orth <= 0.003) & (det_error <= 0.003)
    batch = torch.arange(len(source),device=source.device)
    mask = ((torch.arange(rotations.shape[1],device=source.device)[None] != source[:,None])
            & valid_pose & valid_pose[batch,source,None])
    # Never invert an invalid source matrix (possibly singular); placeholders
    # below are used ONLY for masked pose labels, not for geometry or rays.
    safe_R = torch.where(valid_pose[...,None,None],rotations,torch.eye(3,device=rotations.device))
    safe_p = torch.where(valid_pose[...,None],positions,torch.zeros_like(positions))
    gt_R, gt_p = relative_pose_gt(safe_R, safe_p, source)
    count = mask.sum().clamp_min(1)
    # Masked reductions remain differentiable finite zero with no valid poses.
    rot = ((output.rotation.float()-gt_R).square().sum((-2,-1))*mask).sum()/count/8
    sigma = torch.as_tensor(scale,device=source.device,dtype=torch.float32).square().mean().sqrt()
    trans_map = F.smooth_l1_loss(output.translation.float()/sigma, gt_p/sigma, beta=0.05,reduction='none').mean(-1)
    trans = (trans_map*mask).sum()/count
    gt_fov = 2*torch.atan(torch.stack((width/(2*K[...,0,0]),height/(2*K[...,1,1])),dim=-1))
    if output.fov.shape != gt_fov.shape:
        raise ValueError('intrinsics supervision requires per-frame [B,21,2] FOV, never clip averaging')
    focal = F.smooth_l1_loss(output.fov.float(),gt_fov,beta=0.05)
    with torch.no_grad():
        rel = output.rotation.float().transpose(-1,-2) @ gt_R
        cosine = ((rel.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)
        angle = (torch.rad2deg(torch.acos(cosine))*mask).sum()/count
        terr = ((output.translation.float()-gt_p).norm(dim=-1)*mask).sum()/count
        ferr = ((torch.tan(gt_fov/2)/torch.tan(output.fov.float()/2))-1).abs().mean()
    return dict(pose_rotation=rot, pose_translation=trans, fov=focal,
                rotation_deg=angle, translation_m=terr, focal_relative_error=ferr,
                pose_valid_pairs=mask.sum().float(), pose_valid_fraction=mask.sum().float()/(len(source)*(rotations.shape[1]-1)))


def camera_ray_objective(prediction, target_xyz, valid, source, target, output,
                         cameras, mean, scale, cfg, beta=0.05):
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
    losses = camera_supervision_losses(output,source[:,0],K,R,p,scale,*prediction.shape[-2:])
    losses.update(diagonal_xyz=diag_xyz, offdiagonal_xyz=off_xyz, ray=ray, front=front, ray_deviation_m=deviation)
    weights = cfg['loss_weights']
    total = sum(float(weights[name])*losses[name] for name in weights)
    return total, losses
