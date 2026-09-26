"""Read-only GT baselines: how much better than trivial are logged camera metrics?

Computes, per dataset, over a deterministic clip subsample:
  - identity-pose baseline (predict R=I, t=0): mean rotation angle / translation norm
    of the GT relative pose, under the same valid-pose masking the trainer uses;
  - constant-focal baselines (clip-mean K and dataset-mean K) using the same
    focal_relative_error definition as training.

No model, no GPU, no checkpoint writes. Purpose: calibrate absolute magnitudes.
"""
import json
import os
from pathlib import Path
import sys
import numpy as np
import yaml
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from worldbridge.data.factory import load_training_datasets
from worldbridge.trainer.camera_objective import relative_pose_gt, validate_supervision_camera
from worldbridge.utils.io import atomic_json

OUT = Path('/data/WorldBridge4D-runs/h032-camera-baseline-20260924')
CLIPS = 512
SO3_TOL = 0.003


def rotation_angle_deg(rel):
    cosine = np.clip((np.trace(rel, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cosine))


def relative_pose(R, p, source):
    # rel[s, t] = inv(R_s) @ R_t ; p_rel[s, t] = inv(R_s) @ (p_t - p_s)
    # Explicit per-source solve: NumPy batch broadcasting differs from torch
    # for (S,1,3,3) against (S,3,3) and would silently drop the pair axis.
    Rs, ps = R[source], p[source]
    rel = np.stack([np.linalg.solve(Rs[i], R) for i in range(len(Rs))])
    trans = np.stack([np.linalg.solve(Rs[i], (p - ps[i]).T).T for i in range(len(Rs))])
    u, _, vh = np.linalg.svd(rel)
    sign = np.linalg.det(u @ vh)
    corr = np.ones_like(sign)[..., None].repeat(3, -1)
    corr[..., 2] = sign
    rel = (u * corr[..., None, :]) @ vh
    return rel, trans


def fov_from_K(K, width, height):
    return 2 * np.arctan(np.stack((width / (2 * K[:, 0, 0]), height / (2 * K[:, 1, 1])), -1))


def focal_relative_error(gt_fov, pred_fov):
    return np.abs(np.tan(gt_fov / 2) / np.tan(pred_fov / 2) - 1)


def verify_against_trainer(dataset):
    """The baseline must reproduce trainer semantics exactly (torch, one source per row)."""
    import torch
    camera = dataset.supervision_camera(0)
    R = np.asarray(camera['rotations'], np.float64)
    p = np.asarray(camera['positions'], np.float64)
    src = np.arange(len(R))
    rot, trans = relative_pose(R, p, src)
    gt_rot, gt_p = relative_pose_gt(torch.as_tensor(np.repeat(R[None], len(src), 0)),
                                    torch.as_tensor(np.repeat(p[None], len(src), 0)),
                                    torch.as_tensor(src))
    assert np.abs(rot - gt_rot.numpy()).max() < 1e-9
    assert np.abs(trans - gt_p.numpy()).max() < 1e-9


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only read-only analysis'
    assert not OUT.exists(), f'{OUT} already exists; do not overwrite'
    cfg = yaml.safe_load((ROOT / 'configs/h032_camera_ray_k10_to210000.yaml').read_text())
    datasets = load_training_datasets(cfg)
    report = {}
    for name, dataset in datasets.items():
        verify_against_trainer(dataset)
        size = int(getattr(dataset, 'image_size', 256))
        corpus = len(dataset)
        indices = sorted(set(np.linspace(0, corpus - 1, min(CLIPS, corpus)).astype(int).tolist()))
        rot_pairs, trans_pairs = [], []
        rot_sub, trans_sub = [], []
        clip_focal, frame_focal, frame_fov, tag = [], [], [], []
        skipped = 0
        for i in indices:
            camera = dataset.supervision_camera(i)
            try:
                validate_supervision_camera(camera, size, size)
            except ValueError:
                skipped += 1
                continue
            K = np.asarray(camera['intrinsics'], np.float64)
            R = np.asarray(camera['rotations'], np.float64)
            p = np.asarray(camera['positions'], np.float64)
            orth = np.max(np.abs(np.transpose(R, (0, 2, 1)) @ R - np.eye(3)), axis=(1, 2))
            det = np.abs(np.linalg.det(R) - 1)
            valid = (orth <= SO3_TOL) & (det <= SO3_TOL)
            fov = fov_from_K(K, size, size)
            clip_focal.append([K[:, 0, 0].mean(), K[:, 1, 1].mean()])
            frame_focal.append(np.stack((K[:, 0, 0], K[:, 1, 1]), -1))
            frame_fov.append(fov)
            tag.append(np.full(len(K), valid.all()))
            if not valid.all():
                continue
            # All ordered non-diagonal pairs, matching the trainer's mask shape.
            src = np.arange(21)
            rel, trans = relative_pose(R, p, src)
            off = ~np.eye(21, dtype=bool)
            rot_pairs.append(rotation_angle_deg(rel)[off])
            trans_pairs.append(np.linalg.norm(trans, axis=-1)[off])
            # Realistic K10 pairing: 9 random non-diagonal targets per source.
            rng = np.random.default_rng([20260812, int(i), 771])
            for s in range(21):
                targets = rng.choice([t for t in range(21) if t != s], 9, replace=False)
                rot_sub.append(rotation_angle_deg(rel[s, targets]))
                trans_sub.append(np.linalg.norm(trans[s, targets], axis=-1))
        if not rot_pairs:
            report[name] = dict(corpus=corpus, sampled=len(indices), skipped=skipped,
                                error='no valid-pose clips in subsample')
            print(json.dumps(dict(event='H032_CAMERA_BASELINE_DATASET', dataset=name, **report[name])), flush=True)
            continue
        rots = np.concatenate(rot_pairs)
        transs = np.concatenate(trans_pairs)
        id_rot, id_trans = float(rots.mean()), float(transs.mean())
        id_rot_med, id_trans_med = float(np.median(rots)), float(np.median(transs))
        sub_rot, sub_trans = float(np.mean(rot_sub)), float(np.mean(trans_sub))
        fov_all = np.concatenate(frame_fov)
        focal_all = np.concatenate(frame_focal)
        clip_mean = np.stack(clip_focal)
        gt_fov = fov_all
        pred_clip = np.concatenate([np.repeat(Km[None], 21, 0) for Km in clip_mean])
        pred_clip_fov = fov_from_K(np.stack([np.diag([fx, fy, 1.0]) for fx, fy in pred_clip]), size, size)
        pred_global = np.stack([np.diag([focal_all[:, 0].mean(), focal_all[:, 1].mean(), 1.0])] * len(focal_all))
        pred_global_fov = fov_from_K(pred_global, size, size)
        clip_err = focal_relative_error(gt_fov, pred_clip_fov)
        global_err = focal_relative_error(gt_fov, pred_global_fov)
        report[name] = dict(
            corpus=corpus, sampled=len(indices), clips_used=len(rot_pairs), skipped=skipped,
            image_size=size,
            identity_pose_baseline=dict(
                rotation_deg_mean=id_rot, rotation_deg_median=id_rot_med,
                rotation_deg_p90=float(np.percentile(rots, 90)),
                translation_m_mean=id_trans, translation_m_median=id_trans_med,
                translation_m_p90=float(np.percentile(transs, 90)),
                pairs=int(rots.size)),
            identity_pose_baseline_k10_pairing=dict(
                rotation_deg_mean=sub_rot, translation_m_mean=sub_trans, targets_per_source=9),
            intrinsics_constant_baseline=dict(
                clip_mean_focal_relative_error_mean=float(clip_err.mean()),
                clip_mean_focal_relative_error_median=float(np.median(clip_err)),
                dataset_mean_focal_relative_error_mean=float(global_err.mean()),
                gt_focal_spread_relative=float(np.mean(np.std(focal_all, 0) / np.mean(focal_all, 0)))),
        )
        print(json.dumps(dict(event='H032_CAMERA_BASELINE_DATASET', dataset=name, **report[name])), flush=True)
    OUT.mkdir(exist_ok=False)
    atomic_json(OUT / 'baseline.json', report)
    atomic_json(OUT / 'complete.json', dict(event='H032_CAMERA_BASELINE_OK', clips=CLIPS, report=report))
    print('H032_CAMERA_BASELINE_OK', flush=True)


if __name__ == '__main__':
    main()
