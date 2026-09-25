"""Why is pose at the trivial baseline? Two read-only CPU diagnostics.

A) Loss-space GT baselines under the exact training definitions
   (sigma from mixture coordinate stats, beta=0.05, trainer masking), so logged
   camera losses can be compared against: predict identity rotation + zero
   translation, and predict the dataset-mean relative pose.
B) Camera-head parameter and Adam drift across saved checkpoints. If the 51
   camera tensors barely moved since196020, the head is gradient-starved; if
   they moved substantially while metrics stayed at baseline, it is fitting the
   prior/mean (information bottleneck), not a capacity problem.
"""
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
import yaml
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'research/analysis'))
from h032_camera_baseline import (fov_from_K, focal_relative_error, relative_pose,  # noqa: E402
                                  rotation_angle_deg, verify_against_trainer)
from worldbridge.data.factory import load_training_datasets  # noqa: E402
from worldbridge.trainer.camera_objective import validate_supervision_camera  # noqa: E402
from worldbridge.trainer.tracking import load_stats  # noqa: E402
from worldbridge.utils.io import atomic_json  # noqa: E402

OUT = Path('/data/WorldBridge4D-runs/h032-camera-diagnose-20260925')
CLIPS = 256
SO3_TOL = 0.003
BETA = 0.05
CKPTS = [
    ('196020', '/data/WorldBridge4D-runs/h032-camera-ray-k10-handoff-20260924/capacity.pt'),
    ('198000', '/data/WorldBridge4D-runs/h032-source-camera-ray-k10-to210000-20260924/checkpoint-0198000.pt'),
    ('199000', '/data/WorldBridge4D-runs/h032-source-camera-ray-k10-to210000-20260924/checkpoint-0199000.pt'),
    ('200000', '/data/WorldBridge4D-runs/h032-source-camera-ray-k10-to210000-20260924/checkpoint-0200000.pt'),
]


def smooth_l1_zero(x, beta=BETA):
    return torch.where(x.abs() < beta, 0.5 * x.square() / beta, x.abs() - 0.5 * beta)


def loss_baselines(cfg, datasets, sigma):
    report = {}
    for name, dataset in datasets.items():
        size = int(getattr(dataset, 'image_size', 256))
        corpus = len(dataset)
        indices = sorted(set(np.linspace(0, corpus - 1, min(CLIPS, corpus)).astype(int).tolist()))
        rot_pairs, trans_pairs, fov_pairs, valid_frames = [], [], [], 0
        for i in indices:
            camera = dataset.supervision_camera(i)
            try:
                validate_supervision_camera(camera, size, size)
            except ValueError:
                continue
            K = np.asarray(camera['intrinsics'], np.float64)
            R = np.asarray(camera['rotations'], np.float64)
            p = np.asarray(camera['positions'], np.float64)
            orth = np.max(np.abs(np.transpose(R, (0, 2, 1)) @ R - np.eye(3)), axis=(1, 2))
            valid = (orth <= SO3_TOL) & (np.abs(np.linalg.det(R) - 1) <= SO3_TOL)
            fov_pairs.append(fov_from_K(K, size, size))
            valid_frames += int(valid.all())
            if not valid.all():
                continue
            rel, trans = relative_pose(R, p, np.arange(len(R)))
            off = ~np.eye(len(R), dtype=bool)
            rot_pairs.append(rel[off])
            trans_pairs.append(trans[off])
        if not rot_pairs:
            report[name] = dict(sampled=len(indices), clips_used=0, note='no valid-pose clip')
            print(json.dumps(dict(event='H032_CAMERA_DIAG_LOSS', dataset=name, **report[name])), flush=True)
            continue
        rel = np.concatenate(rot_pairs)
        trans = np.concatenate(trans_pairs)
        fov = np.concatenate(fov_pairs)
        pairs = len(rel)
        t = torch.as_tensor
        # identity rotation + zero translation predictor
        id_rot_loss = float((t(rel) - torch.eye(3, dtype=torch.float64)).square().sum((-2, -1)).mean() / 8)
        id_trans_loss = float(smooth_l1_zero(t(trans) / sigma).mean())
        id_rot_deg = float(rotation_angle_deg(rel).mean())
        id_trans_m = float(np.linalg.norm(trans, axis=-1).mean())
        # dataset-mean relative pose predictor (chordal SO(3) mean + mean vector)
        mean_matrix = t(rel).mean(0)
        u, _, vh = torch.linalg.svd(mean_matrix)
        sign = torch.det(u @ vh)
        corr = torch.ones(3, dtype=torch.float64)
        corr[2] = sign
        r_mean = (u * corr) @ vh
        t_mean = t(trans).mean(0)
        mean_rot_loss = float((t(rel) - r_mean).square().sum((-2, -1)).mean() / 8)
        mean_trans_loss = float(smooth_l1_zero((t(trans) - t_mean) / sigma).mean())
        rel_to_mean = r_mean.numpy()[None].transpose(0, 2, 1) @ rel
        mean_rot_deg = float(rotation_angle_deg(rel_to_mean).mean())
        mean_trans_m = float(np.linalg.norm(trans - t_mean.numpy()[None], axis=-1).mean())
        # intrinsics in loss space: dataset-constant and clip-constant (oracle)
        fov_t = t(fov)
        ds_const = fov_t.mean(0)
        ds_fov_loss = float(smooth_l1_zero(fov_t - ds_const).mean())
        ds_focal_err = float(focal_relative_error(np.asarray(fov),
                                                 np.broadcast_to(np.asarray(ds_const.numpy()), np.asarray(fov).shape)).mean())
        report[name] = dict(
            sampled=len(indices), clips_used=len(rot_pairs), pose_pairs=int(pairs),
            sigma=sigma,
            identity_predictor=dict(rotation_loss=id_rot_loss, translation_loss=id_trans_loss,
                                    rotation_deg=id_rot_deg, translation_m=id_trans_m),
            dataset_mean_pose_predictor=dict(rotation_loss=mean_rot_loss, translation_loss=mean_trans_loss,
                                             rotation_deg=mean_rot_deg, translation_m=mean_trans_m,
                                             mean_translation_norm=float(np.linalg.norm(t_mean.numpy()))),
            intrinsics_dataset_constant_predictor=dict(fov_loss=ds_fov_loss,
                                                        focal_relative_error=ds_focal_err),
            gt_fov_mean_rad=[float(x) for x in np.asarray(fov).mean(0)],
        )
        print(json.dumps(dict(event='H032_CAMERA_DIAG_LOSS', dataset=name, **report[name])), flush=True)
    return report


def parameter_drift():
    report = {}
    baseline = None
    for step, path in CKPTS:
        p = Path(path)
        if not p.is_file():
            report[step] = dict(missing=True, path=str(p))
            continue
        d = torch.load(p, map_location='cpu', mmap=True, weights_only=True)
        weights = {k: v.float().clone() for k, v in d['model'].items() if k.startswith('camera_head.')}
        opt_groups = {g['name']: len(g['params']) for g in d['optimizer']['param_groups']}
        cam_keys = sorted(k for k in d['optimizer']['state'] if k.startswith('camera_head.'))
        moments = {k: (d['optimizer']['state'][k]['exp_avg'].float().clone(),
                       d['optimizer']['state'][k]['exp_avg_sq'].float().clone(),
                       float(d['optimizer']['state'][k]['step']))
                   for k in cam_keys}
        del d
        if baseline is None:
            baseline = weights
        drift = {k: float((weights[k] - baseline[k]).norm() / (baseline[k].norm() + 1e-12)) for k in weights}
        wnorm = {k: float(weights[k].norm()) for k in weights}
        m_ratio = {k: float(moments[k][0].norm() / (wnorm[k] + 1e-12)) for k in moments}
        implied = {}
        for k in moments:
            m, v, st = moments[k]
            step_size = 1e-4 * (m.abs() / (v.sqrt() + 1e-8))
            implied[k] = float(step_size.mean())
        report[step] = dict(
            path=str(p), camera_tensors=len(weights), optimizer_groups=opt_groups,
            optimizer_camera_states=len(cam_keys), adam_step=next(iter(moments.values()))[2] if moments else None,
            relative_weight_drift_vs196020=dict(
                mean=float(np.mean(list(drift.values()))), max=float(np.max(list(drift.values()))),
                per_tensor={k: round(v, 5) for k, v in sorted(drift.items(), key=lambda kv: -kv[1])[:8]}),
            mean_abs_exp_avg_over_weight_norm=float(np.mean(list(m_ratio.values()))),
            mean_implied_adam_step=float(np.mean(list(implied.values()))),
        )
        print(json.dumps(dict(event='H032_CAMERA_DIAG_DRIFT', step=step, **report[step])), flush=True)
    return report


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only read-only analysis'
    assert not OUT.exists(), f'{OUT} already exists; do not overwrite'
    cfg = yaml.safe_load((ROOT / 'configs/h032_camera_ray_k10_to210000.yaml').read_text())
    _, scale = load_stats(cfg)
    sigma = torch.as_tensor(np.asarray(scale, np.float32)).square().mean().sqrt().item()
    datasets = load_training_datasets(cfg)
    for dataset in datasets.values():
        verify_against_trainer(dataset)
    loss_report = loss_baselines(cfg, datasets, sigma)
    drift_report = parameter_drift()
    OUT.mkdir(exist_ok=False)
    atomic_json(OUT / 'loss_baselines.json', loss_report)
    atomic_json(OUT / 'parameter_drift.json', drift_report)
    atomic_json(OUT / 'complete.json', dict(event='H032_CAMERA_DIAGNOSE_OK', sigma=sigma, beta=BETA,
                                            clips=CLIPS, loss_baselines=loss_report, parameter_drift=drift_report))
    print('H032_CAMERA_DIAGNOSE_OK', flush=True)


if __name__ == '__main__':
    main()
