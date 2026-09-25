"""Fixed-clip evaluation set and GT-only baselines for the H033 camera head.

Tier A (this script; CPU-only, no model, no GPU): pin one deterministic clip list
per dataset and compute the trivial baselines on exactly those clips, plus the GT
pose-error structure versus the distance to the nearest Wan latent anchor
(0,4,8,12,16,20) that the 6->21 temporal mixing interpolates between.

Tier B (after210k, GPU) will evaluate the H032 and H033 checkpoints on the same
clips and compare against these numbers; without Tier A the comparison would have
no reference for what "learned" means.

Caveat recorded up front: the trainer samples clips uniformly per step, so this
is a *fixed* set, not a held-out split.
"""
import json
import os
from pathlib import Path
import sys
import numpy as np
import yaml
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'research/analysis'))
from h032_camera_baseline import (focal_relative_error, fov_from_K, relative_pose,
                                  rotation_angle_deg)
from worldbridge.data.factory import load_training_datasets
from worldbridge.trainer.camera_objective import validate_supervision_camera
from worldbridge.utils.io import atomic_json

OUT = Path('/data/WorldBridge4D-runs/h033-camera-eval-set-20260925')
CLIPS_PER_DATASET = 32
ANCHORS = (0, 4, 8, 12, 16, 20)
SO3_TOL = 0.003


def anchor_distance(frame: int) -> int:
    return min(abs(int(frame) - anchor) for anchor in ANCHORS)


def evaluate_clip(dataset, index: int):
    camera = dataset.supervision_camera(index)
    size = int(getattr(dataset, 'image_size', 256))
    report = validate_supervision_camera(camera, size, size)
    K = np.asarray(camera['intrinsics'], np.float64)
    R = np.asarray(camera['rotations'], np.float64)
    p = np.asarray(camera['positions'], np.float64)
    orth = np.max(np.abs(np.transpose(R, (0, 2, 1)) @ R - np.eye(3)), axis=(1, 2))
    valid = (orth <= SO3_TOL) & (np.abs(np.linalg.det(R) - 1) <= SO3_TOL)
    fov = fov_from_K(K, size, size)
    focal = np.stack((K[:, 0, 0], K[:, 1, 1]), -1)
    if not valid.all():
        return dict(index=int(index), size=size, pose_valid_frames=int(valid.sum()),
                    rotation_deg=None, translation_m=None,
                    focal_clip_mean_error=float(focal_relative_error(fov, np.repeat(
                        fov.mean(0)[None], 21, 0)).mean()),
                    rotation_deg_by_distance={}, translation_m_by_distance={})
    src = np.arange(21)
    rel, trans = relative_pose(R, p, src)
    off = ~np.eye(21, dtype=bool)
    rot = rotation_angle_deg(rel)[off]
    tr = np.linalg.norm(trans, axis=-1)[off]
    pairs = np.stack(np.nonzero(off), axis=1)          # (source, target) rows
    by_distance = {}
    for distance in range(max(abs(t - a) for t in range(21) for a in ANCHORS) + 1):
        keep = np.array([anchor_distance(t) == distance for _, t in pairs])
        if keep.any():
            by_distance[str(distance)] = dict(pairs=int(keep.sum()),
                                              rotation_deg=float(rot[keep].mean()),
                                              translation_m=float(tr[keep].mean()))
    return dict(index=int(index), size=size, pose_valid_frames=int(valid.sum()),
                rotation_deg=float(rot.mean()), translation_m=float(tr.mean()),
                focal_clip_mean_error=float(focal_relative_error(fov, np.repeat(
                    fov.mean(0)[None], 21, 0)).mean()),
                rotation_deg_by_distance={k: v['rotation_deg'] for k, v in by_distance.items()},
                translation_m_by_distance={k: v['translation_m'] for k, v in by_distance.items()},
                pairs_by_distance={k: v['pairs'] for k, v in by_distance.items()},
                invalid_pose_frames=report['invalid_pose_frames'])


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Tier A is CPU-only'
    assert not OUT.exists(), f'{OUT} already exists'
    cfg = yaml.safe_load((ROOT/'configs/h033_camera_query_ray_to210000.yaml').read_text())
    datasets = load_training_datasets(cfg)
    clips, baselines = {}, {}
    for name, dataset in datasets.items():
        corpus = len(dataset)
        indices = sorted(set(np.linspace(0, corpus - 1, min(CLIPS_PER_DATASET, corpus)
                                         ).astype(int).tolist()))
        rows = [evaluate_clip(dataset, index) for index in indices]
        clips[name] = [row['index'] for row in rows]
        usable = [row for row in rows if row['rotation_deg'] is not None]
        ones = [row for row in rows if row['rotation_deg'] is None]
        all_focal = [row['focal_clip_mean_error'] for row in rows]
        aggregate = dict(
            corpus=corpus, sampled=len(indices), pose_usable=len(usable),
            skipped_invalid_pose=len(ones),
            identity_pose_baseline=dict(
                rotation_deg_mean=float(np.mean([row['rotation_deg'] for row in usable])),
                rotation_deg_median=float(np.median([row['rotation_deg'] for row in usable])),
                translation_m_mean=float(np.mean([row['translation_m'] for row in usable])),
                translation_m_median=float(np.median([row['translation_m'] for row in usable]))),
            focal_oracle_clip_constant=dict(mean=float(np.mean(all_focal)),
                                            median=float(np.median(all_focal))),
            by_anchor_distance={},
        )
        for distance in map(str, range(4)):
            keys = [row['rotation_deg_by_distance'].get(distance) for row in usable]
            keys = [value for value in keys if value is not None]
            trans = [row['translation_m_by_distance'].get(distance) for row in usable]
            trans = [value for value in trans if value is not None]
            if keys:
                aggregate['by_anchor_distance'][distance] = dict(
                    rotation_deg_mean=float(np.mean(keys)), translation_m_mean=float(np.mean(trans)))
        baselines[name] = aggregate
        print(json.dumps(dict(event='H033_EVAL_SET_DATASET', dataset=name, **aggregate)), flush=True)
    OUT.mkdir(exist_ok=False)
    atomic_json(OUT/'eval_clips.json', clips)
    atomic_json(OUT/'baselines.json', baselines)
    atomic_json(OUT/'complete.json', dict(
        event='H033_EVAL_SET_OK', clips_per_dataset=CLIPS_PER_DATASET, anchors=list(ANCHORS),
        note='fixed deterministic clips, not a held-out split; Tier B reuses eval_clips.json',
        clips=clips, baselines=baselines))
    print('H033_EVAL_SET_OK', flush=True)


if __name__ == '__main__':
    main()
