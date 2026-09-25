"""Generate the H033 protocol: decoder-native camera token plus ray field.

Minimal diff from the H032 K10 protocol: only ``camera_supervision``, the camera
bookkeeping fields and identity/tag metadata change. Trunk, data mixture,
sampling (K10), optimizer groups and seeds are inherited unchanged.
"""
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[2]
PARENT_STEP = 210000


def camera_parameter_count(query_dim: int, pose_hidden: int, ray_hidden: int) -> int:
    """Exact count of CameraQueryHead + RayFieldHead parameters."""
    return (query_dim                                                        # camera token
            + 2*query_dim + (query_dim*pose_hidden + pose_hidden)             # pose LN + Linear
            + (pose_hidden*pose_hidden + pose_hidden)                         # pose Linear
            + (pose_hidden*3 + 3) + (pose_hidden*4 + 4)                       # translation/quaternion
            + ((query_dim+2)*ray_hidden + ray_hidden)                         # ray Linear
            + (ray_hidden*ray_hidden + ray_hidden)                            # ray Linear
            + (ray_hidden*3 + 3))                                             # ray output


def camera_supervision(query_dim: int = 1536) -> dict:
    return dict(
        pose_hidden=256, ray_hidden=256, seed=424243, learning_rate=1e-4,
        warmup_steps=500, phase_start_step=196000,
        intrinsics_mode='per_frame_source_independent',
        # Camera-scale translation normalisation replaces the point-cloud sigma
        # (5.62 m) that left PO/DR translation supervision in smooth-L1's
        # quadratic region. Values are the measured median relative translation.
        pose_translation_scale=dict(kubric=0.30, pointodyssey=0.032, dynamic_replica=0.020),
        loss_weights=dict(diagonal_xyz=0.5, offdiagonal_xyz=0.5, diagonal_ray=0.1,
                          ray_field=0.1, front=0.01, pose_rotation=0.1, pose_translation=0.1),
    )


def make_config(parent: dict | None = None, clips_seen: dict | None = None) -> dict:
    cfg = dict(parent) if parent else yaml.safe_load(
        (ROOT/'configs/h032_camera_ray_k10_to210000.yaml').read_text())
    trunk_non_wan = 194597133
    # H033 deleted the cycle, boundary, edge-contrast, schedule-extension and
    # staged-input machinery, so none of those keys may survive into the config.
    for key in ('cycle_reprojection_enabled', 'cycle_reprojection_datasets',
                'cycle_reprojection_weight', 'cycle_reprojection_pixel_stride',
                'cycle_reprojection_huber_delta', 'cycle_reprojection_normalize_to_xyz',
                'cycle_reprojection_normalization_epsilon',
                'cycle_reprojection_normalization_max_scale', 'boundary_supervision',
                'source_edge_contrast_weight', 'schedule_extension_start_step',
                'schedule_extension_horizon_steps'):
        cfg.pop(key, None)
    cfg['camera_supervision'] = camera_supervision()
    cfg['expected_non_wan_parameters'] = trunk_non_wan + camera_parameter_count(1536, 256, 256)
    cfg['finetune_expected_global_step'] = PARENT_STEP
    # The parent counters only exist once the parent run reaches its endpoint, so
    # the gate reads them from the pinned checkpoint and writes them here.
    if clips_seen is None:
        cfg.pop('finetune_expected_clips_seen', None)
    else:
        cfg['finetune_expected_clips_seen'] = {str(k): int(v) for k, v in clips_seen.items()}
    cfg['finetune_drop_prefixes'] = ['camera_head.']
    cfg['max_steps'] = 210000
    cfg['lr_restart']['group_learning_rates']['camera_head'] = 1e-4
    cfg['tracking']['group'] = 'h033-decoder-camera-query-ray-field'
    cfg['tracking']['tags'] = ['h033', 'decoder-camera-query', 'ray-field',
                               'camera-scale-translation', 'b1-a4-k10', 'native512-full']
    return cfg


if __name__ == '__main__':
    (ROOT/'configs/h033_camera_query_ray_to210000.yaml').write_text(
        yaml.safe_dump(make_config(), sort_keys=False))
