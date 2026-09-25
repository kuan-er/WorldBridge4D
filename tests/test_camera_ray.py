"""H033 camera contract: decoder-native pose token plus per-frame ray field."""
from copy import deepcopy
import math
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
import importlib.util

import yaml
from worldbridge.models.camera import (
    CameraOutput, normalised_grid_coordinates, quaternion_to_matrix,
)
from worldbridge.models.decoder import DenseQueryDecoder
from worldbridge.models.outputs import DenseQueryOutput, StructuredZ4D
from worldbridge.models.worldbridge import DenseQueryWanModel
from worldbridge.data.sampling import sample_eligible_targets
from worldbridge.trainer.camera_objective import (
    camera_ray_objective, camera_supervision_losses, diagonal_ray_loss, ray_field_loss,
    relative_pose_gt, unit_rays_at, unit_source_rays, validate_supervision_camera,
)
from worldbridge.trainer.config import validate_config
from worldbridge.trainer.optimizer import apply_fresh_group_warmup, parameter_groups
from worldbridge.models.decoder.query_decoder import CAMERA_MODULE_PREFIXES
from worldbridge.trainer.trainer import CAMERA_PARAMETER_PREFIXES

CONFIG = Path('configs/h033_camera_query_ray_to210000.yaml')
_SPEC = importlib.util.spec_from_file_location(
    'h033_make_config', Path(__file__).resolve().parents[1] / 'research/analysis/h033_make_config.py')
m033 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(m033)
TRUNK_NON_WAN = 194597133
GRID = 8


def config():
    return yaml.safe_load(CONFIG.read_text())


class TinyBackbone(nn.Module):
    def __init__(self, channels: int = 16):
        super().__init__()
        self.channels = channels
        self.gain = nn.Parameter(torch.ones(1))
        self.adapter_parameters = [self.gain]

    def forward(self, latent, encoder_hidden_states=None):
        batch = latent.shape[0]
        dense = torch.randn(batch, self.channels, 21, GRID, GRID) * self.gain
        motion = torch.randn(batch, 21, 2, self.channels)
        return StructuredZ4D(dense=dense, motion=motion)


def tiny_decoder(camera: bool = True) -> DenseQueryDecoder:
    torch.manual_seed(0)
    return DenseQueryDecoder(
        num_frames=21, latent_shape=(16, 21, GRID, GRID), query_dim=64, embedding_dim=16,
        num_layers=2, num_heads=4, upsample_channels=(64, 32, 16),
        output_size=(32, 32), query_grid_size=GRID, structured_motion_slots=2,
        structured_local_queries=True, source_rgb_pyramid=False, native_512=False,
        camera_supervision=dict(seed=424243, pose_hidden=32, ray_hidden=32) if camera else None,
    )


def batch(batch_size: int = 2, pairs: int = 4, source: int = 3, height: int = 32, width: int = 32,
          focal: float = 20.0, principal: tuple[float, float] | None = None):
    """Synthetic batch shaped like the K10 sampling contract (one diagonal pair)."""
    source_t = torch.full((batch_size, pairs), source, dtype=torch.long)
    target_t = torch.tensor([[source, source+1, source+2, source+3]], dtype=torch.long).expand(
        batch_size, -1).contiguous()
    K = torch.eye(3).repeat(batch_size, 21, 1, 1)
    K[:, :, 0, 0] = focal
    K[:, :, 1, 1] = focal
    cx, cy = principal if principal is not None else ((width - 1) / 2, (height - 1) / 2)
    K[:, :, 0, 2] = cx
    K[:, :, 1, 2] = cy
    R = torch.eye(3).repeat(batch_size, 21, 1, 1)
    p = torch.zeros(batch_size, 21, 3)
    p[:, :, 0] = torch.arange(21) * 0.2
    return source_t, target_t, (K, R, p)


def structured(batch_size: int = 2) -> StructuredZ4D:
    torch.manual_seed(7)
    return StructuredZ4D(dense=torch.randn(batch_size, 16, 21, GRID, GRID),
                         motion=torch.randn(batch_size, 21, 2, 16))


def forward(decoder: DenseQueryDecoder, source_t, target_t, z4d: StructuredZ4D | None = None):
    return decoder(z4d if z4d is not None else structured(source_t.shape[0]), source_t, target_t)


# --------------------------------------------------------------------------- protocol

def test_h033_config_is_the_single_supported_profile():
    cfg = config()
    validate_config(cfg, 2)
    assert cfg['targets_per_source'] == 10 and cfg['camera_k10'] is True
    assert cfg['native_kubric512_full'] is True and cfg['native_dr512'] is True
    assert cfg['trainable_mode'] == 'full' and cfg['precision'] == 'bf16'
    assert cfg['fsdp_master_precision'] == 'fp32'
    assert (cfg['gradient_accumulation'], cfg['microbatch_per_gpu']) == (4, 1)
    assert cfg['coordinate_frame'] == 'source'
    # The fork point and the endpoint come from the generator, so a re-planned
    # phase start (202000 now) cannot leave a stale literal behind.
    assert cfg['max_steps'] == m033.END_STEP == 210000
    assert cfg['lr_restart']['end_step'] == cfg['max_steps']
    assert cfg['finetune_expected_global_step'] == m033.PARENT_STEP
    assert cfg['camera_supervision']['phase_start_step'] == m033.PARENT_STEP
    assert cfg['checkpoint_steps'][-1] == m033.END_STEP
    assert cfg['checkpoint_steps'][0] == m033.PARENT_STEP + 1000
    assert cfg['finetune_drop_prefixes'] == ['camera_head.']
    assert cfg['camera_supervision'] == m033.camera_supervision()
    assert m033.camera_parameter_count(1536, 256, 256) + 194597133 == cfg['expected_non_wan_parameters']
    # H033 deleted these routes; a config that still carries them must be refused.
    for key in ('cycle_reprojection_enabled', 'boundary_supervision',
                'source_edge_contrast_weight', 'schedule_extension_start_step'):
        broken = config()
        broken[key] = 0
        with pytest.raises(ValueError):
            validate_config(broken, 2)
    for key in ('native_kubric512_b1_a4_k9', 'xyz_b2_a2_k15'):
        broken = config()
        broken[key] = True
        with pytest.raises(ValueError):
            validate_config(broken, 2)


def test_h033_rejects_unaudited_camera_profiles():
    for change in (
        {'loss_weights': {'diagonal_xyz': 0.5, 'offdiagonal_xyz': 0.5, 'ray': 0.1}},
        {'pose_translation_scale': {'kubric': 0.3}},
        {'pose_translation_scale': -1.0},
        {'intrinsics_mode': 'per_clip'},
        {'pose_hidden': 0},
    ):
        cfg = config()
        cfg['camera_supervision'] = {**cfg['camera_supervision'], **change}
        with pytest.raises(ValueError):
            validate_config(cfg, 2)
    cfg = config()
    cfg['camera_supervision'] = {**cfg['camera_supervision'], 'skip_zero_weight_cycle': False}
    with pytest.raises(ValueError):
        validate_config(cfg, 2)
    cfg = config()
    cfg['trainable_mode'] = 'decoder_only'
    with pytest.raises(ValueError):
        validate_config(cfg, 2)


def test_k10_exactly_one_diagonal_and_nine_non_diagonal():
    valid = np.ones((21, 3, 3), dtype=bool)
    for source in range(21):
        targets = sample_eligible_targets(valid, 10, np.random.default_rng(123), diagonal_source=source)
        assert len(targets) == len(set(targets)) == 10
        assert (targets == source).sum() == 1 and (targets != source).sum() == 9


def test_native512_query_grid_is_densified_but_the_upsampler_stays_32():
    decoder = DenseQueryDecoder(
        num_frames=21, latent_shape=(16, 21, 32, 32), query_dim=64, embedding_dim=16,
        num_layers=1, num_heads=4, upsample_channels=(64, 32, 16, 8), output_size=(256, 256),
        query_grid_size=32, structured_motion_slots=2, structured_local_queries=True,
        source_rgb_pyramid=True, native_512=True, camera_supervision=None,
    )
    assert decoder.query_grid_shape == (64, 64)
    assert decoder.upsampler_grid_shape == (32, 32)
    assert tuple(decoder.query_coordinates.shape) == (64 * 64, 2)
    # native512 uses integer grid coordinates, not the latent linspace.
    assert torch.equal(decoder.query_coordinates[1], torch.tensor([1.0, 0.0]))
    assert torch.equal(decoder.query_coordinates[-1], torch.tensor([63.0, 63.0]))


def test_camera_parameter_prefixes_match_the_model():
    decoder = tiny_decoder()
    names = {name for name, _ in decoder.named_parameters()}
    camera = {name for name in names if name.startswith(CAMERA_MODULE_PREFIXES)}
    assert camera == {name for name in names if name.split('.')[0] in ('camera_pose', 'camera_rays')}
    assert len(decoder.camera_parameters()) == len(camera) == 17
    assert decoder.camera_enabled
    assert CAMERA_PARAMETER_PREFIXES == ('decoder.camera_pose.', 'decoder.camera_rays.')


# --------------------------------------------------------------------------- readout

def test_quaternion_to_matrix_matches_scipy_reference():
    """The local formula is pinned against an independent implementation."""
    Rotation = pytest.importorskip('scipy.spatial.transform').Rotation
    rng = np.random.default_rng(11)
    q = rng.normal(size=(4096, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    reference = Rotation.from_quat(q).as_matrix()
    got = quaternion_to_matrix(torch.as_tensor(q, dtype=torch.float32)).double().numpy()
    assert np.abs(got - reference).max() < 1e-6
    assert np.abs(got @ got.transpose(0, 2, 1) - np.eye(3)).max() < 1e-6
    assert np.abs(np.linalg.det(got) - 1).max() < 1e-6
    # XYZW scalar-last, Hamilton, active rotation: 90 degrees about each axis.
    for axis in range(3):
        vector = np.zeros(3)
        vector[axis] = math.sin(math.pi / 4)
        quaternion = np.array([[vector[0], vector[1], vector[2], math.cos(math.pi / 4)]])
        assert np.allclose(quaternion_to_matrix(torch.tensor(quaternion))[0].numpy(),
                           Rotation.from_quat(quaternion[0]).as_matrix(), atol=1e-6)
    # The conversion always runs in FP32, even from a BF16 input under autocast.
    assert quaternion_to_matrix(torch.zeros(2, 4, dtype=torch.bfloat16)).dtype == torch.float32
    assert torch.isfinite(quaternion_to_matrix(torch.zeros(2, 4))).all()
    assert torch.allclose(quaternion_to_matrix(torch.zeros(2, 4)),
                          torch.eye(3).expand(2, 3, 3))


def test_quaternion_to_matrix_gradient_matches_central_differences():
    rng = np.random.default_rng(12)
    quaternion = torch.tensor(rng.normal(size=(1, 4)), dtype=torch.float32)
    weight = torch.tensor(rng.normal(size=(1, 3, 3)), dtype=torch.float32)
    tracked = quaternion.clone().requires_grad_(True)
    (quaternion_to_matrix(tracked) * weight).sum().backward()
    analytic = tracked.grad.clone()
    step = 1e-3
    numeric = torch.zeros_like(quaternion)
    for index in range(4):
        for sign in (1.0, -1.0):
            perturbed = quaternion.clone()
            perturbed[0, index] += sign * step
            numeric[0, index] += sign * (quaternion_to_matrix(perturbed) * weight).sum() / (2 * step)
    relative = float((analytic - numeric).abs().max() / analytic.abs().max())
    assert relative < 1e-3, relative


def test_pose_and_ray_shapes_diagonal_identity_and_unit_rays():
    decoder = tiny_decoder()
    source_t, target_t, _ = batch(pairs=4)
    output = forward(decoder, source_t, target_t)
    camera = output.camera
    assert camera is not None
    assert camera.rotation.shape == (2, 4, 3, 3)
    assert camera.translation.shape == (2, 4, 3)
    assert camera.rays.shape == (2, 3, GRID, GRID)
    assert camera.ray_frames.tolist() == [3, 3]
    assert torch.allclose(camera.rotation[:, 0], torch.eye(3).expand(2, 3, 3))
    assert torch.allclose(camera.translation[:, 0], torch.zeros(2, 3))
    assert torch.allclose(camera.rays.norm(dim=1), torch.ones(2, GRID, GRID), atol=1e-5)


def test_ray_field_and_pose_do_not_depend_on_unrelated_pairs():
    decoder = tiny_decoder()
    decoder.eval()
    source_t, target_t, _ = batch(pairs=4)
    other = target_t.clone()
    other[:, 2] = 21 - 1 - other[:, 2].clamp(max=20)  # change only non-diagonal pairs
    z4d = structured()
    with torch.no_grad():
        first = forward(decoder, source_t, target_t, z4d)
        second = forward(decoder, source_t, other, z4d)
    assert torch.equal(first.camera.rays, second.camera.rays)
    assert torch.equal(first.camera.rotation[:, 0], second.camera.rotation[:, 0])
    # A different diagonal frame is a different camera prediction.
    moved_source = source_t + 1
    with torch.no_grad():
        third = forward(decoder, moved_source, target_t + 1, z4d)
    assert not torch.equal(first.camera.rays, third.camera.rays)


def test_ray_decode_recovers_real_intrinsics_including_offset_principal_point():
    for principal in (None, (131.4, 98.2)):
        source_t, _, (K, _, _) = batch(batch_size=1, focal=250.0, principal=principal,
                                       height=256, width=256)
        grid = 32
        ys, xs = _pixel_grid(grid, 256, 256)
        rays = unit_rays_at(K[:, 0], ys, xs)
        output = CameraOutput(torch.eye(3)[None, None].expand(1, 1, 3, 3),
                              torch.zeros(1, 1, 3), rays, torch.zeros(1, dtype=torch.long))
        decoded = output.decode_pinhole(256, 256)
        expected_cx, expected_cy = principal if principal else (127.5, 127.5)
        assert float(decoded['focal_x']) == pytest.approx(250.0, abs=1e-3)
        assert float(decoded['focal_y']) == pytest.approx(250.0, abs=1e-3)
        assert float(decoded['principal_x']) == pytest.approx(expected_cx, abs=1e-3)
        assert float(decoded['principal_y']) == pytest.approx(expected_cy, abs=1e-3)
        assert bool(decoded['valid'])


def test_degenerate_ray_field_is_flagged_not_reported():
    rays = torch.zeros(1, 3, 8, 8)
    rays[:, 2] = -1.0  # perfectly flat: focal length is not identifiable
    output = CameraOutput(torch.eye(3)[None, None].expand(1, 1, 3, 3),
                          torch.zeros(1, 1, 3), rays, torch.zeros(1, dtype=torch.long))
    assert not bool(output.decode_pinhole(64, 64)['valid'])


def _pixel_grid(grid: int, height: int, width: int):
    from worldbridge.trainer.camera_objective import pixel_grid_coordinates
    return pixel_grid_coordinates(grid, height, width, torch.device('cpu'), torch.float32)


def test_gt_rays_use_real_principal_points_never_the_predicted_centre():
    K = torch.eye(3)[None].repeat(1, 1, 1)
    K[:, 0, 0] = 100.0
    K[:, 1, 1] = 120.0
    K[:, 0, 2] = 111.5
    K[:, 1, 2] = 77.25
    height = width = 32
    full = unit_source_rays(K, height, width)
    grid = 8
    ys, xs = _pixel_grid(grid, height, width)
    sampled = unit_rays_at(K, ys, xs)
    centre_only = unit_rays_at(K, torch.zeros(1), torch.zeros(1))
    assert not torch.allclose(sampled[:, :, 0, 0], sampled[:, :, -1, -1])
    # Grid pixel centres sit at (i+0.5)*scale-0.5, so the first centre is (1.5,1.5).
    expected = torch.tensor([(1.5 - 111.5) / 100.0, -(1.5 - 77.25) / 120.0, -1.0])
    expected = expected / expected.norm()
    assert torch.allclose(sampled[0, :, 0, 0], expected, atol=1e-6)
    assert torch.allclose(full[0, :, 2, 2], sampled[0, :, 0, 0], atol=4e-2)
    assert not torch.allclose(centre_only, sampled[:, :, 0, 0])


# --------------------------------------------------------------------------- objective

def test_objective_is_finite_balanced_and_camera_gradients_flow():
    decoder = tiny_decoder()
    source_t, target_t, cameras = batch()
    output = forward(decoder, source_t, target_t)
    cfg = config()['camera_supervision']
    prediction = torch.randn(2, 4, 3, 32, 32)
    target_xyz = torch.randn(2, 4, 3, 32, 32)
    valid = torch.ones(2, 4, 32, 32, dtype=torch.bool)
    total, losses = camera_ray_objective(prediction, target_xyz, valid, source_t, target_t,
                                         output.camera, cameras, torch.zeros(3), torch.ones(3),
                                         cfg, dataset='kubric')
    assert torch.isfinite(total)
    for key in ('diagonal_xyz', 'offdiagonal_xyz', 'diagonal_ray', 'front', 'ray_field',
                'ray_angle_deg', 'pose_rotation', 'pose_translation', 'rotation_deg',
                'translation_m', 'focal_relative_error', 'fov'):
        assert torch.isfinite(torch.as_tensor(losses[key])), key
    # Three non-diagonal pairs per item survive the diagonal mask.
    assert float(losses['pose_valid_pairs']) == 6.0
    total.backward()
    grads = {name: parameter.grad for name, parameter in decoder.named_parameters()
             if name.startswith(CAMERA_MODULE_PREFIXES)}
    assert grads and all(value is not None and torch.isfinite(value).all()
                         for value in grads.values())


def test_translation_supervision_uses_the_camera_scale_not_the_pointcloud_sigma():
    decoder = tiny_decoder()
    source_t, target_t, cameras = batch()
    output = forward(decoder, source_t, target_t)
    base = config()['camera_supervision']
    prediction = torch.zeros(2, 4, 3, 32, 32)
    target_xyz = torch.zeros(2, 4, 3, 32, 32)
    valid = torch.ones(2, 4, 32, 32, dtype=torch.bool)
    kubric = camera_supervision_losses(output.camera, source_t, target_t, *cameras,
                                       torch.ones(3), 32, 32, base, dataset='kubric')
    point = camera_supervision_losses(output.camera, source_t, target_t, *cameras,
                                      torch.ones(3), 32, 32, base, dataset='pointodyssey')
    assert float(kubric['pose_translation']) != pytest.approx(float(point['pose_translation']))
    # The metric itself is scale-free (metres), so it must not depend on sigma.
    assert float(kubric['translation_m']) == pytest.approx(float(point['translation_m']))
    assert float(kubric['translation_m']) > 0


def test_invalid_gt_pose_masks_only_its_own_labels():
    decoder = tiny_decoder()
    source_t, target_t, (K, R, p) = batch()
    shear = torch.tensor([[1.0, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    R[:, 5] = shear  # frame 5 has a non-rigid annotation base
    output = forward(decoder, source_t, target_t)
    cfg = config()['camera_supervision']
    losses = camera_supervision_losses(output.camera, source_t, target_t, K, R, p,
                                       torch.ones(3), 32, 32, cfg, dataset='kubric')
    assert torch.isfinite(losses['pose_rotation']) and torch.isfinite(losses['ray_field'])
    assert float(losses['pose_valid_fraction']) < 1.0
    # Frame 3 stays valid; frame 5 is the shared source for other pairs, so all
    # of its pairs are masked while the diagonal frame still has ray labels.
    assert 0.0 < float(losses['pose_valid_pairs']) < 6.0
    assert float(losses['ray_angle_deg']) >= 0.0


def test_ray_field_loss_is_zero_for_perfect_rays():
    source_t, target_t, (K, R, p) = batch(batch_size=2)
    ys, xs = _pixel_grid(GRID, 32, 32)
    rays = unit_rays_at(K[torch.arange(2), torch.tensor([3, 3])], ys, xs)
    output = CameraOutput(torch.eye(3)[None, None].expand(2, 1, 3, 3), torch.zeros(2, 1, 3),
                          rays, torch.tensor([3, 3]))
    field, angle = ray_field_loss(output, K, 32, 32)
    assert float(field) == pytest.approx(0.0, abs=1e-6)
    assert float(angle) == pytest.approx(0.0, abs=0.05)
    decoded = output.decode_pinhole(32, 32)
    assert float(decoded['focal_x'][0]) == pytest.approx(float(K[0, 3, 0, 0]), rel=1e-3)


def test_diagonal_ray_projection_still_uses_gt_intrinsics():
    source_t, target_t, (K, R, p) = batch()
    valid = torch.ones(2, 4, 32, 32, dtype=torch.bool)
    xyz = torch.zeros(2, 4, 3, 32, 32)
    mean = torch.zeros(3)
    scale = torch.ones(3)
    ray, front, deviation = diagonal_ray_loss(xyz, valid, source_t, target_t, K, mean, scale)
    assert float(ray) == pytest.approx(0.0, abs=1e-8)
    assert float(deviation) == pytest.approx(0.0, abs=1e-8)
    behind = xyz.clone()
    behind[:, 0, 2] = 5.0
    _, front_behind, _ = diagonal_ray_loss(behind, valid, source_t, target_t, K, mean, scale)
    assert float(front_behind) > 0.0


def test_relative_pose_direction_and_rounding_are_unchanged():
    K, perturbed, positions = None, None, None
    rotations = torch.eye(3)[None, None].repeat(1, 3, 1, 1)
    angle = torch.tensor(0.3)
    rotations[0, 1] = torch.tensor([[torch.cos(angle), -torch.sin(angle), 0.0],
                                    [torch.sin(angle), torch.cos(angle), 0.0],
                                    [0.0, 0.0, 1.0]])
    rotations[0, 2] = rotations[0, 1] @ rotations[0, 1]
    positions = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])
    reference = torch.tensor([0])
    R, p = relative_pose_gt(rotations, positions, reference)
    assert torch.allclose(R[0, 0], torch.eye(3), atol=1e-6)
    assert torch.allclose(p[0, 0], torch.zeros(3), atol=1e-6)
    assert torch.allclose(p[0, 1], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(R[0, 1], rotations[0, 1], atol=1e-6)


def test_calibration_guards_still_hold():
    _, _, (K, R, p) = batch(batch_size=1)
    camera = dict(intrinsics=K[0].numpy(), rotations=R[0].numpy(), positions=p[0].numpy())
    report = validate_supervision_camera(camera, 32, 32)
    assert report['invalid_pose_frames'] == 0
    shifted = dict(camera)
    moved = K[0].numpy().copy()
    moved[:, 0, 2] += 25.0
    shifted['intrinsics'] = moved
    with pytest.raises(ValueError):
        validate_supervision_camera(shifted, 32, 32)


def test_fresh_camera_warmup_is_anchored_to_the_phase_not_the_resume_step():
    parameter = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([{'params': [parameter], 'lr': 1e-4, 'name': 'camera_head'}])
    for update in (196001, 196250, 196500, 200000):
        optimizer.param_groups[0]['lr'] = 1e-4
        actual = apply_fresh_group_warmup(optimizer, {'camera_head'}, update, 196000, 500)
        expected = 1e-4 * min((update - 196000) / 500, 1.0)
        assert optimizer.param_groups[0]['lr'] == pytest.approx(expected)
        assert actual == pytest.approx(min((update - 196000) / 500, 1.0))


# --------------------------------------------------------------------------- model wiring

def test_model_attaches_camera_through_the_decoder_and_groups_it_once():
    model = DenseQueryWanModel(TinyBackbone(), tiny_decoder())
    model.train()
    latent = torch.randn(2, 16, 6, 32, 32)
    source_t, target_t, _ = batch()
    condition = torch.zeros(2, 512, 4096)
    prediction, z4d, output = model(latent, source_t, target_t, condition)
    # The native text condition is required: the null-condition and shared-latent
    # override paths were only reachable from the deleted cycle route.
    with pytest.raises(TypeError):
        model(latent, source_t, target_t)
    assert prediction.shape[:2] == (2, 4) and output.camera is not None
    groups = parameter_groups(model, {'learning_rate': 3e-6, 'backbone_learning_rate': 5e-7,
                                      'geometry_learning_rate': 3e-6, 'weight_decay': 0.0,
                                      'camera_supervision': {'learning_rate': 1e-4}})
    names = {group['name']: group['params'] for group in groups}
    assert 'camera_head' in names and 'dense_decoder' in names
    camera_ids = {id(value) for value in model.camera_head_parameters()}
    assert {id(value) for value in names['camera_head']} == camera_ids
    assert not ({id(value) for value in names['dense_decoder']} & camera_ids)
    total = sum(value.numel() for group in groups for value in group['params'])
    assert total == sum(value.numel() for _, value in model.named_parameters())


def test_non_wan_accounting_counts_the_camera_readout_once():
    from worldbridge.trainer.trainer import non_wan_parameter_counts
    model = DenseQueryWanModel(TinyBackbone(), tiny_decoder())
    adapter, decoder, camera = non_wan_parameter_counts(model)
    assert camera == sum(p.numel() for p in model.camera_head_parameters()) > 0
    assert decoder == sum(p.numel() for p in model.decoder.parameters())
    assert adapter == sum(p.numel() for p in model.backbone.adapter_parameters)
    # The camera readout is inside the decoder, so the audited non-Wan total is
    # adapter + decoder; the old code added camera again (the 926218 mismatch).
    assert decoder > camera
    assert adapter + decoder + camera != adapter + decoder


def test_decoder_without_camera_config_has_no_camera_readout():
    decoder = tiny_decoder(camera=False)
    assert not decoder.camera_enabled and decoder.camera_parameters() == ()
    source_t, target_t, _ = batch()
    output = forward(decoder, source_t, target_t)
    assert output.camera is None
