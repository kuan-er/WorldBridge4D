from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from worldbridge.data.boundaries import source_boundary_band
from worldbridge.data.sampling import sample_eligible_targets
from worldbridge.evaluation.diagnostic_metrics import depth_discontinuity, segmentation_discontinuity, edge_distance
from worldbridge.trainer.batching import load_geometry
from worldbridge.trainer.config import validate_config
from worldbridge.trainer.objective import masked_pair_smooth_l1, boundary_weighted_pair_smooth_l1

ROOT = Path(__file__).resolve().parents[1]
POLICY = {'multiplier': 2.0, 'radius_px': 2, 'depth_relative_jump': 0.05}


@pytest.mark.parametrize('segmented', [False, True])
def test_boundary_band_matches_existing_gt_diagnostic_definition(segmented):
    rng = np.random.default_rng(17)
    depth = rng.uniform(1, 1.01, (20, 24)).astype(np.float32)
    depth[:, 12:] *= 2
    valid = np.ones_like(depth, bool)
    valid[4:6, 2:5] = False
    seg = None
    if segmented:
        seg = np.zeros(depth.shape, np.int64)
        seg[12:, :12] = 1
    edge = depth_discontinuity(depth, valid, 0.05)
    if seg is not None:
        edge |= segmentation_discontinuity(seg)
    expected = edge_distance(edge) <= 2
    before = (depth.copy(), valid.copy())
    actual = source_boundary_band(depth, valid, seg)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(depth, before[0])
    np.testing.assert_array_equal(valid, before[1])
    assert actual.dtype == bool


def test_boundary_has_no_rgb_dependency_and_marks_both_sides():
    depth = np.ones((9, 12), np.float32)
    valid = np.ones_like(depth, bool)
    assert not source_boundary_band(depth, valid).any()
    depth[:, 6:] = 2
    expected = np.zeros_like(valid); expected[:, 5:7] = True
    np.testing.assert_array_equal(source_boundary_band(depth, valid, radius_px=0), expected)
    seg = np.zeros(depth.shape, np.int64); seg[4:] = 1
    assert source_boundary_band(np.ones_like(depth), valid, seg, radius_px=0)[3:5].all()


@pytest.mark.parametrize('kwargs', [{'radius_px': -1}, {'radius_px': 1.5}, {'relative_jump': 0}, {'relative_jump': float('nan')}])
def test_boundary_rejects_invalid_definition(kwargs):
    with pytest.raises(ValueError):
        source_boundary_band(np.ones((3, 4)), np.ones((3, 4), bool), **kwargs)


def fixture():
    torch.manual_seed(91)
    p = torch.randn(2, 3, 3, 4, 5, dtype=torch.float32, requires_grad=True)
    t = torch.randn_like(p)
    v = torch.ones(2, 3, 4, 5, dtype=torch.bool)
    v[0, 1] = False
    v[1, 0, 0] = False
    b = torch.zeros(2, 4, 5, dtype=torch.bool); b[:, :, :2] = True
    return p, t, v, b


def test_multiplier1_exactly_recovers_old_loss_and_gradient():
    p, t, v, b = fixture()
    old = masked_pair_smooth_l1(p, t, v)
    new = boundary_weighted_pair_smooth_l1(p, t, v, b, multiplier=1)
    assert torch.equal(old, new)
    a = torch.autograd.grad(old, p, retain_graph=True)[0]
    c = torch.autograd.grad(new, p)[0]
    assert torch.equal(a, c)


@pytest.mark.parametrize('all_boundary', [False, True])
def test_uniform_weights_reduce_to_old_loss(all_boundary):
    p, t, v, b = fixture(); b.fill_(all_boundary)
    old = masked_pair_smooth_l1(p, t, v)
    new = boundary_weighted_pair_smooth_l1(p, t, v, b)
    torch.testing.assert_close(old, new, rtol=0, atol=0)
    a = torch.autograd.grad(old, p, retain_graph=True)[0]
    c = torch.autograd.grad(new, p)[0]
    torch.testing.assert_close(a, c, rtol=0, atol=0)


def test_weighting_normalizes_within_each_pair_and_retains_valid_occluded_points():
    p = torch.zeros((1, 2, 3, 1, 2), requires_grad=True)
    with torch.no_grad():
        p[:, :, 0] = torch.tensor([2., 4.])
    t = torch.zeros_like(p)
    v = torch.tensor([[[[True, True]], [[True, False]]]])
    b = torch.tensor([[[True, False]]])
    loss = boundary_weighted_pair_smooth_l1(p, t, v, b, beta=1)
    assert loss.item() == pytest.approx(((2 * 1.5 + 3.5) / 3 + 1.5) / 2)
    loss.backward()
    assert p.grad[0, 0, 0, 0, 0] != 0  # Valid point, regardless of its visibility.
    assert p.grad[0, 1, :, 0, 1].eq(0).all()


@pytest.mark.parametrize('failure', ['empty', 'float_boundary', 'bad_shape', 'bad_multiplier'])
def test_weighted_loss_rejects_invalid_contract(failure):
    p, t, v, b = fixture(); multiplier = 2
    if failure == 'empty': v.fill_(False)
    if failure == 'float_boundary': b = b.float()
    if failure == 'bad_shape': b = b[:, :1]
    if failure == 'bad_multiplier': multiplier = float('nan')
    with pytest.raises(ValueError):
        boundary_weighted_pair_smooth_l1(p, t, v, b, multiplier)


class Dataset:
    def __init__(self): self.boundary_calls = []
    def __len__(self): return 3
    def source_all_targets_with_visibility(self, index, source):
        xyz = np.full((21, 3, 5, 7), index + source, np.float32)
        valid = np.ones((21, 5, 7), bool)
        if source == 0: valid[:] = False
        visible = valid.copy(); visible[10:] = False
        return xyz, valid, visible
    def source_rgb(self, index, source): return np.full((5, 7, 3), index + source, np.uint8)
    def cycle_camera(self, index): return {'positions': np.zeros((21, 3), np.float32)}
    def source_boundary_context(self, index, source):
        self.boundary_calls.append((index, source))
        depth = np.ones((5, 7), np.float32); depth[:, 4:] = 2
        return depth, np.ones_like(depth, bool), None


def test_geometry_boundary_path_preserves_selected_clip_source_arrays_and_rng():
    ds = Dataset()
    a = load_geometry(ds, 1, np.arange(21), 15, 8, True, True)[0]
    b = load_geometry(ds, 1, np.arange(21), 15, 8, True, True, POLICY)[0]
    assert a[:2] == b[:2] == (1, 1)
    for i in range(2, 6): np.testing.assert_array_equal(a[i], b[i])
    np.testing.assert_array_equal(a[6]['positions'], b[6]['positions'])
    assert a[7] is None and b[7].dtype == bool and ds.boundary_calls == [(1, 1)]
    rng = np.random.default_rng(38)
    np.testing.assert_array_equal(sample_eligible_targets(a[3], 15, deepcopy(rng)),
                                  sample_eligible_targets(b[3], 15, deepcopy(rng)))


def test_boundary_data_failure_must_not_choose_another_clip():
    ds = Dataset()
    def fail(index, source):
        ds.boundary_calls.append((index, source))
        raise ValueError('bad boundary GT')
    ds.source_boundary_context = fail
    with pytest.raises(ValueError, match='bad boundary GT'):
        load_geometry(ds, 1, np.arange(21), 15, 8, True, True, POLICY)
    assert ds.boundary_calls == [(1, 1)]


def test_boundary_config_changes_only_selected_objective_and_terminal_records():
    base = yaml.safe_load((ROOT / 'configs/h030_150k_to_152500_gpu23_b2_k15_fp32_cycle0_rgb1x.yaml').read_text())
    cfg = yaml.safe_load((ROOT / 'configs/h030_150k_to_155k_gpu23_b2_k15_fp32_cycle0_rgb1x_boundary2x.yaml').read_text())
    validate_config(cfg, world=2)
    assert {k for k in base.keys() | cfg.keys() if base.get(k) != cfg.get(k)} == {'boundary_supervision','max_steps','lr_restart','checkpoint_steps','tracking'}
    assert cfg['boundary_supervision'] == POLICY
    assert cfg['lr_restart'] == {**base['lr_restart'], 'end_step':155000}
    assert cfg['selected_checkpoint_step'] == 150000
    assert cfg['max_steps'] == 155000 and cfg['cuda_empty_cache_every_steps'] == 0
    assert cfg['cycle_reprojection_enabled'] and cfg['cycle_reprojection_weight'] == 0
    for change in [{'multiplier':4}, {'radius_px':3}, {'depth_relative_jump':0.1}]:
        bad = deepcopy(cfg); bad['boundary_supervision'].update(change)
        with pytest.raises(ValueError, match='boundary control'):
            validate_config(bad, world=2)
    bad = deepcopy(cfg); bad['cycle_reprojection_weight'] = 0.3
    with pytest.raises(ValueError, match='boundary control'):
        validate_config(bad, world=2)
