from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from worldbridge.data.boundaries import source_contrast_edges
from worldbridge.data.sampling import sample_eligible_targets
from worldbridge.trainer.batching import load_geometry
from worldbridge.trainer.config import validate_config
from worldbridge.trainer.objective import boundary_weighted_pair_smooth_l1, source_edge_contrast_loss
from test_boundary_supervision import Dataset, POLICY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/h030_150k_to_155k_gpu23_b2_k15_boundary2x_edgecontrast01.yaml'


def test_contrast_edges_oriented_depth_and_dense_instance_union_not_band():
    depth = np.ones((5, 7)); depth[:, 4:] = 2
    valid = np.ones_like(depth, bool)
    seg = np.zeros(depth.shape, np.int64); seg[3:] = 1
    edges = source_contrast_edges(depth, valid, valid, seg)
    expected = np.zeros((2, 5, 7), bool)
    expected[0, 2] = True; expected[1, :, 3] = True
    np.testing.assert_array_equal(edges, expected)
    depth_only = source_contrast_edges(depth, valid, valid)
    assert not depth_only[0].any() and depth_only[1, :, 3].all()
    # Edges connect actual differing neighbors, not tangential neighbors in a band.
    assert not edges[1, :, 2].any() and not edges[1, :, 4].any()


@pytest.mark.parametrize('bad', ['source_invalid', 'depth_invalid', 'nan', 'negative'])
def test_contrast_needs_two_known_source_endpoints(bad):
    depth = np.array([[1., 2., 3.]])
    dv = np.ones((1, 3), bool); sv = dv.copy()
    if bad == 'source_invalid': sv[0, 1] = False
    if bad == 'depth_invalid': dv[0, 1] = False
    if bad == 'nan': depth[0, 1] = np.nan
    if bad == 'negative': depth[0, 1] = -1
    assert not source_contrast_edges(depth, dv, sv).any()


@pytest.mark.parametrize('case', ['shape', 'seg_float', 'seg_shape', 'jump_nan', 'jump_zero'])
def test_contrast_mask_rejects_invalid_contract(case):
    d = np.ones((4, 5)); v = np.ones_like(d, bool); sv = v; seg = None; jump = .05
    if case == 'shape': sv = v[:1]
    if case == 'seg_float': seg = d
    if case == 'seg_shape': seg = np.zeros((1, 5), np.int64)
    if case == 'jump_nan': jump = float('nan')
    if case == 'jump_zero': jump = 0
    with pytest.raises(ValueError):
        source_contrast_edges(d, v, sv, seg, relative_jump=jump)


def test_contrast_loss_pair_normalization_empty_pairs_and_valid_occluded_targets():
    p = torch.zeros(1, 3, 3, 1, 3, requires_grad=True)
    with torch.no_grad():
        p[0, 0, 0, 0] = torch.tensor([0., 2., 99.])
        p[0, 1, 0, 0] = torch.tensor([0., 4., 8.])
    t = torch.zeros_like(p)
    v = torch.tensor([[[[True, True, False]], [[True, True, True]], [[False, False, False]]]])
    e = torch.zeros(1, 2, 1, 3, dtype=torch.bool); e[:, 1, :, :2] = True
    # Target1 can be entirely occluded: there is deliberately no visibility argument.
    loss, count, pairs = source_edge_contrast_loss(p, t, v, e, beta=1)
    assert loss.item() == pytest.approx((1.5 + 3.5) / 2)
    assert (int(count), int(pairs)) == (3, 2)
    loss.backward()
    assert p.grad[0, 1].abs().sum() > 0
    assert p.grad[0, 0, :, :, 2].eq(0).all() and p.grad[0, 2].eq(0).all()


def sample():
    torch.manual_seed(537)
    p = torch.randn(2, 3, 3, 4, 5, dtype=torch.float64, requires_grad=True)
    t = torch.randn_like(p)
    v = torch.rand(2, 3, 4, 5) > .2
    e = torch.rand(2, 2, 4, 5) > .6
    e[:, 0, -1] = False; e[:, 1, :, -1] = False
    return p, t, v, e


def test_contrast_matches_independent_edge_enumeration_loss_and_gradient():
    p, t, v, e = sample()
    losses = []; expected_count = 0
    for b in range(p.shape[0]):
        for k in range(p.shape[1]):
            errors = []
            for direction, dy, dx in [(0, 1, 0), (1, 0, 1)]:
                for y in range(p.shape[-2] - dy):
                    for x in range(p.shape[-1] - dx):
                        if e[b, direction, y, x] and v[b, k, y, x] and v[b, k, y+dy, x+dx]:
                            errors.append(F.smooth_l1_loss(
                                p[b, k, :, y+dy, x+dx] - p[b, k, :, y, x],
                                t[b, k, :, y+dy, x+dx] - t[b, k, :, y, x],
                                beta=.05, reduction='sum'))
            expected_count += len(errors)
            if errors: losses.append(torch.stack(errors).mean())
    expected = torch.stack(losses).mean()
    actual, count, pairs = source_edge_contrast_loss(p, t, v, e)
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
    assert int(count) == expected_count and int(pairs) == len(losses)
    torch.testing.assert_close(torch.autograd.grad(actual, p, retain_graph=True)[0],
                               torch.autograd.grad(expected, p)[0], rtol=1e-14, atol=1e-14)


def test_contrast_translation_invariance_and_zero_weight_gradient_parity():
    p, t, v, e = sample()
    value = source_edge_contrast_loss(p, t, v, e)[0]
    shifted = source_edge_contrast_loss(p + torch.randn(2, 3, 3, 1, 1), t, v, e)[0]
    torch.testing.assert_close(value, shifted, rtol=1e-14, atol=1e-14)
    boundary = torch.ones(2, 4, 5, dtype=torch.bool)
    base = boundary_weighted_pair_smooth_l1(p, t, v, boundary)
    total = base + 0.0 * value
    assert torch.equal(base, total)
    assert torch.equal(torch.autograd.grad(base, p, retain_graph=True)[0],
                       torch.autograd.grad(total, p)[0])


def test_contrast_gradient_restores_collapsed_surface_without_translation_force():
    t = torch.zeros(1, 1, 3, 1, 2, dtype=torch.float64); t[0, 0, 0, 0, 1] = 1
    p = t.clone(); p[0, 0, 0, 0] = torch.tensor([.25, .75]); p.requires_grad_()
    v = torch.ones(1, 1, 1, 2, dtype=torch.bool)
    e = torch.zeros(1, 2, 1, 2, dtype=torch.bool); e[0, 1, 0, 0] = True
    loss = source_edge_contrast_loss(p, t, v, e)[0]; loss.backward()
    assert p.grad[0, 0, 0, 0, 0] > 0 and p.grad[0, 0, 0, 0, 1] < 0
    assert p.grad.sum().item() == 0
    assert source_edge_contrast_loss(t + 2, t, v, e)[0].item() == 0


@pytest.mark.parametrize('empty', ['edges', 'targets'])
def test_empty_contrast_is_differentiable_zero_not_clip_fallback(empty):
    p, t, v, e = sample()
    if empty == 'edges': e.fill_(False)
    else: v.fill_(False)
    loss, count, pairs = source_edge_contrast_loss(p, t, v, e)
    assert loss.item() == 0 and count == 0 and pairs == 0
    loss.backward(); assert p.grad.eq(0).all()


@pytest.mark.parametrize('bad', ['target', 'valid_shape', 'valid_type', 'edge_shape', 'edge_type'])
def test_contrast_loss_rejects_invalid_contract(bad):
    p, t, v, e = sample()
    if bad == 'target': t = t[..., :1]
    if bad == 'valid_shape': v = v[..., :1]
    if bad == 'valid_type': v = v.float()
    if bad == 'edge_shape': e = e[:, :1]
    if bad == 'edge_type': e = e.float()
    with pytest.raises(ValueError): source_edge_contrast_loss(p, t, v, e)


def test_edge_path_preserves_boundary_control_selection_arrays_and_target_rng():
    ds = Dataset()
    a = load_geometry(ds, 1, np.arange(21), 15, 8, True, True, POLICY)[0]
    b = load_geometry(ds, 1, np.arange(21), 15, 8, True, True, POLICY, True)[0]
    assert a[:2] == b[:2] == (1, 1)
    for i in [2, 3, 4, 5, 7]: np.testing.assert_array_equal(a[i], b[i])
    np.testing.assert_array_equal(a[6]['positions'], b[6]['positions'])
    assert a[8] is None and b[8].shape == (2, 5, 7) and b[8].dtype == bool
    rng_a = np.random.default_rng(38); rng_b = deepcopy(rng_a)
    np.testing.assert_array_equal(sample_eligible_targets(a[3], 15, rng_a),
                                  sample_eligible_targets(b[3], 15, rng_b))
    assert rng_a.bit_generator.state == rng_b.bit_generator.state
    assert ds.boundary_calls == [(1, 1), (1, 1)]


def test_edge_data_failure_cannot_select_another_clip(monkeypatch):
    ds = Dataset()
    def fail(*args, **kwargs): raise ValueError('bad contrast GT')
    monkeypatch.setattr('worldbridge.trainer.batching.source_contrast_edges', fail)
    with pytest.raises(ValueError, match='bad contrast GT'):
        load_geometry(ds, 1, np.arange(21), 15, 8, True, True, POLICY, True)
    assert ds.boundary_calls == [(1, 1)]


def test_edge_config_changes_one_scientific_factor_from_existing_boundary_arm():
    base = yaml.safe_load((ROOT / 'configs/h030_150k_to_155k_gpu23_b2_k15_fp32_cycle0_rgb1x_boundary2x.yaml').read_text())
    cfg = yaml.safe_load(CONFIG.read_text()); validate_config(cfg, world=2)
    assert {k for k in base.keys() | cfg.keys() if base.get(k) != cfg.get(k)} == {'source_edge_contrast_weight', 'tracking'}
    assert cfg['source_edge_contrast_weight'] == .1 and cfg['boundary_supervision'] == POLICY
    assert cfg['selected_checkpoint_step'] == cfg['lr_restart']['start_step'] == 150000
    assert cfg['lr_restart']['warmup_steps'] == 500 and cfg['max_steps'] == 155000
    assert set(cfg['lr_restart']['group_learning_rates'].values()) == {3e-6}
    assert cfg['cuda_empty_cache_every_steps'] == 0 and cfg['cycle_reprojection_weight'] == 0
    for weight in [-1, .2, float('nan')]:
        bad = deepcopy(cfg); bad['source_edge_contrast_weight'] = weight
        with pytest.raises(ValueError, match='contrast weight'): validate_config(bad, world=2)
    for boundary in [None, {**POLICY, 'multiplier':1}]:
        bad = deepcopy(cfg); bad['boundary_supervision'] = boundary
        with pytest.raises(ValueError, match='requires boundary2x'): validate_config(bad, world=2)
