from pathlib import Path

import pytest
import torch
import yaml

from worldbridge.trainer.config import validate_config
from worldbridge.trainer.objective import boundary_weighted_pair_smooth_l1, source_edge_contrast_loss

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/h030_150k_to_155k_gpu23_b2_k15_boundary2x_edgecontrast0p01.yaml'


def test_dose_only_delta_preserves_origin_budget_sampling_and_architecture():
    base = yaml.safe_load((ROOT / 'configs/h030_150k_to_155k_gpu23_b2_k15_boundary2x_edgecontrast01.yaml').read_text())
    cfg = yaml.safe_load(CONFIG.read_text())
    assert {k for k in base.keys() | cfg.keys() if base.get(k) != cfg.get(k)} == {'source_edge_contrast_weight', 'tracking'}
    assert base['source_edge_contrast_weight'] == .1 and cfg['source_edge_contrast_weight'] == .01
    assert cfg['lr_restart']['start_step'] == cfg['selected_checkpoint_step'] == 150000
    assert cfg['max_steps'] == 155000 and cfg['lr_restart']['warmup_steps'] == 500
    assert cfg['lr_restart']['group_learning_rates'] == dict.fromkeys(
        ('dense_decoder', 'source_rgb_decay', 'source_rgb_no_decay'), 3e-6)
    assert cfg['boundary_supervision'] == {'multiplier': 2.0, 'radius_px': 2, 'depth_relative_jump': .05}
    assert cfg['cuda_empty_cache_every_steps'] == 0
    assert cfg['cycle_reprojection_enabled'] and cfg['cycle_reprojection_weight'] == 0


@pytest.mark.parametrize('weight', [0.0, 0.01, 0.1])
def test_only_selected_doses_admitted(weight):
    cfg = yaml.safe_load(CONFIG.read_text()); cfg['source_edge_contrast_weight'] = weight
    validate_config(cfg, world=2)


@pytest.mark.parametrize('weight', [-1, 0.001, 0.02, 0.2, float('inf'), float('nan')])
def test_unaudited_doses_rejected(weight):
    cfg = yaml.safe_load(CONFIG.read_text()); cfg['source_edge_contrast_weight'] = weight
    with pytest.raises(ValueError, match='contrast weight'): validate_config(cfg, world=2)


def test_tenfold_auxiliary_gradient_reduction_not_a_primary_loss_or_mask_change():
    torch.manual_seed(1729)
    p = torch.randn(2, 3, 3, 4, 5, dtype=torch.float64, requires_grad=True)
    t = torch.randn_like(p)
    valid = torch.rand(2, 3, 4, 5) > .2
    band = torch.rand(2, 4, 5) > .5
    edges = torch.ones(2, 2, 4, 5, dtype=torch.bool)
    edges[:, 0, -1] = False; edges[:, 1, :, -1] = False
    primary = boundary_weighted_pair_smooth_l1(p, t, valid, band)
    aux, count, pairs = source_edge_contrast_loss(p, t, valid, edges)
    assert count > 0 and pairs == 6
    g0 = torch.autograd.grad(primary, p, retain_graph=True)[0]
    g1 = torch.autograd.grad(primary + .1 * aux, p, retain_graph=True)[0]
    g01 = torch.autograd.grad(primary + .01 * aux, p)[0]
    torch.testing.assert_close(g01 - g0, .1 * (g1 - g0), rtol=1e-10, atol=1e-15)
    assert torch.isfinite(g01).all()
