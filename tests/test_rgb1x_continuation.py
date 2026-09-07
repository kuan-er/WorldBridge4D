from pathlib import Path

import pytest
import torch
import yaml

from worldbridge.trainer.config import validate_config
from worldbridge.trainer.schedulers import apply_lr_restart_schedule
from worldbridge.evaluation.diagnostics import diagnostic_device

ROOT = Path(__file__).resolve().parents[1]


def test_rgb1x_continuation_retains_own_state_policy_and_no_rewarmup():
    original = yaml.safe_load((ROOT / 'configs/h030_150k_to_152500_gpu23_b2_k15_fp32_cycle0_rgb1x.yaml').read_text())
    continued = yaml.safe_load((ROOT / 'configs/h030_152500_to_155k_gpu23_b2_k15_fp32_cycle0_rgb1x.yaml').read_text())
    validate_config(continued, world=2)
    assert {k for k in original if original[k] != continued[k]} == {'max_steps', 'lr_restart', 'checkpoint_steps', 'tracking'}
    assert {k for k in original['lr_restart'] if original['lr_restart'][k] != continued['lr_restart'][k]} == {'end_step'}
    assert continued['max_steps'] == continued['lr_restart']['end_step'] == 155000
    assert continued['lr_restart']['start_step'] == 150000
    assert continued['lr_restart']['warmup_steps'] == 500
    assert set(continued['lr_restart']['group_learning_rates'].values()) == {3e-6}
    assert continued['checkpoint_steps'] == [152501, 152600, 153000, 153500, 154500, 155000]
    assert continued['cuda_empty_cache_every_steps'] == 0
    for step in (152501, 152600, 153500, 155000):
        params = [torch.nn.Parameter(torch.tensor([0.125001])) for _ in range(3)]
        opt = torch.optim.AdamW([{'params': [p], 'name': name, '_base_lr': 0.003} for p, name in zip(params, continued['lr_restart']['group_learning_rates'])], lr=0.003)
        for p in params:
            p.grad = torch.ones_like(p)
        opt.step()
        before = [(opt.state[p]['step'].clone(), opt.state[p]['exp_avg'].clone()) for p in params]
        factor = apply_lr_restart_schedule(opt, step, continued['lr_restart'])
        assert factor == 1.0
        assert all(group['lr'] == group['_base_lr'] == 3e-6 for group in opt.param_groups)
        for p, (age, moment) in zip(params, before):
            torch.testing.assert_close(opt.state[p]['step'], age)
            torch.testing.assert_close(opt.state[p]['exp_avg'], moment)


def test_diagnostic_device_selects_and_initializes_only_visible_device(monkeypatch):
    selected = []
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda device: selected.append(str(device)))
    assert str(diagnostic_device()) == 'cuda:0'
    assert selected == ['cuda:0']
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    with pytest.raises(RuntimeError, match='exactly one'):
        diagnostic_device()
    assert selected == ['cuda:0']
