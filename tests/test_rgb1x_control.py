from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from worldbridge.trainer.config import validate_config
from worldbridge.trainer.schedulers import apply_lr_restart_schedule

ROOT = Path(__file__).resolve().parents[1]


def configs():
    return [yaml.safe_load((ROOT / 'configs' / name).read_text()) for name in (
        'h030_150k_to_152500_gpu23_b2_k15_fp32_cycle0.yaml',
        'h030_150k_to_152500_gpu23_b2_k15_fp32_cycle0_rgb1x.yaml',
    )]


def test_rgb1x_changes_only_actual_rgb_rates_and_tracking():
    base, control = configs()
    validate_config(control, world=2)
    assert {k for k in base.keys() | control.keys() if base.get(k) != control.get(k)} == {'lr_restart', 'tracking'}
    assert control['lr_restart'] == {
        **base['lr_restart'],
        'group_learning_rates': {'dense_decoder': 3e-6, 'source_rgb_decay': 3e-6, 'source_rgb_no_decay': 3e-6},
    }
    # This retained production-init field is NOT the runtime rate policy.
    assert control['source_rgb_learning_rate_multiplier'] == 10.0
    assert control['selected_checkpoint_step'] == 150000 and control['max_steps'] == 152500
    assert control['fsdp_master_precision'] == 'fp32' and control['cycle_reprojection_weight'] == 0.0


def test_absolute_rates_override_legacy_metadata_without_resetting_adam():
    base, control = configs()
    names = list(control['lr_restart']['group_learning_rates'])
    parameters = [torch.nn.Parameter(torch.tensor([0.020001])) for _ in names]
    optimizer = torch.optim.AdamW([
        {'name': name, 'params': [p], 'lr': 0.0, '_base_lr': 0.003}
        for name, p in zip(names, parameters)
    ], lr=0.0)
    for p in parameters:
        p.grad = torch.tensor([0.01])
    optimizer.step()
    optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    state = {p: deepcopy(optimizer.state[p]) for p in parameters}
    for step in (150001, 150100, 150500, 151500, 152500):
        factor = apply_lr_restart_schedule(optimizer, step, control['lr_restart'])
        assert factor == min(1.0, (step - 150000) / 500)
        for group in optimizer.param_groups:
            assert group['lr'] == pytest.approx(3e-6 * factor)
            assert group['_base_lr'] == 3e-6
            original_rate = base['lr_restart']['group_learning_rates'][group['name']] * factor
            assert group['lr'] == pytest.approx(original_rate * (1 if group['name'] == 'dense_decoder' else 0.1))
        for p in parameters:
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                assert torch.equal(optimizer.state[p][key], state[p][key])
    with pytest.raises(ValueError):
        apply_lr_restart_schedule(optimizer, 152501, control['lr_restart'])
