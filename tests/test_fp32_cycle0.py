from pathlib import Path

import torch
import yaml

from worldbridge.trainer.config import validate_config
from worldbridge.trainer.objective import loss_scale_to_reference


def test_fp32_cycle0_changes_only_aux_gradient_and_terminal_budget():
    root = Path(__file__).resolve().parents[1]
    baseline = yaml.safe_load((root / 'configs/h030_150k_to_155k_gpu23_b2_k15_fp32_master.yaml').read_text())
    control = yaml.safe_load((root / 'configs/h030_150k_to_152500_gpu23_b2_k15_fp32_cycle0.yaml').read_text())
    validate_config(control, world=2)
    assert {k for k in baseline.keys() | control.keys() if baseline.get(k) != control.get(k)} == {
        'cycle_reprojection_weight', 'max_steps', 'lr_restart', 'checkpoint_steps', 'tracking',
    }
    assert control['cycle_reprojection_enabled'] is True
    assert control['cycle_b2_a2_k15'] is True
    assert control['cycle_reprojection_weight'] == 0.0
    assert control['fsdp_master_precision'] == 'fp32' and control['precision'] == 'bf16'
    assert control['lr_restart'] == {**baseline['lr_restart'], 'end_step': 152500}
    assert control['max_steps'] - control['selected_checkpoint_step'] == 2500
    assert control['checkpoint_steps'] == [s for s in baseline['checkpoint_steps'] if s <= 152500]


def test_zero_cycle_coefficient_preserves_exact_xyz_gradient():
    p = torch.tensor([0.2, 0.8], requires_grad=True)
    q = torch.tensor([-0.5], requires_grad=True)
    xyz = p.square().mean()
    cycle = (p + q).square().mean() + 0.1
    scale = loss_scale_to_reference(xyz, cycle)
    combined = xyz + 0.0 * scale * cycle
    grad_p, grad_q = torch.autograd.grad(combined, (p, q), retain_graph=True)
    (expected,) = torch.autograd.grad(xyz, (p,))
    assert torch.equal(combined, xyz)
    assert torch.equal(grad_p, expected)
    assert torch.count_nonzero(grad_q) == 0
    assert not scale.requires_grad
