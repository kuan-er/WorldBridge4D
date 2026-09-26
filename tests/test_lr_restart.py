from pathlib import Path

import pytest
import torch
import yaml

from worldbridge.trainer.config import validate_config
from worldbridge.trainer.schedulers import apply_lr_restart_schedule


def setup():
    rates = {"dense_decoder": 3e-6, "source_rgb_decay": 3e-5, "source_rgb_no_decay": 3e-5}
    optimizer = torch.optim.AdamW([
        {"params": [torch.nn.Parameter(torch.ones(1))], "name": name, "lr": 0.0, "_base_lr": 0.003}
        for name in rates
    ])
    return optimizer, {"start_step": 150000, "end_step": 155000, "warmup_steps": 500,
                       "group_learning_rates": rates}


def test_restart_overrides_exhausted_schedule_and_preserves_moments():
    optimizer, phase = setup()
    for group in optimizer.param_groups:
        group["params"][0].grad = torch.ones(1)
    optimizer.step()
    before = {id(p): optimizer.state[p]["exp_avg"].clone() for g in optimizer.param_groups for p in g["params"]}
    assert apply_lr_restart_schedule(optimizer, 150001, phase) == pytest.approx(0.002)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(6e-9)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(6e-8)
    assert apply_lr_restart_schedule(optimizer, 150500, phase) == 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-6)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(3e-5)
    assert apply_lr_restart_schedule(optimizer, 155000, phase) == 1
    for group in optimizer.param_groups:
        for p in group["params"]:
            torch.testing.assert_close(optimizer.state[p]["exp_avg"], before[id(p)])


def test_checkpoint_reload_does_not_restart_warmup():
    optimizer, phase = setup()
    apply_lr_restart_schedule(optimizer, 150250, phase)
    restored, _ = setup()
    restored.load_state_dict(optimizer.state_dict())
    assert apply_lr_restart_schedule(restored, 150251, phase) == pytest.approx(251 / 500)


def test_restart_rejects_wrong_phase_or_optimizer_groups():
    optimizer, phase = setup()
    for step in (150000, 155001):
        with pytest.raises(ValueError, match="outside"):
            apply_lr_restart_schedule(optimizer, step, phase)
    phase["group_learning_rates"] = {"wrong": 1e-6}
    with pytest.raises(ValueError, match="groups"):
        apply_lr_restart_schedule(optimizer, 150001, phase)


def test_supported_profile_lr_restart_guards():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/h033_camera_query_ray_to210000.yaml").read_text())
    validate_config(config, world=2)
    with pytest.raises(ValueError):
        validate_config({**config, "microbatch_per_gpu": 2}, world=2)
    with pytest.raises(ValueError):
        validate_config({**config, "targets_per_source": 13}, world=2)
    rates = dict(config["lr_restart"]["group_learning_rates"])
    rates.pop("camera_head")
    with pytest.raises(ValueError):
        validate_config({**config, "lr_restart": {**config["lr_restart"],
                                                  "group_learning_rates": rates}}, world=2)
    with pytest.raises(ValueError):
        validate_config({**config, "max_steps": 220000}, world=2)

