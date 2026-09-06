from pathlib import Path

import pytest
import torch
import yaml

from worldbridge.trainer.config import validate_config
from worldbridge.trainer.schedulers import apply_lr_restart_schedule


def test_native_reuse_10k_changes_only_requested_phase_controls():
    root = Path(__file__).resolve().parents[1]
    old = yaml.safe_load((root / "configs/h030_150k_lrrestart_gpu23_b2_k19.yaml").read_text())
    new = yaml.safe_load((root / "configs/h030_150k_to_160k_gpu23_b2_k19_native_reuse.yaml").read_text())
    validate_config(new, world=2)
    assert {k for k in set(old) | set(new) if old.get(k) != new.get(k)} == {
        "max_steps", "lr_restart", "cuda_empty_cache_every_steps", "checkpoint_steps", "tracking",
    }
    assert new["cuda_empty_cache_every_steps"] == 0
    assert new["lr_restart"]["start_step"] == new["selected_checkpoint_step"] == 150000
    assert new["lr_restart"]["end_step"] == new["max_steps"] == 160000
    assert new["max_steps"] - new["selected_checkpoint_step"] == 10000
    assert {**old["lr_restart"], "end_step": 160000} == new["lr_restart"]
    rates = new["lr_restart"]["group_learning_rates"]
    optimizer = torch.optim.AdamW([
        {"params": [torch.nn.Parameter(torch.ones(1))], "name": name, "lr": 0.0}
        for name in rates
    ])
    assert apply_lr_restart_schedule(optimizer, 150001, new["lr_restart"]) == pytest.approx(1 / 500)
    assert apply_lr_restart_schedule(optimizer, 160000, new["lr_restart"]) == 1
