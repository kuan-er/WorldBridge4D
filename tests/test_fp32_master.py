from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from worldbridge.trainer.config import validate_config
from worldbridge.trainer.precision import assert_fp32_optimizer_storage, prepare_fsdp_master_parameters

ROOT = Path(__file__).resolve().parents[1]


def test_master_conversion_preserves_values_freeze_and_bf16_compute():
    torch.manual_seed(424242)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2)).bfloat16()
    model[0].requires_grad_(False)
    before = {n: p.clone() for n, p in model.named_parameters()}
    flags = {n: p.requires_grad for n, p in model.named_parameters()}
    prepare_fsdp_master_parameters(model, "fp32")
    for n, p in model.named_parameters():
        assert p.dtype == torch.float32 and torch.equal(p, before[n].float())
        assert p.requires_grad == flags[n]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert model(torch.ones(1, 4)).dtype == torch.bfloat16


def test_default_master_path_is_unchanged_and_bad_mode_rejected():
    model = torch.nn.Linear(2, 2).bfloat16()
    prepare_fsdp_master_parameters(model, "model")
    assert model.weight.dtype == torch.bfloat16
    with pytest.raises(ValueError):
        prepare_fsdp_master_parameters(model, "fp16")


def test_fp32_checkpoint_values_survive_prepared_model_resume():
    model = torch.nn.Linear(2, 2).bfloat16()
    prepare_fsdp_master_parameters(model, "fp32")
    state = {name: torch.full_like(value, 0.020001) for name, value in model.state_dict().items()}
    assert not torch.equal(state["weight"], state["weight"].bfloat16().float())
    model.load_state_dict(state)
    for name, value in model.state_dict().items():
        assert torch.equal(value, state[name])


def test_adam_resume_casts_moments_without_resetting_values_or_counters():
    old = torch.nn.Parameter(torch.tensor([0.02], dtype=torch.bfloat16))
    old_optimizer = torch.optim.AdamW([old], lr=3e-6)
    old.grad = torch.full_like(old, 0.001)
    old_optimizer.step()
    checkpoint = deepcopy(old_optimizer.state_dict())
    new = torch.nn.Parameter(old.detach().float())
    optimizer = torch.optim.AdamW([new], lr=3e-6)
    optimizer.load_state_dict(checkpoint)
    assert_fp32_optimizer_storage(optimizer)
    old_state, new_state = old_optimizer.state[old], optimizer.state[new]
    for key in ("exp_avg", "exp_avg_sq"):
        assert new_state[key].dtype == torch.float32
        assert torch.equal(new_state[key], old_state[key].float())
    assert torch.equal(new_state["step"], old_state["step"])
    with pytest.raises(RuntimeError):
        assert_fp32_optimizer_storage(old_optimizer)


def test_small_adam_updates_round_away_in_bf16_but_accumulate_in_fp32():
    initial = torch.tensor([0.02], dtype=torch.bfloat16)
    bf = torch.nn.Parameter(initial.clone())
    fp = torch.nn.Parameter(initial.float())
    optimizers = [torch.optim.AdamW([p], lr=3e-6, weight_decay=0) for p in (bf, fp)]
    for _ in range(100):
        for p, optimizer in zip((bf, fp), optimizers):
            p.grad = torch.full_like(p, 0.001)
            optimizer.step()
    assert torch.equal(bf, initial)
    assert not torch.equal(fp, initial.float())
    assert float(initial.float() - fp) > 0.0002


def test_precision_control_is_fp32_master_with_bf16_compute():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/h033_camera_query_ray_to210000.yaml").read_text())
    validate_config(config, world=2)
    assert config["fsdp_master_precision"] == "fp32"
    assert config["precision"] == "bf16"
    for override in ({"fsdp_master_precision": "fp16"}, {"precision": "fp32"},
                     {"fsdp_master_precision": "model"}):
        with pytest.raises(ValueError):
            validate_config({**config, **override}, world=2)

