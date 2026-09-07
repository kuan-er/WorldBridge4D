from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch
import yaml

from worldbridge.trainer.config import validate_config
from worldbridge.trainer.schedulers import apply_lr_restart_schedule

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("handoff", ROOT / "research/analysis/h030_preserve_fp32_cycle0_checkpoint.py")
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


def test_continuation_changes_only_terminal_budget_and_records():
    base = yaml.safe_load((ROOT / "configs/h030_150k_to_152500_gpu23_b2_k15_fp32_cycle0.yaml").read_text())
    new = yaml.safe_load((ROOT / "configs/h030_152500_to_155k_gpu23_b2_k15_fp32_cycle0.yaml").read_text())
    validate_config(new, world=2)
    assert {k for k in base.keys() | new.keys() if base.get(k) != new.get(k)} == {
        "max_steps", "lr_restart", "checkpoint_steps", "tracking",
    }
    assert new["max_steps"] == 155000
    assert new["lr_restart"] == {**base["lr_restart"], "end_step": 155000}
    assert new["checkpoint_steps"] == [152501, 152510, 152600, 153000, 153500, 154500, 155000]
    parameters = [torch.nn.Parameter(torch.tensor([0.020001])) for _ in handoff.RATES]
    opt = torch.optim.AdamW([{"params": [p], "name": name} for name, p in zip(handoff.RATES, parameters)], lr=3e-6)
    assert apply_lr_restart_schedule(opt, 152500, base["lr_restart"]) == 1.0
    saved = deepcopy(opt.state_dict())
    opt.load_state_dict(saved)
    for step in (152501, 152600, 154999, 155000):
        assert apply_lr_restart_schedule(opt, step, new["lr_restart"]) == 1.0
        assert {g["name"]: g["lr"] for g in opt.param_groups} == handoff.RATES
    with pytest.raises(ValueError):
        apply_lr_restart_schedule(opt, 155001, new["lr_restart"])


def test_native_fp32_adam_resume_matches_uninterrupted_next_update():
    p = torch.nn.Parameter(torch.tensor([0.020001]))
    opt = torch.optim.AdamW([p], lr=3e-6)
    for _ in range(3):
        p.grad = torch.tensor([0.001])
        opt.step()
    saved = deepcopy(opt.state_dict())
    q = torch.nn.Parameter(p.detach().clone())
    resumed = torch.optim.AdamW([q], lr=3e-6)
    resumed.load_state_dict(saved)
    assert not torch.equal(q, q.bfloat16().float())
    for a, b in (("exp_avg", "exp_avg"), ("exp_avg_sq", "exp_avg_sq"), ("step", "step")):
        assert torch.equal(opt.state[p][a], resumed.state[q][b])
    p.grad = torch.tensor([-0.002])
    q.grad = p.grad.clone()
    opt.step(); resumed.step()
    assert torch.equal(p, q)
    for key in ("exp_avg", "exp_avg_sq", "step"):
        assert torch.equal(opt.state[p][key], resumed.state[q][key])


def fixture_payload():
    step = 152500
    status = {"completed_steps": step, "world_size": 2, "clips_seen": {"kubric": 427000, "pointodyssey": 366000, "dynamic_replica": 427000}}
    payload = {"format": 3, "training_state": {"global_step": step, "world_size": 2, "rng_states": [{}, {}], "clips_seen": status["clips_seen"].copy()},
               "config": {"precision": "bf16", "fsdp_master_precision": "fp32", "cycle_reprojection_enabled": True, "cycle_reprojection_weight": 0.0},
               "model": {}, "optimizer": {"param_groups": [], "state": {}}}
    for index, (name, lr) in enumerate(handoff.RATES.items()):
        payload["model"][name] = torch.tensor([0.020001], dtype=torch.float32)
        payload["optimizer"]["param_groups"].append({"name": name, "lr": lr, "params": [name]})
        payload["optimizer"]["state"][name] = {"exp_avg": torch.tensor([0.01]), "exp_avg_sq": torch.tensor([0.001]), "step": torch.tensor(67500 if index else 52500)}
    return payload, status


def save_fixture(tmp_path, payload, status):
    source = tmp_path / "source"
    source.mkdir()
    torch.save(payload, source / "checkpoint-0152500.pt")
    (source / "train_status.json").write_text(json.dumps(status))
    return source


def test_preserve_full_state_is_idempotent_same_inode_and_values(tmp_path):
    payload, status = fixture_payload()
    source = save_fixture(tmp_path, payload, status)
    destination = tmp_path / "protected"
    result = handoff.preserve(source, destination, 152500)
    assert result == handoff.preserve(source, destination, 152500)
    assert os.path.samefile(source / "checkpoint-0152500.pt", destination / "checkpoint-0152500.pt")
    assert json.loads((destination / "train_status.json").read_text()) == status
    restored = torch.load(destination / "checkpoint-0152500.pt", weights_only=False)
    for name in handoff.RATES:
        assert torch.equal(restored["model"][name], payload["model"][name])
        for key in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(restored["optimizer"]["state"][name][key], payload["optimizer"]["state"][name][key])


@pytest.mark.parametrize("fault", ["checkpoint_step", "sidecar_step", "world", "rng", "counters", "master", "moment", "cycle", "lr", "age"])
def test_handoff_rejects_wrong_or_incomplete_state(tmp_path, fault):
    payload, status = fixture_payload()
    if fault == "checkpoint_step": payload["training_state"]["global_step"] -= 1
    elif fault == "sidecar_step": status["completed_steps"] -= 1
    elif fault == "world": payload["training_state"]["world_size"] = 1
    elif fault == "rng": payload["training_state"]["rng_states"] = [{}]
    elif fault == "counters": status["clips_seen"]["kubric"] -= 1
    elif fault == "master": payload["model"]["dense_decoder"] = payload["model"]["dense_decoder"].bfloat16()
    elif fault == "moment": payload["optimizer"]["state"]["dense_decoder"]["exp_avg"] = torch.zeros(1, dtype=torch.bfloat16)
    elif fault == "cycle": payload["config"]["cycle_reprojection_weight"] = 0.3
    elif fault == "lr": payload["optimizer"]["param_groups"][0]["lr"] = 0.0
    elif fault == "age": payload["optimizer"]["state"]["dense_decoder"]["step"] = torch.tensor(0)
    source = save_fixture(tmp_path, payload, status)
    with pytest.raises(ValueError):
        handoff.preserve(source, tmp_path / "protected", 152500)
    assert not (tmp_path / "protected").exists()


def advanced_status(status):
    return {**status, "completed_steps": status["completed_steps"] + 500,
            "clips_seen": {k: v + (1200 if k == "pointodyssey" else 1400)
                           for k, v in status["clips_seen"].items()}}


def test_explicit_historical_sidecar_restore_does_not_modify_live_producer(tmp_path):
    payload, observed = fixture_payload()
    live = advanced_status(observed)
    source = save_fixture(tmp_path, payload, live)
    source_bytes = (source / "train_status.json").read_bytes()
    destination = tmp_path / "protected"
    with pytest.raises(ValueError, match="handoff step mismatch"):
        handoff.preserve(source, destination, 152500)
    assert not destination.exists()
    report = handoff.preserve(source, destination, 152500, restore_planning_status=observed)
    assert (source / "train_status.json").read_bytes() == source_bytes
    assert json.loads((destination / "train_status.json").read_bytes()) == observed
    assert os.path.samefile(source / "checkpoint-0152500.pt", destination / "checkpoint-0152500.pt")
    assert report["live_sidecar_step_at_read"] == 153000
    assert report["live_sidecar_sha256_at_read"] == handoff.hashlib.sha256(source_bytes).hexdigest()
    assert "not_copied_live_sidecar" in report["sidecar_provenance"]
    assert report == handoff.preserve(source, destination, 152500, restore_planning_status=observed)


@pytest.mark.parametrize("fault", ["observed_step", "observed_counter", "extra_field", "same_step",
                                  "world", "counter_total", "counter_decrease", "master", "moment"])
def test_sidecar_restore_rejects_inconsistent_history_and_preserves_full_guard(tmp_path, fault):
    payload, observed = fixture_payload()
    live = advanced_status(observed)
    if fault == "observed_step": observed["completed_steps"] -= 1
    elif fault == "observed_counter": observed["clips_seen"]["kubric"] -= 1
    elif fault == "extra_field": observed["invented"] = True
    elif fault == "same_step": live = deepcopy(observed)
    elif fault == "world": live["world_size"] = 1
    elif fault == "counter_total": live["clips_seen"]["kubric"] += 1
    elif fault == "counter_decrease":
        live["clips_seen"]["kubric"] -= 1401
        live["clips_seen"]["dynamic_replica"] += 1401
    elif fault == "master": payload["model"]["dense_decoder"] = payload["model"]["dense_decoder"].bfloat16()
    elif fault == "moment": payload["optimizer"]["state"]["dense_decoder"]["exp_avg"] = torch.zeros(1, dtype=torch.bfloat16)
    source = save_fixture(tmp_path, payload, live)
    before = (source / "train_status.json").read_bytes()
    with pytest.raises(ValueError):
        handoff.preserve(source, tmp_path / "protected", 152500, restore_planning_status=observed)
    assert not (tmp_path / "protected").exists()
    assert (source / "train_status.json").read_bytes() == before
