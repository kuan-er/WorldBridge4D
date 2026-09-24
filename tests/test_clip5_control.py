from pathlib import Path

import yaml

from worldbridge.trainer.config import validate_config


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/h031_k512_dr512_full_k9_unfreeze_183000.yaml"
CONTROL = ROOT / "configs/h031_k512_dr512_full_k9_clip5_185500_300.yaml"


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def test_clip5_control_is_a_bounded_matched_resume():
    base, control = load(BASE), load(CONTROL)
    validate_config(control, 2)
    changed = {key for key in base.keys() | control.keys() if base.get(key) != control.get(key)}
    assert changed == {
        "max_steps", "resume_status_path", "gradient_clip", "checkpoint_steps",
        "checkpoint_every_after", "checkpoint_keep_last", "tracking",
    }
    assert control["gradient_clip"] == 5.0
    assert control["max_steps"] == 185800
    assert control["max_steps"] - 185500 == 300
    assert control["selected_checkpoint_step"] == 183000
    assert control["lr_restart"] == base["lr_restart"]
    assert control["dataset_mix_counts"] == {"kubric": 10, "pointodyssey": 5, "dynamic_replica": 5}
    assert control["trainable_mode"] == "full"
    assert control["precision"] == "bf16" and control["fsdp_master_precision"] == "fp32"
    assert control["microbatch_per_gpu"] == 1
    assert control["gradient_accumulation"] == 4
    assert control["targets_per_source"] == 9
    assert control["checkpoint_steps"] == [185800]
    assert control["checkpoint_every_after"] == 0
