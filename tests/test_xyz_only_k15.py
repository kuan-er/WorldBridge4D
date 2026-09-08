"""Guard the optional single-variable control; this does not select or launch it."""
from pathlib import Path

import pytest
import yaml

from worldbridge.trainer.config import validate_config

ROOT = Path(__file__).resolve().parents[1]


def config(name):
    return yaml.safe_load((ROOT / "configs" / name).read_text())


def test_xyz_only_matches_cycle_through_155k():
    cycle = config("h030_150k_to_160k_gpu23_b2_k15_native_reuse.yaml")
    xyz = config("h030_150k_to_155k_gpu23_b2_k15_xyz_only.yaml")
    validate_config(cycle, world=2)
    validate_config(xyz, world=2)
    assert {key for key in cycle.keys() | xyz.keys() if cycle.get(key) != xyz.get(key)} == {
        "cycle_reprojection_enabled", "cycle_b2_a2_k15", "xyz_b2_a2_k15",
        "max_steps", "lr_restart", "checkpoint_steps", "tracking",
    }
    assert xyz["cycle_reprojection_enabled"] is False
    assert xyz["xyz_b2_a2_k15"] is True
    assert "cycle_b2_a2_k15" not in xyz
    assert xyz["lr_restart"] == {**cycle["lr_restart"], "end_step": 155000}
    assert xyz["max_steps"] - xyz["selected_checkpoint_step"] == 5000
    assert xyz["checkpoint_steps"] == [s for s in cycle["checkpoint_steps"] if s <= 155000]
    assert xyz["cuda_empty_cache_every_steps"] == 0
    assert xyz["microbatch_per_gpu"] == xyz["gradient_accumulation"] == 2
    assert xyz["targets_per_source"] == 15


@pytest.mark.parametrize("override", [
    {"cycle_reprojection_enabled": True},
    {"cycle_b2_a2_k15": True},
    {"cycle_b2_a2_k19": True},
    {"targets_per_source": 19},
    {"targets_per_source": 13},
    {"microbatch_per_gpu": 1},
    {"gradient_accumulation": 4},
    {"xyz_b2_a2_k15": False},
])
def test_xyz_profile_rejects_unrequested_protocol(override):
    xyz = config("h030_150k_to_155k_gpu23_b2_k15_xyz_only.yaml")
    with pytest.raises(ValueError):
        validate_config({**xyz, **override}, world=2)


def test_cycle_k15_still_requires_cycle_objective():
    cycle = config("h030_150k_to_160k_gpu23_b2_k15_native_reuse.yaml")
    with pytest.raises(ValueError, match="require the cycle objective"):
        validate_config({**cycle, "cycle_reprojection_enabled": False}, world=2)


def test_xyz_profile_still_requires_two_ranks():
    xyz = config("h030_150k_to_155k_gpu23_b2_k15_xyz_only.yaml")
    with pytest.raises(ValueError, match="exactly 2 ranks"):
        validate_config(xyz, world=1)
