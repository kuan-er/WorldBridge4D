from pathlib import Path

import pytest
import yaml

from worldbridge.trainer.config import validate_config


def test_user_k15_variant_preserves_native_reuse_lr_and_batch():
    root = Path(__file__).resolve().parents[1]
    previous = yaml.safe_load((root / "configs/h030_150k_to_160k_gpu23_b2_k19_native_reuse.yaml").read_text())
    config = yaml.safe_load((root / "configs/h030_150k_to_160k_gpu23_b2_k15_native_reuse.yaml").read_text())
    validate_config(config, world=2)
    assert config["targets_per_source"] == 15
    assert config["microbatch_per_gpu"] == config["gradient_accumulation"] == 2
    assert config["cuda_empty_cache_every_steps"] == 0
    assert {key for key in previous.keys() | config.keys() if previous.get(key) != config.get(key)} == {
        "targets_per_source", "cycle_b2_a2_k19", "cycle_b2_a2_k15", "tracking",
    }
    with pytest.raises(ValueError, match="only one"):
        validate_config({**config, "cycle_b2_a2_k19": True}, world=2)
    with pytest.raises(ValueError, match="targets_per_source=15"):
        validate_config({**config, "targets_per_source": 19}, world=2)
    with pytest.raises(ValueError):
        validate_config({**config, "gradient_accumulation": 4}, world=2)
