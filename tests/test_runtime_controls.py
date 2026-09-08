from pathlib import Path

import pytest
import yaml

from worldbridge.trainer.config import validate_config
from worldbridge.trainer.distributed import initialize_distributed


@pytest.mark.parametrize("key,value", [
    ("cuda_empty_cache_every_steps", -1),
    ("trace_first_updates", -1),
    ("runtime_stall_traceback_seconds", float("nan")),
    ("distributed_timeout_seconds", 0),
])
def test_runtime_controls_fail_closed(key, value):
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/h030_150k_lrrestart_gpu23_b2_k19.yaml").read_text())
    validate_config(config, world=2)
    with pytest.raises(ValueError):
        validate_config({**config, key: value}, world=2)


def test_invalid_collective_timeout_fails_before_cuda_or_process_group():
    with pytest.raises(ValueError, match="timeout"):
        initialize_distributed(timeout_seconds=0)
