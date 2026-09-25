"""H033 keeps the K10 sampling contract that the camera readout depends on."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import yaml

from worldbridge.data.sampling import sample_eligible_targets
from worldbridge.trainer.config import validate_config

ROOT = Path(__file__).resolve().parents[1]


def configs():
    return tuple(yaml.safe_load(Path(p).read_text()) for p in (
        'configs/h032_camera_ray_k10_to210000.yaml',
        'configs/h033_camera_query_ray_to210000.yaml'))


def test_h033_inherits_k10_sampling_and_the_camera_guard():
    parent, new = configs()
    assert parent['targets_per_source'] == new['targets_per_source'] == 10
    assert 'camera_supervision' in parent and 'camera_supervision' in new
    assert new['camera_k10'] is True
    assert {k for k in parent.keys() | new.keys() if parent.get(k) != new.get(k)} == {
        'camera_supervision', 'expected_non_wan_parameters', 'finetune_expected_global_step',
        'finetune_expected_clips_seen', 'finetune_drop_prefixes', 'tracking',
    }
    validate_config(new, 2)


@pytest.mark.parametrize('change', [
    {'targets_per_source': 9},
    {'camera_k10': False},
])
def test_k10_requires_the_explicit_camera_profile(change):
    _, cfg = configs()
    cfg.update(change)
    with pytest.raises(ValueError):
        validate_config(cfg, 2)


def test_k10_exactly_one_diagonal_and_nine_non_diagonal():
    valid = np.ones((21, 3, 3), dtype=bool)
    for source in range(21):
        targets = sample_eligible_targets(valid, 10, np.random.default_rng(123),
                                          diagonal_source=source)
        assert len(targets) == len(set(targets)) == 10
        assert (targets == source).sum() == 1 and (targets != source).sum() == 9


def test_h033_drops_the_deleted_camera_head_from_a_parent_checkpoint():
    _, cfg = configs()
    assert cfg['finetune_drop_prefixes'] == ['camera_head.']
    nested = deepcopy(cfg)
    nested['finetune_drop_prefixes'] = []
    validate_config(nested, 2)  # the config stays valid; the loader is what enforces it
