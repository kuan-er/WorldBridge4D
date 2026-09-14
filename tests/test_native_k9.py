from pathlib import Path

import pytest
import yaml

from worldbridge.trainer.config import validate_config

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'configs/h030_150k_to_155k_gpu23_b2_k15_boundary2x_edgecontrast0p01.yaml'
VARIANT = ROOT / 'configs/h030_151500_to_155k_gpu23_b2_k9_boundary2x_edgecontrast0p01.yaml'


def test_k9_only_changes_target_profile_and_tracking():
    old = yaml.safe_load(PARENT.read_text())
    new = yaml.safe_load(VARIANT.read_text())
    validate_config(old, world=2)
    validate_config(new, world=2)
    assert {k for k in old.keys() | new.keys() if old.get(k) != new.get(k)} == {
        'targets_per_source', 'cycle_b2_a2_k15', 'cycle_b2_a2_k9', 'tracking',
    }
    assert new['targets_per_source'] == 9
    assert 2 * new['microbatch_per_gpu'] * new['gradient_accumulation'] * new['targets_per_source'] == 72
    assert new['lr_restart'] == old['lr_restart']


@pytest.mark.parametrize('override', [
    {'cycle_b2_a2_k15': True}, {'cycle_b2_a2_k19': True},
    {'xyz_b2_a2_k15': True}, {'cycle_reprojection_enabled': False},
    {'targets_per_source': 15}, {'targets_per_source': 7},
    {'microbatch_per_gpu': 1}, {'gradient_accumulation': 4},
])
def test_k9_rejects_incompatible_profiles(override):
    config = yaml.safe_load(VARIANT.read_text())
    with pytest.raises(ValueError):
        validate_config({**config, **override}, world=2)


def test_k9_requires_two_ranks_and_explicit_opt_in():
    config = yaml.safe_load(VARIANT.read_text())
    with pytest.raises(ValueError):
        validate_config(config, world=1)
    config.pop('cycle_b2_a2_k9')
    with pytest.raises(ValueError):
        validate_config(config, world=2)
