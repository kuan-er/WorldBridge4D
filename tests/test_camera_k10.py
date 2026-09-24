from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import yaml
from worldbridge.data.sampling import sample_eligible_targets
from worldbridge.trainer.config import validate_config


def configs():
    return tuple(yaml.safe_load(Path(p).read_text()) for p in (
        'configs/h032_camera_ray_196000_to210000.yaml',
        'configs/h032_camera_ray_k10_to210000.yaml'))


def test_k10_is_only_target_count_and_explicit_profile_change():
    old,new=configs()
    assert {k for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}=={'targets_per_source','camera_k10'}
    assert new['targets_per_source']==10 and new['camera_k10'] is True
    validate_config(old,2); validate_config(new,2)
    assert new['max_steps']==210000
    assert new['camera_supervision']['phase_start_step']==196000
    assert new['lr_restart']['start_step']==183000


@pytest.mark.parametrize('change',[{'camera_k10':False},{'targets_per_source':9},{'targets_per_source':11},{'camera_supervision':None}])
def test_k10_requires_explicit_camera_profile(change):
    _,cfg=configs(); cfg.update(change)
    with pytest.raises(ValueError): validate_config(cfg,2)


def test_k10_exactly_one_diagonal_and_nine_non_diagonal():
    valid=np.ones((21,3,3),dtype=bool)
    for source in range(21):
        t=sample_eligible_targets(valid,10,np.random.default_rng(123),diagonal_source=source)
        assert len(t)==len(set(t))==10 and (t==source).sum()==1 and (t!=source).sum()==9
