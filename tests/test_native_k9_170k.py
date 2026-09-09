from pathlib import Path
import copy
import pytest
import yaml
from worldbridge.trainer.config import validate_config
from worldbridge.trainer.preflight import preflight_end_step

CONFIG = 'configs/h031_k512_k9_mix50_prefix5_to170000.yaml'


def config():
    return yaml.safe_load(Path(CONFIG).read_text())


def test_only_authorized_170k_delta():
    new = config(); validate_config(new, 2)
    old = yaml.safe_load(Path('configs/h031_k512_k9_mix50_152768_to154768.yaml').read_text())
    assert {k for k in new.keys() | old.keys() if new.get(k) != old.get(k)} == {
        'native_kubric512_k9_mix_170k', 'max_steps', 'lr_restart', 'checkpoint_steps', 'tracking'}
    restart = copy.deepcopy(old['lr_restart']); restart['end_step'] = 170000
    assert new['lr_restart'] == restart
    assert new['selected_checkpoint_step'] == 152768
    assert new['max_steps'] == 170000
    assert preflight_end_step(152769, 170000) == 152774


@pytest.mark.parametrize('key,value', [('max_steps', 169999), ('targets_per_source', 5),
    ('native_kubric512_k9_mix_trial', False), ('native_kubric512_b1_a4_k9', False),
    ('native_kubric512_k9_mix_170k', False)])
def test_invalid_170k_config_rejected(key, value):
    c = config(); c[key] = value
    with pytest.raises(ValueError): validate_config(c, 2)


@pytest.mark.parametrize('key,value', [('start_step',152768), ('warmup_steps',0), ('end_step',160010)])
def test_lr_origin_and_hold_horizon_guard(key, value):
    c = config(); c['lr_restart'][key] = value
    with pytest.raises(ValueError): validate_config(c, 2)
