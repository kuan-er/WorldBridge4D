from pathlib import Path
import pytest
import yaml
from worldbridge.trainer.config import validate_config

CONFIG = 'configs/h031_k512_k3_mix50_gpu25_to170000.yaml'

def config():
    return yaml.safe_load(Path(CONFIG).read_text())

def test_minimal_k3_override_preserves_science():
    c = config()
    validate_config(c, 2)
    old = yaml.safe_load(Path('configs/h031_k512_k5_mix50_gpu24_to170000.yaml').read_text())
    assert {k for k in c.keys() | old.keys() if c.get(k) != old.get(k)} == {
        'native_kubric512_k5_mix_resume', 'native_kubric512_k3_mix_resume', 'targets_per_source', 'tracking'}
    assert c['microbatch_per_gpu'] * c['gradient_accumulation'] * 2 == 8
    assert c['targets_per_source'] == 3
    assert c['max_steps'] == c['lr_restart']['end_step'] == 170000
    assert c['lr_restart']['start_step'] == 150000
    assert c['cycle_reprojection_enabled'] and c['cycle_reprojection_weight'] == 0

@pytest.mark.parametrize('key,value', [
    ('targets_per_source', 5), ('targets_per_source', 9),
    ('native_kubric512_k11_trial', True), ('native_kubric512_k5_mix_resume', True),
    ('native_kubric512_k3_mix_resume', False), ('native_kubric512_b1_a4_k9', False),
    ('native_kubric512_k9_mix_trial', False), ('native_kubric512_k9_mix_170k', False),
    ('native_kubric512_b1_a4_k5', True), ('gradient_accumulation', 2),
    ('runtime_stall_traceback_seconds', -1)])
def test_invalid_k3_contract_rejected(key, value):
    c = config(); c[key] = value
    with pytest.raises(ValueError):
        validate_config(c, 2)
