from pathlib import Path
import copy
import pytest
import yaml
from worldbridge.trainer.config import validate_config

CONFIG = 'configs/h031_k512_k5_mix50_gpu24_to170000.yaml'

def config():
    return yaml.safe_load(Path(CONFIG).read_text())

def test_minimal_k5_override_preserves_science_and_batch():
    c = config()
    validate_config(c, 2)
    old = yaml.safe_load(Path('configs/h031_k512_k11_mix50_prefix5_nostalltrace_to170000.yaml').read_text())
    assert {k for k in c.keys() | old.keys() if c.get(k) != old.get(k)} == {
        'native_kubric512_k11_trial', 'native_kubric512_k5_mix_resume', 'targets_per_source', 'tracking'}
    assert c['microbatch_per_gpu'] * c['gradient_accumulation'] * 2 == 8
    assert c['targets_per_source'] == 5
    assert c['max_steps'] == c['lr_restart']['end_step'] == 170000

@pytest.mark.parametrize('key,value', [
    ('targets_per_source', 11), ('targets_per_source', 9),
    ('native_kubric512_k11_trial', True), ('native_kubric512_k5_mix_resume', False),
    ('native_kubric512_b1_a4_k9', False), ('native_kubric512_k9_mix_trial', False),
    ('native_kubric512_k9_mix_170k', False), ('native_kubric512_b1_a4_k5', True),
    ('gradient_accumulation', 2), ('runtime_stall_traceback_seconds', -1)])
def test_invalid_combinations_fail(key, value):
    c = copy.deepcopy(config()); c[key] = value
    with pytest.raises(ValueError):
        validate_config(c, 2)
