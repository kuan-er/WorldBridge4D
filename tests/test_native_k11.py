from pathlib import Path
import numpy as np
import pytest
import yaml
from worldbridge.trainer.config import validate_config
from worldbridge.data.sampling import sample_eligible_targets

CONFIG = 'configs/h031_k512_k11_mix50_prefix5_to170000.yaml'


def config(): return yaml.safe_load(Path(CONFIG).read_text())


def test_only_K_scientific_delta_and_exact_batch_budget():
    c = config(); validate_config(c, 2)
    old = yaml.safe_load(Path('configs/h031_k512_k9_mix50_prefix5_to170000.yaml').read_text())
    assert {k for k in c.keys() | old.keys() if c.get(k) != old.get(k)} == {
        'native_kubric512_k11_trial', 'targets_per_source', 'tracking'}
    assert c['targets_per_source'] * c['microbatch_per_gpu'] * c['gradient_accumulation'] * 2 == 88
    assert c['native_kubric512_b1_a4_k9']  # same native512 architecture/input path
    assert c['max_steps'] == c['lr_restart']['end_step'] == 170000


@pytest.mark.parametrize('key,value', [('targets_per_source',9), ('targets_per_source',13),
    ('native_kubric512_k11_trial',False), ('native_kubric512_k9_mix_170k',False),
    ('native_kubric512_k9_mix_trial',False), ('native_kubric512_b1_a4_k9',False)])
def test_unregistered_combinations_rejected(key,value):
    c = config(); c[key] = value
    with pytest.raises(ValueError): validate_config(c,2)


def test_eleven_distinct_eligible_targets_no_padding_or_duplication():
    valid = np.zeros((21,2,2),dtype=bool); valid[:11,0,0] = True
    result = sample_eligible_targets(valid,11,np.random.default_rng(424242))
    assert len(result) == len(set(result.tolist())) == 11
    assert set(result.tolist()) == set(range(11))
    valid[10] = False
    with pytest.raises(ValueError): sample_eligible_targets(valid,11,np.random.default_rng(424242))
