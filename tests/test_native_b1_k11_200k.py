"""User K11-only continuation of B1/A4/K9 on unchanged model and200k horizon."""
from pathlib import Path
import pytest
import yaml
from worldbridge.trainer.config import validate_config

CONFIG='configs/h031_k512_b1_a4_k11_mix50_gpu67_to200000.yaml'

def config():
    return yaml.safe_load(Path(CONFIG).read_text())

def test_only_k_profile_and_tracking_changed():
    c=config(); parent=yaml.safe_load(Path('configs/h031_k512_b1_a4_k9_mix50_gpu67_to200000.yaml').read_text())
    validate_config(c,2);validate_config(parent,2)
    assert {k for k in c.keys() | parent.keys() if c.get(k)!=parent.get(k)} == {
        'native_kubric512_b1_a4_k9_mix_200k','native_kubric512_b1_a4_k11_mix_200k',
        'native_kubric512_k11_trial','targets_per_source','tracking'}
    assert c['microbatch_per_gpu']==1 and c['gradient_accumulation']==4
    assert c['targets_per_source']==11 and 1*4*2*11==88
    assert c['max_steps']==c['lr_restart']['end_step']==200000
    assert c['selected_checkpoint_step']==152768
    assert c['lr_restart']==parent['lr_restart']

@pytest.mark.parametrize('key,value',[
    ('native_kubric512_b1_a4_k11_mix_200k',False),('native_kubric512_k11_trial',False),
    ('targets_per_source',9),('microbatch_per_gpu',2),('gradient_accumulation',2),
    ('max_steps',170000),('native_kubric512_b1_a4_k9_mix_200k',True),
    ('native_kubric512_b2_a2_k9_mix_200k',True),('native_kubric512_b2_a2_k5_mix_200k',True),
    ('native_kubric512_k5_mix_resume',True),('native_kubric512_k3_mix_resume',True),
    ('native_kubric512_k3_mix_200k',True),('native_kubric512_b1_a4_k9',False),
    ('native_kubric512_k9_mix_170k',False)])
def test_invalid_profile_rejected(key,value):
    c=config();c[key]=value
    with pytest.raises(ValueError):validate_config(c,2)

def test_b2a2_and_wrong_world_rejected():
    with pytest.raises(ValueError):validate_config(config(),1)
    c=config();c.update(microbatch_per_gpu=2,gradient_accumulation=2)
    with pytest.raises(ValueError):validate_config(c,2)
