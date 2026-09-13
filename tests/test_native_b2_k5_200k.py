"""User K5-only retry, retaining B2/A2/GPU6,7 and200k horizon."""
from pathlib import Path
import pytest
import yaml
from worldbridge.trainer.config import validate_config
from test_native_b2_k9_200k import test_b2_k9_forward_reverse_backward_native_alignment as check_shapes

CONFIG='configs/h031_k512_b2_a2_k5_mix50_gpu67_to200000.yaml'

def config():
    return yaml.safe_load(Path(CONFIG).read_text())

def test_only_k_profile_and_tracking_changed():
    c=config(); old=yaml.safe_load(Path('configs/h031_k512_b2_a2_k9_mix50_gpu67_to200000.yaml').read_text())
    validate_config(c,2); validate_config(old,2)
    assert {k for k in c.keys() | old.keys() if c.get(k)!=old.get(k)} == {
        'native_kubric512_b2_a2_k9_mix_200k','native_kubric512_b2_a2_k5_mix_200k',
        'native_kubric512_k5_mix_resume','targets_per_source','tracking'}
    assert c['microbatch_per_gpu']==c['gradient_accumulation']==2
    assert c['targets_per_source']==5 and 2*2*2*5==40
    assert c['max_steps']==c['lr_restart']['end_step']==200000
    assert c['selected_checkpoint_step']==152768
    assert c['lr_restart']==old['lr_restart']

@pytest.mark.parametrize('key,value',[
    ('native_kubric512_b2_a2_k5_mix_200k',False),('native_kubric512_k5_mix_resume',False),
    ('native_kubric512_b2_a2_k9_mix_200k',True),('targets_per_source',9),
    ('microbatch_per_gpu',1),('gradient_accumulation',4),('max_steps',170000),
    ('native_kubric512_k3_mix_resume',True),('native_kubric512_k11_trial',True),
    ('native_kubric512_k3_mix_200k',True),('native_kubric512_b1_a4_k9',False),
    ('native_kubric512_k9_mix_170k',False)])
def test_invalid_profile_rejected(key,value):
    c=config();c[key]=value
    with pytest.raises(ValueError):validate_config(c,2)

@pytest.mark.parametrize('resolution',[256,512])
def test_b2_k5_actual_forward_reverse_backward(resolution):
    check_shapes(resolution,pairs=5)
