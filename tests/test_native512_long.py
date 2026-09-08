from pathlib import Path
import copy
import pytest
import yaml

from worldbridge.trainer.config import validate_config

PATH=Path('configs/h031_k512_po256_dr256_k5_150010_to160010.yaml')


def config():return yaml.safe_load(PATH.read_text())


def test_full10k_scope_and_unchanged_science():
    a=yaml.safe_load(Path('configs/h031_k512_po256_dr256_b1_a4_k5_dr_mmap.yaml').read_text());b=config()
    validate_config(a,2);validate_config(b,2)
    changed={k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
    assert changed=={'native_kubric512_k5_10k','native_capacity_test_only','max_steps','selected_checkpoint_step','selected_checkpoint_sha256','selected_checkpoint_path','checkpoint_steps','lr_restart','datasets','tracking'}
    assert b['max_steps']-b['selected_checkpoint_step']==10000
    assert b['targets_per_source']==5 and b['microbatch_per_gpu']==1 and b['gradient_accumulation']==4
    restart=copy.deepcopy(b['lr_restart']);restart['end_step']=a['lr_restart']['end_step'];assert restart==a['lr_restart']
    datasets=copy.deepcopy(b['datasets']);datasets['kubric']['native_geometry_root']=a['datasets']['kubric']['native_geometry_root'];datasets['kubric'].pop('native_geometry_full_corpus');assert datasets==a['datasets']


@pytest.mark.parametrize('key,value',[('max_steps',170010),('max_steps',160000),('selected_checkpoint_step',150000),('native_capacity_test_only',True),('native_kubric512_k5_10k',False),('targets_per_source',9)])
def test_long_profile_fails_closed(key,value):
    cfg=config();cfg[key]=value
    with pytest.raises(ValueError):validate_config(cfg,2)


def test_no_restart_or_partial_GT_admission():
    cfg=config();cfg['datasets']['kubric']['native_geometry_full_corpus']=False
    with pytest.raises(ValueError):validate_config(cfg,2)
    cfg=config();cfg['lr_restart']['start_step']=150010
    with pytest.raises(ValueError):validate_config(cfg,2)
