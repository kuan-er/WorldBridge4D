from pathlib import Path
import copy
import numpy as np
import pytest
import yaml

from worldbridge.trainer.config import validate_config
from worldbridge.trainer.batching import load_geometry

CONFIG = Path('configs/h031_k512_po256_dr256_b1_a4_k9_timeout900.yaml')


def config():
    return yaml.safe_load(CONFIG.read_text())


def test_k9_only_admitted_delta():
    a = yaml.safe_load(Path('configs/h031_k512_po256_dr256_b1_a4_k15_timeout900.yaml').read_text())
    b = config(); validate_config(a,2); validate_config(b,2)
    assert {k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)} == {
        'native_kubric512_b1_a4_k15','native_kubric512_b1_a4_k9','targets_per_source','tracking'}
    assert b['microbatch_per_gpu']==1 and b['gradient_accumulation']==4
    assert b['targets_per_source']==9 and b['max_steps']==150010
    assert b['distributed_timeout_seconds']==900
    assert 2*b['microbatch_per_gpu']*b['gradient_accumulation']*b['targets_per_source']==72


@pytest.mark.parametrize('key,value',[
    ('targets_per_source',15),('targets_per_source',7),('native_kubric512_b1_a4_k15',True),
    ('microbatch_per_gpu',2),('gradient_accumulation',2),('max_steps',155000),
    ('cycle_reprojection_weight',0.1),('fsdp_master_precision','model')])
def test_k9_fail_closed(key,value):
    cfg=config(); cfg[key]=value
    with pytest.raises(ValueError):validate_config(cfg,2)


def test_k9_native_data_route_and_existing_external_cache(monkeypatch):
    from worldbridge.data import factory
    from worldbridge.data.datasets import native_kubric
    sentinel=object()
    monkeypatch.setattr(native_kubric,'NativeKubricDataset',lambda values:sentinel)
    monkeypatch.setattr(factory,'_load_pointodyssey_dataset',lambda values,**kwargs:kwargs)
    cfg=config();assert factory.load_dataset(cfg,'kubric') is sentinel
    assert factory.load_dataset(cfg,'pointodyssey')['allow_missing_latents']
    with pytest.raises(ValueError):factory.load_dataset(cfg,'kubric',allow_missing_latents=True)


def test_lower_k_can_change_eligible_source_not_exact_k15_replay():
    class Dataset:
        def __len__(self):return 1
        def source_all_targets_with_visibility(self,index,source):
            valid=np.zeros((21,1,1),bool);valid[:9 if source==0 else 15]=True
            return np.zeros((21,3,1,1),np.float32),valid,valid.copy()
        def cycle_camera(self,index):return {'positions':np.zeros((21,3),np.float32)}
    ds=Dataset();perm=np.arange(21)
    small,_=load_geometry(ds,0,perm,9,123,False,True)
    large,_=load_geometry(ds,0,perm,15,123,False,True)
    assert small[1]==0 and large[1]==1
