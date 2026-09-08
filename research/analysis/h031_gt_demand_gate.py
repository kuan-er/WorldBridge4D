"""Bounded exact raw/published GT parity gate for immediate K5 long training."""
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch
import yaml

from worldbridge.data.cache.native import file_sha256
from worldbridge.data.datasets.native_kubric import NativeKubricDataset
from worldbridge.data.sampling import sample_eligible_targets
from worldbridge.trainer.batching import load_geometry
from worldbridge.trainer.config import validate_config
from worldbridge.trainer.geometry_replay import assert_exact
from worldbridge.utils.io import atomic_json


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    path=Path('configs/h031_k512_po256_dr256_k5_10k_ondemand.yaml');cfg=yaml.safe_load(path.read_text());validate_config(cfg,2)
    assert file_sha256(cfg['selected_checkpoint_path'])==cfg['selected_checkpoint_sha256']
    base=yaml.safe_load(Path('configs/h031_k512_po256_dr256_b1_a4_k5_dr_mmap.yaml').read_text())
    old=NativeKubricDataset(base['datasets']['kubric']);keys=sorted(map(int,old.geometry_ready['entries']))
    selected=[keys[0],keys[len(keys)//2],keys[-1]];start=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='h031-demand-parity-',dir='/data/WorldBridge4D-runs') as temporary:
        values=copy.deepcopy(cfg['datasets']['kubric']);values['native_geometry_root']=temporary
        new=NativeKubricDataset(values)
        for index in selected:
            expected=old.sample(index)
            np_state=np.random.get_state();pt_state=torch.get_rng_state();py_state=random.getstate()
            actual=new.sample(index)
            assert_exact(asdict(expected),asdict(actual))
            assert_exact(np_state,np.random.get_state());assert torch.equal(pt_state,torch.get_rng_state());assert py_state==random.getstate()
            perm=np.random.default_rng(index).permutation(21)
            a,_=load_geometry(old,index,perm,5,123,True,True)
            b,_=load_geometry(new,index,perm,5,123,True,True);assert_exact(a,b)
            rng=np.random.default_rng(index);rng2=copy.deepcopy(rng)
            assert_exact(sample_eligible_targets(a[3],5,rng),sample_eligible_targets(b[3],5,rng2));assert rng.bit_generator.state==rng2.bit_generator.state
            entry=old.geometry_ready['entries'][str(index)];destination=Path(temporary)/f'geom_{index:08d}.npz'
            os.link(old.geometry_root/entry['file'],destination)
            atomic_json(destination.with_suffix('.json'),dict(file=destination.name,sha256=entry['sha256'],manifest_sha256=new.manifest['sha256'],row=new.manifest['records'][index],RGB_sha256=entry['RGB_sha256']))
            published=new.demand_reader.read(index);assert_exact(asdict(actual),asdict(published))
            print(json.dumps(dict(event='NATIVE_GT_DEMAND_PARITY',index=index,fields_geometry_sampling_RNG_exact=True,elapsed_seconds=time.perf_counter()-start)),flush=True)
    report=dict(event='NATIVE_GT_DEMAND_CPU_OK',config_sha256=file_sha256(path),origin_sha256=cfg['selected_checkpoint_sha256'],native_manifest_sha256=old.manifest['sha256'],indices=selected,raw_and_published_exact=True,geometry_RGB_camera_targets_RNG_exact=True,start=150010,stop=160010,K=5,scope='three_real_native_clips_and_regressions_not_full_corpus_GT_predecode',elapsed_seconds=time.perf_counter()-start)
    atomic_json(Path('/data/WorldBridge4D-runs/h031-gt-demand-gate-20260908.json'),report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
