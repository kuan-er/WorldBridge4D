"""Real K5 DR input parity: legacy NPZ versus shared mmap, no training changes."""
import argparse
import copy
import json
import os
from pathlib import Path
import time

import numpy as np
import yaml

from worldbridge.data.cache.native import file_sha256
from worldbridge.data.factory import load_dataset
from worldbridge.data.sampling import deterministic_sample_plan, sample_eligible_targets
from worldbridge.trainer.batching import load_geometry
from worldbridge.trainer.config import validate_config
from worldbridge.trainer.geometry_replay import assert_exact
from worldbridge.utils.io import atomic_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    cfg=yaml.safe_load(Path(args.config).read_text());validate_config(cfg,2)
    assert cfg['targets_per_source']==5 and cfg['native_kubric512_b1_a4_k5']
    prior=json.loads(Path('/data/WorldBridge4D-runs/h031-native512-capacity-inputs-20260908/cpu_complete.json').read_text())
    assert prior['config_sha256']==file_sha256('configs/h031_k512_po256_dr256_b1_a4_k15_capacity.yaml')
    assert prior['origin_sha256']==cfg['selected_checkpoint_sha256'] and prior['optimizer_states']==193 and prior['decoder_checkpoint_names_shapes_exact']
    old_cfg=copy.deepcopy(cfg)
    for k in ('trajectory_mmap_root','trajectory_mmap_complete_sha256'):old_cfg['datasets']['dynamic_replica'].pop(k)
    old=load_dataset(old_cfg,'dynamic_replica');new=load_dataset(cfg,'dynamic_replica')
    vae_sha=file_sha256(cfg['vae_checkpoint']);old.set_lazy_vae_sha256(vae_sha);new.set_lazy_vae_sha256(vae_sha)
    assert old.rows==new.rows and old.geometry_indices==new.geometry_indices
    assert new.geometry._trajectory_mmap is not None
    # Prove the training factory's chosen adapter cannot decode legacy NPZ/pth.
    def forbidden(*args):raise AssertionError('mmap geometry reached legacy stream decoder')
    new.geometry._load_stream=forbidden
    requests=[]
    for rank in range(2):
        for slot in range(4):
            index,_,rng=deterministic_sample_plan(old,'dynamic_replica',cfg['seed'],150000,slot,rank,4)
            requests.append((index,rank,slot,rng))
    streams=sorted({str(row['stream']) for row in old.geometry.rows})
    assert len(streams)==435 and len(old)==6090
    for number in (0,len(streams)//2,len(streams)-1):
        index=next(i for i,g in enumerate(old.geometry_indices) if old.geometry.rows[g]['stream']==streams[number])
        requests.append((index,0,number,np.random.default_rng(number)))
    start=time.perf_counter();results=[]
    for index,rank,slot,rng in requests:
        perm=np.random.default_rng(np.random.SeedSequence([cfg['seed'],150000,slot,rank,771])).permutation(21)
        fallback=int(np.random.SeedSequence([cfg['seed'],150000,slot,rank,772]).generate_state(1)[0])
        g=old.geometry_indices[index]
        assert_exact(old.geometry._load_clip(old.geometry.rows[g]),new.geometry._load_clip(new.geometry.rows[g]))
        a,old_seconds=load_geometry(old,index,perm,5,fallback,True,True)
        b,mmap_seconds=load_geometry(new,index,perm,5,fallback,True,True)
        assert_exact(a,b)
        rng2=copy.deepcopy(rng);targets=sample_eligible_targets(a[3],5,rng)
        np.testing.assert_array_equal(targets,sample_eligible_targets(b[3],5,rng2))
        assert rng.bit_generator.state==rng2.bit_generator.state
        selected,source=a[:2];reverse=int(targets[np.argmax(np.abs(targets-source))])
        assert_exact(old.source_rgb(selected,reverse),new.source_rgb(selected,reverse))
        assert_exact(old.clean_latent(selected),new.clean_latent(selected))
        results.append(dict(index=index,selected=selected,source=source,rank=rank,slot=slot,targets=targets.tolist(),legacy_geometry_seconds=old_seconds,mmap_geometry_seconds=mmap_seconds))
        print(json.dumps(dict(event='DR_MMAP_TRAINING_PARITY',count=len(results),index=index,source=source,elapsed_seconds=time.perf_counter()-start)),flush=True)
    report=dict(event='DR_MMAP_K5_TRAINING_CPU_OK',config_sha256=file_sha256(args.config),origin_sha256=cfg['selected_checkpoint_sha256'],mmap_complete_sha256=cfg['datasets']['dynamic_replica']['trajectory_mmap_complete_sha256'],seed=cfg['seed'],B=1,A=4,K=5,world=2,pairs_per_update=40,real_requests=results,scope='first8_DR_requests_plus_first_middle_last_stream_samples_all_source_synthetic_parity',exact_geometry_RGB_camera_latent_targets_RNG=True,no_legacy_mmap_fallback=True,timing_not_controlled_speedup_benchmark=True,elapsed_seconds=time.perf_counter()-start)
    atomic_json(Path(args.output),report)
    print(json.dumps({k:v for k,v in report.items() if k!='real_requests'}),flush=True)


if __name__=='__main__':main()
