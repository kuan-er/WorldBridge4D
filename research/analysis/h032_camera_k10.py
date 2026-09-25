"""User-authorized K9 -> K10 full-state continuation; all other science retained."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import numpy as np
import torch
import yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import h032_camera_train as base
from h032_make_config import make_config as base_config
from worldbridge.data.cache.native import file_sha256
from worldbridge.trainer.config import validate_config
from worldbridge.utils.io import atomic_json
CONFIG=ROOT/'configs/h032_camera_ray_k10_to210000.yaml'
OLD_HANDOFF=base.HANDOFF
OLD_OUTPUT=base.OUTPUT
HANDOFF=Path('/data/WorldBridge4D-runs/h032-camera-ray-k10-handoff-20260924')
OUTPUT=Path('/data/WorldBridge4D-runs/h032-source-camera-ray-k10-to210000-20260924')
SOURCE=OLD_HANDOFF/'capacity.pt'
START=196020
SMOKE_END=START+20


def make_config():
    cfg=base_config()
    cfg['targets_per_source']=10
    cfg['camera_k10']=True
    return cfg


def configuration():
    cfg=yaml.safe_load(CONFIG.read_text())
    assert cfg==make_config()
    validate_config(cfg,2)
    return cfg


def gate():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    cfg=configuration()
    assert not HANDOFF.exists() and not OUTPUT.exists()
    source_report=json.loads((OLD_HANDOFF/'capacity_complete.json').read_text())
    assert source_report['step']==START and source_report['optimizer_states']==1104
    # Old continuation was SIGTERM-stopped during supervisor preflight, before
    # torchrun. Refuse to roll back if a newer completed checkpoint appears.
    old_status=json.loads((OLD_OUTPUT/'train_status.json').read_text())
    assert old_status['completed_steps']==START
    assert max(int(p.stem.split('-')[-1]) for p in OLD_OUTPUT.glob('checkpoint-*.pt'))==START
    assert file_sha256(SOURCE)==source_report['checkpoint_sha256']
    payload=torch.load(SOURCE,map_location='cpu',mmap=True,weights_only=True)
    assert payload['config']==base_config() and payload['training_state']['global_step']==START
    assert len(payload['optimizer']['state'])==1104
    for rng in payload['training_state']['rng_states']: base.check_rng(rng)
    del payload
    assert json.loads(base.CALIBRATION.read_text())['intrinsics_mode']=='per_frame_source_independent'
    subprocess.run([sys.executable,'-m','pytest','-q','tests'],check=True,cwd=ROOT)
    from worldbridge.data.factory import load_training_datasets
    from worldbridge.data.sampling import deterministic_sample_plan,sample_eligible_targets
    from worldbridge.trainer.batching import load_geometry
    from worldbridge.trainer.schedulers import dataset_for_step
    datasets=load_training_datasets(cfg); probes=[]
    for name,dataset in datasets.items():
        step=next(s for s in range(START,SMOKE_END) if dataset_for_step(s,cfg['seed'],cfg['dataset_mix_counts'])==name)
        for rank in (0,1):
            index,_,rng=deterministic_sample_plan(dataset,name,cfg['seed'],step,0,rank,4)
            perm=np.random.default_rng(np.random.SeedSequence([cfg['seed'],step,0,rank,771])).permutation(21)
            fallback=int(np.random.SeedSequence([cfg['seed'],step,0,rank,772]).generate_state(1)[0])
            values,_=load_geometry(dataset,index,perm,10,fallback,True,True,camera_supervision=True)
            actual,source,xyz,valid,rgb,visible,camera,_,_=values
            targets=sample_eligible_targets(valid,10,rng,diagonal_source=source)
            assert len(targets)==len(set(targets))==10 and targets[0]==source
            assert all(valid[t].any() for t in targets)
            probes.append(dict(dataset=name,rank=rank,index=actual,source=source,targets=targets.tolist()))
            print(json.dumps(dict(event='H032_K10_GT_PROBE',**probes[-1])),flush=True)
    uuids,xml=base.health(['1','6'])
    HANDOFF.mkdir()
    os.link(SOURCE,HANDOFF/'resume.pt')
    atomic_json(HANDOFF/'train_status.json',dict(completed_steps=START,world_size=2))
    (HANDOFF/'gpu_health.xml').write_text(xml)
    wandb_id=(OLD_OUTPUT/'wandb_run_id').read_text().strip()
    assert wandb_id=='thlx6gtz'
    report=dict(event='H032_K10_GATE_OK',source_step=START,smoke_end=SMOKE_END,target=210000,
                checkpoint_sha256=source_report['checkpoint_sha256'],config_sha256=file_sha256(CONFIG),
                gpu_ids=['1','6'],gpu_uuids=uuids,wandb_id=wandb_id,probes=probes,
                optimizer_states=1104,fresh_optimizer_states=0,delta='K9->10; forced1diagonal+9offdiagonal;80pairs/update',
                allocator_env={k:os.environ.get(k) for k in base.ALLOCATOR_KEYS})
    assert all(v is None for v in report['allocator_env'].values())
    atomic_json(HANDOFF/'complete.json',report)
    print(json.dumps(report),flush=True)


def train(capacity):
    cfg=configuration()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1,6'
    report=json.loads((HANDOFF/'complete.json').read_text())
    assert report['config_sha256']==file_sha256(CONFIG)
    assert report['allocator_env']=={k:os.environ.get(k) for k in base.ALLOCATOR_KEYS}
    _,xml=base.health(['1','6'],report['gpu_uuids'])
    (HANDOFF/('capacity_gpu.xml' if capacity else 'continue_gpu.xml')).write_text(xml)
    if capacity:
        assert not OUTPUT.exists()
        OUTPUT.mkdir()
        (OUTPUT/'wandb_run_id').write_text(report['wandb_id']+'\n')
        source=HANDOFF/'resume.pt'; sha=report['checkpoint_sha256']
        extra=['--stop-after-updates','20']
    else:
        review=json.loads((HANDOFF/'capacity_complete.json').read_text())
        assert review['step']==SMOKE_END
        source=HANDOFF/'capacity.pt'; sha=review['checkpoint_sha256']; extra=[]
        assert (OUTPUT/'wandb_run_id').read_text().strip()==report['wandb_id']
    assert file_sha256(source)==sha
    os.environ['PYTHONFAULTHANDLER']='1'
    os.environ['WANDB_NAME']='h032-source-conditioned-camera-ray-k10-to210k'
    argv=[sys.executable,'-m','worldbridge.trainer.soft_torchrun','--standalone','--nproc-per-node=2',
          '--log-dir',str(OUTPUT)+('-capacity-elastic' if capacity else '-elastic'),'--tee','3',
          'scripts/train.py','--config',str(CONFIG),'--output-dir',str(OUTPUT),
          '--resume',str(source),'--startup-preflight-updates','5',*extra]
    print(json.dumps(dict(event='H032_K10_GPU_BEGIN',capacity=capacity,gpus=[1,6],argv=argv)),flush=True)
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,lambda *_:None)
    child=subprocess.Popen(argv,cwd=ROOT)
    code=child.wait(); raise SystemExit(code if code>=0 else 128-code)


def review(step):
    # Reuse the strict finite model/Adam-increment/RNG/counter/SHA reviewer.
    # Its immutable196000 origin remains correct for both the initial20 K9
    # updates and the subsequent K10 phase; mixture/counters are unchanged.
    base.CONFIG=CONFIG; base.OUTPUT=OUTPUT; base.HANDOFF=HANDOFF
    base.configuration=configuration
    base.review(step,capacity_step=SMOKE_END)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',required=True,choices=['write-config','gate','capacity','continue','review-capacity','review-endpoint'])
    args=p.parse_args(); torch.set_num_threads(2)
    os.environ['PYTHONPATH']=str(ROOT/'src')+os.pathsep+os.environ.get('PYTHONPATH','')
    if args.mode=='write-config': CONFIG.write_text(yaml.safe_dump(make_config(),sort_keys=False))
    elif args.mode=='gate': gate()
    elif args.mode.startswith('review'):
        assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
        review(SMOKE_END if args.mode=='review-capacity' else 210000)
    else: train(args.mode=='capacity')
if __name__=='__main__': main()
