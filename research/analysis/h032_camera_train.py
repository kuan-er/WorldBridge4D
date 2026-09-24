"""CPU gate -> two-rank20-update migration smoke -> review ->210k -> review.

Submit each stage through PRL with success-only dependencies and exact GPU IDs.
No unmanaged training, automatic retries, device substitutions, or SIGKILL.
The shared-intrinsics prototype intentionally refuses training on failed audit.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET
import numpy as np
import torch
import yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from h032_make_config import make_config
from h031_resume191000_to200000 import check_rng
from worldbridge.data.cache.native import file_sha256
from worldbridge.trainer.config import validate_config
from worldbridge.trainer.schedulers import dataset_for_step
from worldbridge.utils.io import atomic_json
CONFIG=ROOT/'configs/h032_camera_ray_196000_to210000.yaml'
SOURCE_ROOT=Path('/data/WorldBridge4D-runs/h032-camera-ray-source196000-20260924')
CALIBRATION=Path('/data/WorldBridge4D-runs/h032-camera-calibration-audit-r2-20260924/complete.json')
HANDOFF=Path('/data/WorldBridge4D-runs/h032-camera-ray-handoff-20260924')
OUTPUT=Path('/data/WorldBridge4D-runs/h032-source-camera-ray-196000-to210000-20260924')
ALLOCATOR_KEYS=('PYTORCH_CUDA_ALLOC_CONF','PYTORCH_ALLOC_CONF','PYTORCH_NO_CUDA_MEMORY_CACHING')


def configuration():
    cfg=yaml.safe_load(CONFIG.read_text())
    assert cfg==make_config()
    validate_config(cfg,2)
    return cfg


def health(gpu_ids, expected_uuids=None):
    assert len(gpu_ids)==len(set(gpu_ids))==2
    raw=subprocess.check_output(['nvidia-smi','-i',','.join(gpu_ids),'-q','-x'],timeout=20)
    gpus=ET.fromstring(raw).findall('gpu')
    uuids=[g.findtext('uuid') for g in gpus]
    assert len(uuids)==2
    if expected_uuids is not None: assert uuids==expected_uuids
    for g in gpus:
        for key in ('dram_uncorrectable','sram_uncorrectable_parity','sram_uncorrectable_secded'):
            assert g.findtext('ecc_errors/volatile/'+key)=='0',(g.findtext('uuid'),key)
        for key in ('remapped_row_pending','remapped_row_failure'):
            assert g.findtext('remapped_rows/'+key)=='No',(g.findtext('uuid'),key)
    return uuids,raw.decode()


def gate(gpu_ids):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    cfg=configuration()
    assert not HANDOFF.exists() and not OUTPUT.exists()
    assert shutil.disk_usage(OUTPUT.parent).free >= 68719476736+100000000000
    # A missing audit marker is a hard blocker, not permission to average GT K.
    calibration=json.loads(CALIBRATION.read_text())
    assert calibration['event']=='H032_CAMERA_AUDIT_OK'
    source_report=json.loads((SOURCE_ROOT/'complete.json').read_text())
    assert source_report['event']=='H032_SOURCE_OK' and source_report['review']['global_step']==196000
    assert source_report['review']['clips_seen']==cfg['finetune_expected_clips_seen']
    assert file_sha256(SOURCE_ROOT/'resume.pt')==source_report['checkpoint_sha256']
    subprocess.run([sys.executable,'-m','pytest','-q','tests'],check=True,cwd=ROOT)
    from worldbridge.models.factory import build_real_model
    model=build_real_model(cfg,'cpu',load_wan_pretrained=False)
    payload=torch.load(SOURCE_ROOT/'resume.pt',map_location='cpu',mmap=True,weights_only=True)
    old=payload['model']; current=model.state_dict()
    assert not set(old)-set(current)
    fresh=set(current)-set(old)
    assert fresh and all(n.startswith('camera_head.') for n in fresh)
    assert all(current[n].shape==v.shape for n,v in old.items())
    assert len(fresh)==51 and sum(current[n].numel() for n in fresh)==1414409
    names={n for n,p in model.named_parameters() if p.requires_grad}
    assert set(payload['optimizer']['state']) <= names
    assert names-set(payload['optimizer']['state'])==fresh
    del model,payload,current,old; gc.collect()
    # Actual source-depth diagonal geometry checks on both ranks/all datasets.
    from worldbridge.data.factory import load_training_datasets
    from worldbridge.data.sampling import deterministic_sample_plan, sample_eligible_targets
    from worldbridge.trainer.batching import load_geometry
    from worldbridge.trainer.camera_objective import unit_source_rays
    datasets=load_training_datasets(cfg)
    probes=[]
    for name,dataset in datasets.items():
        step=next(s for s in range(196000,196020) if dataset_for_step(s,cfg['seed'],cfg['dataset_mix_counts'])==name)
        for rank in (0,1):
            index,_,rng=deterministic_sample_plan(dataset,name,cfg['seed'],step,0,rank,4)
            perm=np.random.default_rng(np.random.SeedSequence([cfg['seed'],step,0,rank,771])).permutation(21)
            fallback=int(np.random.SeedSequence([cfg['seed'],step,0,rank,772]).generate_state(1)[0])
            value,seconds=load_geometry(dataset,index,perm,9,fallback,True,True,camera_supervision=True)
            index,source,xyz,valid,rgb,visible,camera,_,_=value
            targets=sample_eligible_targets(valid,9,rng,diagonal_source=source)
            K=torch.from_numpy(camera['intrinsics'][source:source+1])
            rays=unit_source_rays(K,*valid.shape[-2:])[0]
            points=torch.from_numpy(xyz[source]); mask=torch.from_numpy(valid[source])
            perp=points-(points*rays).sum(0,keepdim=True)*rays
            error=perp.norm(dim=0)[mask]
            assert error.numel()>0 and torch.isfinite(error).all() and error.max()<1e-3
            assert rgb.shape[:2]==valid.shape[-2:] and targets[0]==source
            probes.append(dict(dataset=name,rank=rank,step=step,index=index,source=source,
                               targets=targets.tolist(),diagonal_points=int(mask.sum()),
                               GT_ray_max_m=float(error.max()),geometry_seconds=seconds))
            print(json.dumps(dict(event='H032_REAL_GT_PROBE',**probes[-1])),flush=True)
        del value,xyz,valid,rgb,visible,camera
    uuids,xml=health(gpu_ids)
    HANDOFF.mkdir()
    (HANDOFF/'gpu_health.xml').write_text(xml)
    report=dict(event='H032_TRAIN_GATE_OK',config_sha256=file_sha256(CONFIG),
        calibration_sha256=file_sha256(CALIBRATION),source=str(SOURCE_ROOT/'resume.pt'),
        checkpoint_sha256=source_report['checkpoint_sha256'],
        gpu_ids=gpu_ids,gpu_uuids=uuids,probes=probes,seed=cfg['seed'],decoder_seed=cfg['decoder_seed'],
        camera_seed=cfg['camera_supervision']['seed'],python=sys.version,torch=torch.__version__,
        cuda=torch.version.cuda,platform=platform.platform(),allocator_env={k:os.environ.get(k) for k in ALLOCATOR_KEYS},
        migration=dict(old_Adam=1053,fresh_camera_Adam=51),capacity_updates=20,target=210000)
    assert all(v is None for v in report['allocator_env'].values()), 'preserve parent default allocator'
    atomic_json(HANDOFF/'complete.json',report)
    print(json.dumps(report),flush=True)


def review(step):
    cfg=configuration(); path=OUTPUT/f'checkpoint-{step:07d}.pt'
    p=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
    assert p['config']==cfg
    state=p['training_state']; assert state['global_step']==step
    assert state['world_size']==len(state['rng_states'])==2
    assert state['dataset_mix_phase_origin']==183000 and state['dataset_mix_counts']==cfg['dataset_mix_counts']
    assert state['dataset_cycle_offset']==step%20
    for rng in state['rng_states']: check_rng(rng)
    source=torch.load(SOURCE_ROOT/'resume.pt',map_location='cpu',mmap=True,weights_only=True)
    assert p['coordinate_mean']==source['coordinate_mean'] and p['coordinate_scale']==source['coordinate_scale']
    expected=source['training_state']['clips_seen'].copy()
    for update in range(196000,step): expected[dataset_for_step(update,cfg['seed'],cfg['dataset_mix_counts'])]+=8
    assert state['clips_seen']==expected
    old=source['optimizer']['state']; new=p['optimizer']['state']
    assert len(old)==1053 and len(new)==1104 and set(old)<=set(new)
    assert all(n.startswith('camera_head.') for n in set(new)-set(old))
    for n,v in p['model'].items(): assert torch.isfinite(v).all(),n
    for n,v in new.items():
        assert int(v['step'])==(int(old[n]['step']) if n in old else 0)+step-196000,(n,v['step'])
        for key in ('exp_avg','exp_avg_sq'):
            assert v[key].dtype==torch.float32 and torch.isfinite(v[key]).all(),(n,key)
            assert v[key].shape==p['model'][n].shape
    report=dict(event='H032_CHECKPOINT_VERIFIED',step=step,checkpoint=str(path),
                checkpoint_sha256=file_sha256(path),optimizer_states=len(new),old_retained=len(old),
                camera_states=51,world_size=2,rng_ranks=2,clips_seen=expected)
    if step==196020:
        os.link(path,HANDOFF/'capacity.pt')
        atomic_json(HANDOFF/'train_status.json',dict(completed_steps=step,world_size=2))
        atomic_json(HANDOFF/'capacity_complete.json',report)
    else: atomic_json(OUTPUT/'endpoint_review.json',report)
    print(json.dumps(report),flush=True)


def train(gpu_ids, capacity=False):
    cfg=configuration()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==','.join(gpu_ids)
    gate_report=json.loads((HANDOFF/'complete.json').read_text())
    assert gate_report['config_sha256']==file_sha256(CONFIG) and gate_report['gpu_ids']==gpu_ids
    assert gate_report['allocator_env']=={k:os.environ.get(k) for k in ALLOCATOR_KEYS}
    _,xml=health(gpu_ids,gate_report['gpu_uuids'])
    (HANDOFF/('gpu_capacity.xml' if capacity else 'gpu_continue.xml')).write_text(xml)
    if capacity:
        source=SOURCE_ROOT/'resume.pt'; sha=gate_report['checkpoint_sha256']
        assert not OUTPUT.exists(); OUTPUT.mkdir()
        resume_args=['--finetune-from',str(source),'--stop-after-updates','20']
    else:
        review_report=json.loads((HANDOFF/'capacity_complete.json').read_text())
        assert review_report['step']==196020
        source=HANDOFF/'capacity.pt'; sha=review_report['checkpoint_sha256']
        assert (OUTPUT/'wandb_run_id').is_file()
        resume_args=['--resume',str(source)]
    assert file_sha256(source)==sha
    os.environ['PYTHONPATH']=str(ROOT/'src')+os.pathsep+os.environ.get('PYTHONPATH','')
    os.environ['PYTHONFAULTHANDLER']='1'
    os.environ['WANDB_NAME']='h032-source-conditioned-camera-ray-196k-to210k'
    argv=[sys.executable,'-m','worldbridge.trainer.soft_torchrun','--standalone','--nproc-per-node=2',
          '--log-dir',str(OUTPUT)+('-capacity-elastic' if capacity else '-elastic'),'--tee','3',
          'scripts/train.py','--config',str(CONFIG),'--output-dir',str(OUTPUT),
          '--startup-preflight-updates','5',*resume_args]
    print(json.dumps(dict(event='H032_GPU_BEGIN',capacity=capacity,physical_gpus=gpu_ids,argv=argv)),flush=True)
    # PRL SIGTERM goes to the whole group; supervisor waits while trainer saves.
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,lambda *_:None)
    child=subprocess.Popen(argv,cwd=ROOT)
    code=child.wait()
    raise SystemExit(code if code>=0 else 128-code)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--mode',required=True,choices=['gate','capacity','continue','review-capacity','review-endpoint'])
    p.add_argument('--gpus',help='exact two user-confirmed physical IDs, also declare PRL resources')
    args=p.parse_args(); torch.set_num_threads(2)
    if args.mode.startswith('review'):
        assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
        review(196020 if args.mode=='review-capacity' else 210000)
    else:
        assert args.gpus is not None
        ids=args.gpus.split(','); assert len(ids)==len(set(ids))==2 and all(i.isdigit() for i in ids)
        if args.mode=='gate': gate(ids)
        else: train(ids,args.mode=='capacity')
if __name__=='__main__':main()
