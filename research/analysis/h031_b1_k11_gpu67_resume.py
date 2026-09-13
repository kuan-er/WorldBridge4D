"""User K9->K11 on GPU6/7 B1/A4, full175374 to200000; no operational changes."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import yaml
from worldbridge.data.cache.native import file_sha256
from worldbridge.trainer.config import validate_config
from worldbridge.utils.io import atomic_json
from h031_b1_k9_gpu67_resume import health

CONFIG='configs/h031_k512_b1_a4_k11_mix50_gpu67_to200000.yaml'
CONFIG_SHA='6f8836acc0a70864e2961f5056b724af06ae3a791fc43f1939794a5974cb10c7'
PARENT_CONFIG='configs/h031_k512_b1_a4_k9_mix50_gpu67_to200000.yaml'
PARENT_CONFIG_SHA='346819917764096102c0062a24581e8c88a10be914904e26e0995a7634400dae'
STEP=175374
SOURCE=Path('/data/WorldBridge4D-runs/h031-k512-b1-a4-k9-mix50-175361-to200000-gpu67-20260913/checkpoint-0175374.pt')
SHA='12abbe3cbb6f10587e8098fd67340d46a2384d2541abd368ec516fbc1b8713ce'
HANDOFF=Path('/data/WorldBridge4D-runs/h031-b1-k11-gpu67-175374-handoff-20260913')
OUTPUT=Path('/data/WorldBridge4D-runs/h031-k512-b1-a4-k11-mix50-175374-to200000-gpu67-20260913')

def main():
    p=argparse.ArgumentParser();p.add_argument('--gate',action='store_true');args=p.parse_args()
    assert file_sha256(CONFIG)==CONFIG_SHA and file_sha256(PARENT_CONFIG)==PARENT_CONFIG_SHA
    cfg=yaml.safe_load(Path(CONFIG).read_text());validate_config(cfg,2)
    old=yaml.safe_load(Path(PARENT_CONFIG).read_text())
    assert {k for k in cfg.keys() | old.keys() if cfg.get(k)!=old.get(k)}=={
        'native_kubric512_b1_a4_k9_mix_200k','native_kubric512_b1_a4_k11_mix_200k',
        'native_kubric512_k11_trial','targets_per_source','tracking'}
    assert not OUTPUT.exists()
    assert shutil.disk_usage(OUTPUT.parent).free>=68719476736+38400000000
    if args.gate:
        assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
        subprocess.run([sys.executable,'-m','pytest','-q','tests/test_native_b1_k11_200k.py',
            'tests/test_native_b1_k9_200k.py','tests/test_native_b2_k5_200k.py',
            'tests/test_native_b2_k9_200k.py','tests/test_native_k3_200k.py',
            'tests/test_native_k3_mix_resume.py','tests/test_native_k5_mix_resume.py',
            'tests/test_native_k11.py','tests/test_native_k9_170k.py',
            'tests/test_native_k9_mix_trial.py','tests/test_native512_capacity.py',
            'tests/test_native512_k9.py','tests/test_native512_long.py','tests/test_training256.py',
            'tests/test_startup_preflight.py','tests/test_soft_torchrun.py'],check=True)
        assert SOURCE.is_file() and not SOURCE.is_symlink()
        HANDOFF.mkdir(exist_ok=False)
        subprocess.run([sys.executable,'research/analysis/h031_periodic_checkpoint_review.py',
            '--checkpoint',str(SOURCE),'--step',str(STEP),'--config',PARENT_CONFIG,
            '--report',str(HANDOFF/'source_review.json')],check=True)
        review=json.loads((HANDOFF/'source_review.json').read_text())
        assert review['checkpoint_sha256']==SHA and review['global_step']==STEP
        assert review['config_sha256']==PARENT_CONFIG_SHA
        assert review['Adam_states']==193 and review['RNG_ranks']==review['world_size']==2
        assert review['optimizer_steps_incremented']==STEP-152768
        (HANDOFF/'gpu_health.xml').write_text(health())
        (HANDOFF/'resume.pt').symlink_to(SOURCE)
        atomic_json(HANDOFF/'train_status.json',dict(completed_steps=STEP,world_size=2))
        import torch
        report=dict(event='H031_B1_K11_GPU67_GATE_OK',checkpoint_sha256=SHA,config_sha256=CONFIG_SHA,
            source=str(SOURCE),resume_step=STEP,target_step=200000,remaining_updates=200000-STEP,
            B=1,A=4,K=11,world_size=2,clips_per_update=8,pairs_per_update=88,physical_gpus=[6,7],
            seed=cfg['seed'],decoder_seed=cfg['decoder_seed'],fullstate_review=review,
            python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,platform=platform.platform(),
            environment={k:os.environ.get(k) for k in ('PYTHONPATH','OMP_NUM_THREADS','MKL_NUM_THREADS',
                'TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS')},
            capacity='B1K11 native512 requires actual complete update;76GiB admission each, no fallback or allocator change',
            scientific_delta='user_K9_toK11;same B1A4 full175374 model Adam RNG counters LR,not exact target sampling replay',
            health_scope='UUID volatile ECC remap snapshot, not future hardware certification')
        atomic_json(HANDOFF/'complete.json',report);print(json.dumps(report),flush=True)
        return
    assert os.environ['CUDA_VISIBLE_DEVICES']=='6,7' and os.environ['PYTHONFAULTHANDLER']=='1'
    assert all(k not in os.environ for k in ('PYTORCH_CUDA_ALLOC_CONF','PYTORCH_ALLOC_CONF','PYTORCH_NO_CUDA_MEMORY_CACHING'))
    gate=json.loads((HANDOFF/'complete.json').read_text())
    assert gate['checkpoint_sha256']==SHA and gate['config_sha256']==CONFIG_SHA
    assert file_sha256(HANDOFF/'resume.pt')==SHA
    (HANDOFF/'gpu_health_at_launch.xml').write_text(health())
    print(json.dumps(dict(event='H031_B1_K11_GPU67_BEGIN',gate=gate)),flush=True)
    os.execv(sys.executable,[sys.executable,'-m','worldbridge.trainer.soft_torchrun',
        '--standalone','--nproc-per-node=2','--log-dir',str(OUTPUT)+'-elastic','--tee','3',
        'scripts/train.py','--config',CONFIG,'--output-dir',str(OUTPUT),
        '--resume',str(HANDOFF/'resume.pt'),'--startup-preflight-updates','5',
        '--input-readiness','--input-readiness-timeout-seconds','900'])

if __name__=='__main__':main()
