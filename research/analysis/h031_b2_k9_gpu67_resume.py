"""User-authorized B2/A2/K9 GPU6/7 full175361 recovery to200000."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import yaml
from worldbridge.data.cache.native import file_sha256
from worldbridge.trainer.config import validate_config
from worldbridge.utils.io import atomic_json

CONFIG = 'configs/h031_k512_b2_a2_k9_mix50_gpu67_to200000.yaml'
CONFIG_SHA = 'd7ce6285650e73164794373c2f3ea09b14fade61a4bf967b4e6569e49253d916'
PARENT_CONFIG = 'configs/h031_k512_k3_mix50_gpu56_to200000.yaml'
PARENT_CONFIG_SHA = '710b3137e0c7361c2270dd9f0d47f7e6a67a5d86d310e161e6069128b8804ae6'
SOURCE = Path('/data/WorldBridge4D-runs/h031-k512-k3-mix50-169500-to200000-gpu56-20260912/checkpoint-0175361.pt')
SHA = 'b3b965cffb7e1fb11da88bc4bdd4c0592a61d690ef0e5793134651f13d3d43be'
HANDOFF = Path('/data/WorldBridge4D-runs/h031-b2-k9-gpu67-175361-handoff-20260913')
OUTPUT = Path('/data/WorldBridge4D-runs/h031-k512-b2-a2-k9-mix50-175361-to200000-gpu67-20260913')
UUIDS = ['GPU-eb70590e-6978-b98c-eefc-0d158607d190', 'GPU-b78e022b-c84d-ef28-d57e-2b6737b62481']

def health():
    raw = subprocess.check_output(['nvidia-smi', '-i', '6,7', '-q', '-x'], timeout=20)
    gpus = ET.fromstring(raw).findall('gpu')
    assert [g.findtext('uuid') for g in gpus] == UUIDS
    for g in gpus:
        for key in ('dram_uncorrectable', 'sram_uncorrectable_parity', 'sram_uncorrectable_secded'):
            assert g.findtext('ecc_errors/volatile/' + key) == '0', (g.findtext('uuid'), key)
        for key in ('remapped_row_pending', 'remapped_row_failure'):
            assert g.findtext('remapped_rows/' + key) == 'No', (g.findtext('uuid'), key)
    return raw.decode()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--gate',action='store_true'); args=p.parse_args()
    assert file_sha256(CONFIG)==CONFIG_SHA and file_sha256(PARENT_CONFIG)==PARENT_CONFIG_SHA
    cfg=yaml.safe_load(Path(CONFIG).read_text()); validate_config(cfg,2)
    old=yaml.safe_load(Path(PARENT_CONFIG).read_text())
    assert {k for k in cfg.keys() | old.keys() if cfg.get(k)!=old.get(k)} == {
        'native_kubric512_k3_mix_200k','native_kubric512_k3_mix_resume',
        'native_kubric512_b2_a2_k9_mix_200k','microbatch_per_gpu',
        'gradient_accumulation','targets_per_source','tracking'}
    assert not OUTPUT.exists()
    assert shutil.disk_usage(OUTPUT.parent).free >= 68719476736+38400000000
    if args.gate:
        assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
        subprocess.run([sys.executable,'-m','pytest','-q','tests/test_native_b2_k9_200k.py',
            'tests/test_native_k3_200k.py','tests/test_native_k3_mix_resume.py',
            'tests/test_native_k5_mix_resume.py','tests/test_native_k11.py',
            'tests/test_native_k9_170k.py','tests/test_native_k9_mix_trial.py',
            'tests/test_native512_capacity.py','tests/test_native512_k9.py',
            'tests/test_native512_long.py','tests/test_training256.py',
            'tests/test_startup_preflight.py','tests/test_soft_torchrun.py'],check=True)
        assert SOURCE.is_file() and not SOURCE.is_symlink()
        HANDOFF.mkdir(exist_ok=False)
        subprocess.run([sys.executable,'research/analysis/h031_periodic_checkpoint_review.py',
            '--checkpoint',str(SOURCE),'--step','175361','--config',PARENT_CONFIG,
            '--report',str(HANDOFF/'source_review.json')],check=True)
        review=json.loads((HANDOFF/'source_review.json').read_text())
        assert review['checkpoint_sha256']==SHA and review['global_step']==175361
        assert review['config_sha256']==PARENT_CONFIG_SHA
        assert review['Adam_states']==193 and review['RNG_ranks']==review['world_size']==2
        assert review['optimizer_steps_incremented']==22593
        (HANDOFF/'gpu_health.xml').write_text(health())
        (HANDOFF/'resume.pt').symlink_to(SOURCE)
        atomic_json(HANDOFF/'train_status.json',dict(completed_steps=175361,world_size=2))
        import torch
        report=dict(event='H031_B2_K9_GPU67_GATE_OK',checkpoint_sha256=SHA,config_sha256=CONFIG_SHA,
            source=str(SOURCE),resume_step=175361,target_step=200000,remaining_updates=24639,
            B=2,A=2,K=9,world_size=2,clips_per_update=8,pairs_per_update=72,physical_gpus=[6,7],
            seed=cfg['seed'],decoder_seed=cfg['decoder_seed'],fullstate_review=review,
            python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,platform=platform.platform(),
            environment={k:os.environ.get(k) for k in ('PYTHONPATH','OMP_NUM_THREADS','MKL_NUM_THREADS',
                'TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS')},
            capacity='unproven B2K9 native512;76GiB admission each, no fallback or allocator change',
            scientific_delta='B1A4K3_toB2A2K9;same model Adam RNG counters LR,not exact sampling/numerical replay',
            health_scope='UUID volatile ECC remap snapshot, not future hardware certification')
        atomic_json(HANDOFF/'complete.json',report); print(json.dumps(report),flush=True)
        return
    assert os.environ['CUDA_VISIBLE_DEVICES']=='6,7' and os.environ['PYTHONFAULTHANDLER']=='1'
    assert all(k not in os.environ for k in ('PYTORCH_CUDA_ALLOC_CONF','PYTORCH_ALLOC_CONF','PYTORCH_NO_CUDA_MEMORY_CACHING'))
    gate=json.loads((HANDOFF/'complete.json').read_text())
    assert gate['checkpoint_sha256']==SHA and gate['config_sha256']==CONFIG_SHA
    assert file_sha256(HANDOFF/'resume.pt')==SHA
    (HANDOFF/'gpu_health_at_launch.xml').write_text(health())
    print(json.dumps(dict(event='H031_B2_K9_GPU67_BEGIN',gate=gate)),flush=True)
    os.execv(sys.executable,[sys.executable,'-m','worldbridge.trainer.soft_torchrun',
        '--standalone','--nproc-per-node=2','--log-dir',str(OUTPUT)+'-elastic','--tee','3',
        'scripts/train.py','--config',CONFIG,'--output-dir',str(OUTPUT),
        '--resume',str(HANDOFF/'resume.pt'),'--startup-preflight-updates','5',
        '--input-readiness','--input-readiness-timeout-seconds','900'])

if __name__=='__main__': main()
