"""Native512 FULL-mode capacity probe on physical GPUs1/2 — fresh, bounded3 updates."""
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

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(_ROOT / 'src'))

from worldbridge.data.cache.native import file_sha256
from worldbridge.trainer.config import validate_config
from worldbridge.utils.io import atomic_json

CONFIG = 'configs/h031_k512_full_k9_capacity.yaml'
CONFIG_SHA = 'c430aeb1231c22dfa1d21538e43c9e01a280e62b4b322e5676bac2cf3ff88bb3'
HANDOFF = Path('/data/WorldBridge4D-runs/h031-k512-full-k9-gpu12-handoff-20260917')
OUTPUT = Path('/data/WorldBridge4D-runs/h031-k512-full-k9-capacity-gpu12-20260917')
UUIDS = ['GPU-e5923570-eb29-25bc-46f2-298f98a1706b', 'GPU-3bf40a3a-b0df-afe3-7ed5-83c630dabe7a']

def health():
    raw = subprocess.check_output(['nvidia-smi', '-i', '1,2', '-q', '-x'], timeout=20)
    gpus = ET.fromstring(raw).findall('gpu')
    assert [g.findtext('uuid') for g in gpus] == UUIDS
    for g in gpus:
        for key in ('dram_uncorrectable', 'sram_uncorrectable_parity', 'sram_uncorrectable_secded'):
            assert g.findtext('ecc_errors/volatile/' + key) == '0', (g.findtext('uuid'), key)
        for key in ('remapped_row_pending', 'remapped_row_failure'):
            assert g.findtext('remapped_rows/' + key) == 'No', (g.findtext('uuid'), key)
    return raw.decode()

def main():
    p = argparse.ArgumentParser(); p.add_argument('--gate', action='store_true'); args = p.parse_args()
    assert file_sha256(CONFIG) == CONFIG_SHA
    cfg = yaml.safe_load(Path(CONFIG).read_text()); validate_config(cfg, 2)
    assert not OUTPUT.exists()
    assert shutil.disk_usage(OUTPUT.parent).free >= 68719476736 + 38400000000
    if args.gate:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        subprocess.run([sys.executable, '-m', 'pytest', '-q',
            'tests/test_native512_full.py', 'tests/test_native_b1_k11_200k.py',
            'tests/test_native_b1_k9_200k.py', 'tests/test_native_b2_k5_200k.py',
            'tests/test_native_b2_k9_200k.py', 'tests/test_native_k3_200k.py',
            'tests/test_native_k3_mix_resume.py', 'tests/test_native_k5_mix_resume.py',
            'tests/test_native_k11.py', 'tests/test_native_k9_170k.py',
            'tests/test_native_k9_mix_trial.py', 'tests/test_native512_capacity.py',
            'tests/test_native512_k9.py', 'tests/test_native512_long.py',
            'tests/test_training256.py', 'tests/test_startup_preflight.py',
            'tests/test_soft_torchrun.py', 'tests/test_fp32_master.py'], check=True)
        HANDOFF.mkdir(exist_ok=False)
        (HANDOFF / 'gpu_health.xml').write_text(health())
        import torch
        report = dict(event='H031_K512_FULL_K9_GPU12_GATE_OK', config_sha256=CONFIG_SHA,
            output=str(OUTPUT), max_steps=cfg['max_steps'], B=1, A=4, K=9, world_size=2,
            clips_per_update=8, pairs_per_update=72, physical_gpus=[1, 2],
            trainable_mode=cfg['trainable_mode'], precision=cfg['precision'],
            fsdp_master_precision=cfg['fsdp_master_precision'],
            seed=cfg['seed'], decoder_seed=cfg['decoder_seed'],
            python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
            platform=platform.platform(),
            capacity='fresh full-mode native512 K3 probe; FP32 master, no resume, --no-checkpoint',
            scientific_delta='trainable_mode decoder_only->full over native512; fp32 master, pre-attn-RGB off, cosine, cycle0',
            health_scope='UUID volatile ECC remap snapshot, not future hardware certification')
        atomic_json(HANDOFF / 'complete.json', report); print(json.dumps(report), flush=True)
        return
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1,2'
    os.environ['PYTHONFAULTHANDLER'] = '1'
    os.environ['PYTHONPATH'] = str(_ROOT / 'src') + (os.pathsep + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else '')
    assert all(k not in os.environ for k in ('PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF', 'PYTORCH_NO_CUDA_MEMORY_CACHING'))
    gate = json.loads((HANDOFF / 'complete.json').read_text())
    assert gate['config_sha256'] == CONFIG_SHA
    assert file_sha256(CONFIG) == CONFIG_SHA
    (HANDOFF / 'gpu_health_at_launch.xml').write_text(health())
    print(json.dumps(dict(event='H031_K512_FULL_K3_GPU12_BEGIN', gate=gate)), flush=True)
    os.execv(sys.executable, [sys.executable, '-m', 'worldbridge.trainer.soft_torchrun',
        '--standalone', '--nproc-per-node=2', '--log-dir', str(OUTPUT) + '-elastic', '--tee', '3',
        'scripts/train.py', '--config', CONFIG, '--output-dir', str(OUTPUT),
        '--startup-preflight-updates', '3', '--no-checkpoint'])

if __name__ == '__main__':
    main()
