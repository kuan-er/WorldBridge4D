"""Authorized GPU5/6 full169500 K3 recovery to200000; no automatic retry."""
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

CONFIG = 'configs/h031_k512_k3_mix50_gpu56_to200000.yaml'
CONFIG_SHA = '710b3137e0c7361c2270dd9f0d47f7e6a67a5d86d310e161e6069128b8804ae6'
PARENT_CONFIG = 'configs/h031_k512_k3_mix50_gpu25_to170000.yaml'
PARENT_CONFIG_SHA = '667571c64f5943fab02ebb0ec50f44e0d000878843b66ef88e046ee3a64fc7fb'
SOURCE = Path('/data/WorldBridge4D-runs/h031-k512-k3-mix50-156000-to170000-gpu25-20260910/checkpoint-0169500.pt')
SHA = 'dfbf16c586ae3b15c054fe48e501b1246dc40c8104b44a82c5c839a05e8ba78b'
REVIEW = Path('/data/WorldBridge4D-runs/h031-k3-gpu25-post-timeout169500-review-20260912/complete.json')
HANDOFF = Path('/data/WorldBridge4D-runs/h031-k3-gpu56-169500-handoff-20260912')
OUTPUT = Path('/data/WorldBridge4D-runs/h031-k512-k3-mix50-169500-to200000-gpu56-20260912')
UUIDS = ['GPU-5e216d50-d919-1bb1-fa53-0192f2cf3101', 'GPU-eb70590e-6978-b98c-eefc-0d158607d190']

def health():
    raw = subprocess.check_output(['nvidia-smi', '-i', '5,6', '-q', '-x'], timeout=20)
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
    assert file_sha256(CONFIG) == CONFIG_SHA and file_sha256(PARENT_CONFIG) == PARENT_CONFIG_SHA
    cfg = yaml.safe_load(Path(CONFIG).read_text()); validate_config(cfg, 2)
    parent = yaml.safe_load(Path(PARENT_CONFIG).read_text())
    assert {k for k in cfg.keys() | parent.keys() if cfg.get(k) != parent.get(k)} == {
        'native_kubric512_k3_mix_200k', 'max_steps', 'lr_restart', 'checkpoint_steps',
        'graceful_stop_hours', 'tracking'}
    assert shutil.disk_usage(OUTPUT.parent).free >= 68719476736 + 38400000000
    assert not OUTPUT.exists()
    if args.gate:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_native_k3_200k.py',
                        'tests/test_native_k3_mix_resume.py', 'tests/test_native_k5_mix_resume.py',
                        'tests/test_native_k11.py', 'tests/test_native_k9_170k.py',
                        'tests/test_native_k9_mix_trial.py', 'tests/test_native512_long.py',
                        'tests/test_training256.py', 'tests/test_startup_preflight.py',
                        'tests/test_soft_torchrun.py'], check=True)
        review = json.loads(REVIEW.read_text())
        assert review['event'] == 'H031_PERIODIC_FULL_STATE_OK'
        assert review['checkpoint_sha256'] == SHA and review['global_step'] == 169500
        assert review['config_sha256'] == PARENT_CONFIG_SHA
        assert review['Adam_states'] == 193 and review['RNG_ranks'] == review['world_size'] == 2
        assert review['optimizer_steps_incremented'] == 16732
        assert SOURCE.is_file() and not SOURCE.is_symlink() and file_sha256(SOURCE) == SHA
        assert SOURCE.stat().st_size == review['checkpoint_bytes'] == 7864620199
        xml = health()
        import torch
        HANDOFF.mkdir(exist_ok=False)
        (HANDOFF / 'resume.pt').symlink_to(SOURCE)
        atomic_json(HANDOFF / 'train_status.json', dict(completed_steps=169500, world_size=2))
        (HANDOFF / 'gpu_health.xml').write_text(xml)
        report = dict(event='H031_K3_GPU56_GATE_OK', source=str(SOURCE), checkpoint_sha256=SHA,
                      config_sha256=CONFIG_SHA, resume_step=169500, target_step=200000,
                      seed=cfg['seed'], decoder_seed=cfg['decoder_seed'], physical_gpus=[5,6],
                      K=3, clips_per_update=8, pairs_per_update=24, remaining_updates=30500,
                      unsaved_updates_to_replay=139, fullstate_review=review,
                      python=sys.version, platform=platform.platform(), torch=torch.__version__,
                      cuda=torch.version.cuda, executable=sys.executable,
                      environment={k: os.environ.get(k) for k in ('PYTHONPATH', 'OMP_NUM_THREADS',
                                   'MKL_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS')},
                      health_scope='NVML snapshot not hardware certification',
                      capacity='48GiB free admission each; not a reservation against unmanaged jobs',
                      known_risk='prior Gloo readiness timeout root cause unresolved; unchanged900s',
                      walltime='168h checkpoint-first ceiling for extended horizon, no LR restart')
        atomic_json(HANDOFF / 'complete.json', report)
        print(json.dumps(report), flush=True)
        return
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '5,6'
    assert os.environ['PYTHONFAULTHANDLER'] == '1'
    for key in ('PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF', 'PYTORCH_NO_CUDA_MEMORY_CACHING'):
        assert key not in os.environ
    gate = json.loads((HANDOFF / 'complete.json').read_text())
    assert gate['checkpoint_sha256'] == SHA and gate['config_sha256'] == CONFIG_SHA
    assert file_sha256(HANDOFF / 'resume.pt') == SHA
    (HANDOFF / 'gpu_health_at_launch.xml').write_text(health())
    print(json.dumps(dict(event='H031_K3_GPU56_BEGIN', gate=gate,
                          change='GPU25_to56_full169500_K3_hold_extended200000',
                          caveat='same full model Adam RNG counters; no cross-device bitwise guarantee')), flush=True)
    os.execv(sys.executable, [sys.executable, '-m', 'worldbridge.trainer.soft_torchrun',
        '--standalone', '--nproc-per-node=2', '--log-dir', str(OUTPUT)+'-elastic', '--tee', '3',
        'scripts/train.py', '--config', CONFIG, '--output-dir', str(OUTPUT),
        '--resume', str(HANDOFF / 'resume.pt'), '--startup-preflight-updates', '5',
        '--input-readiness', '--input-readiness-timeout-seconds', '900'])

if __name__ == '__main__':
    main()
