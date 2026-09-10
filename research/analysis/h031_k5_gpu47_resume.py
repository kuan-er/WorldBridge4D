"""User-authorized physical4/7 recovery of full156000; unchanged K5 science."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import yaml
from worldbridge.data.cache.native import file_sha256
from worldbridge.trainer.config import validate_config
from worldbridge.utils.io import atomic_json

# Keep exact prior config bytes (gpu24 in filename/tracking is provenance, not device selection).
CONFIG = 'configs/h031_k512_k5_mix50_gpu24_to170000.yaml'
CONFIG_SHA = 'f7de3065a82ac3e31820ae83b63b5448697461b3b43d8791aa46fbed311cefd1'
SOURCE = Path('/data/WorldBridge4D-runs/h031-k512-k5-mix50-155000-to170000-gpu24-20260910/checkpoint-0156000.pt')
SHA = '3260b9f05c7297c40c926a6e9b7310be8007678450c05735b628668049dcc682'
REVIEW = Path('/data/WorldBridge4D-runs/h031-k5-gpu24-post-oom156000-review-20260910/complete.json')
HANDOFF = Path('/data/WorldBridge4D-runs/h031-k5-gpu47-handoff-20260910')
OUTPUT = Path('/data/WorldBridge4D-runs/h031-k512-k5-mix50-156000-to170000-gpu47-20260910')
UUIDS = ['GPU-25430d5b-58a4-d1c1-701b-ade48355308f', 'GPU-b78e022b-c84d-ef28-d57e-2b6737b62481']

def health():
    raw = subprocess.check_output(['nvidia-smi', '-i', '4,7', '-q', '-x'], timeout=20)
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
    assert shutil.disk_usage(OUTPUT.parent).free >= 68719476736 + 38400000000
    assert not OUTPUT.exists()
    if args.gate:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_native_k5_mix_resume.py',
                        'tests/test_startup_preflight.py', 'tests/test_soft_torchrun.py'], check=True)
        review = json.loads(REVIEW.read_text())
        assert review['event'] == 'H031_PERIODIC_FULL_STATE_OK'
        assert review['checkpoint_sha256'] == SHA and review['global_step'] == 156000
        assert review['config_sha256'] == CONFIG_SHA
        assert review['Adam_states'] == 193 and review['RNG_ranks'] == review['world_size'] == 2
        assert review['optimizer_steps_incremented'] == 3232
        assert SOURCE.is_file() and not SOURCE.is_symlink() and file_sha256(SOURCE) == SHA
        xml = health()
        HANDOFF.mkdir(exist_ok=False)
        (HANDOFF / 'resume.pt').symlink_to(SOURCE)
        atomic_json(HANDOFF / 'train_status.json', dict(completed_steps=156000, world_size=2))
        (HANDOFF / 'gpu_health.xml').write_text(xml)
        report = dict(event='H031_K5_GPU47_GATE_OK', source=str(SOURCE), checkpoint_sha256=SHA,
                      config_sha256=CONFIG_SHA, resume_step=156000, target_step=170000,
                      seed=cfg['seed'], decoder_seed=cfg['decoder_seed'], physical_gpus=[4,7],
                      K=5, clips_per_update=8, pairs_per_update=40, unsaved_updates_to_replay=499,
                      fullstate_review=review, health_scope='NVML snapshot not hardware certification',
                      capacity='64GiB free admission; not a reservation against unmanaged future jobs')
        atomic_json(HANDOFF / 'complete.json', report)
        print(json.dumps(report), flush=True)
        return
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '4,7'
    assert os.environ['PYTHONFAULTHANDLER'] == '1'
    for key in ('PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF', 'PYTORCH_NO_CUDA_MEMORY_CACHING'):
        assert key not in os.environ
    gate = json.loads((HANDOFF / 'complete.json').read_text())
    assert gate['checkpoint_sha256'] == SHA and gate['config_sha256'] == CONFIG_SHA
    assert file_sha256(HANDOFF / 'resume.pt') == SHA
    (HANDOFF / 'gpu_health_at_launch.xml').write_text(health())
    print(json.dumps(dict(event='H031_K5_GPU47_BEGIN', gate=gate,
                          change='physical_GPU24_to47_full156000_no_new_warmup_same_K5',
                          caveat='fullstate_resume_not_independently_proven_bitwise_replay')), flush=True)
    os.execv(sys.executable, [sys.executable, '-m', 'worldbridge.trainer.soft_torchrun',
        '--standalone', '--nproc-per-node=2', '--log-dir', str(OUTPUT)+'-elastic', '--tee', '3',
        'scripts/train.py', '--config', CONFIG, '--output-dir', str(OUTPUT),
        '--resume', str(HANDOFF / 'resume.pt'), '--startup-preflight-updates', '5',
        '--input-readiness', '--input-readiness-timeout-seconds', '900'])

if __name__ == '__main__':
    main()
