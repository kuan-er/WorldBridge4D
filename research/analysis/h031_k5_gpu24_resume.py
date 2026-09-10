"""User-authorized K5 GPU2/4 recovery, full155000; no retry loop or device fallback."""
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

CONFIG = 'configs/h031_k512_k5_mix50_gpu24_to170000.yaml'
SOURCE = Path('/data/WorldBridge4D-runs/h031-k512-k11-mix50-prefix5-nostalltrace-to170000-gpu23-20260909/checkpoint-0155000.pt')
SHA = 'db68a0dc7767a56f97e179dde562afdf63aaf19a0b0dfaf23344aeab801bc05e'
REVIEW = Path('/data/WorldBridge4D-runs/h031-post-ecc155000-review-20260909/complete.json')
HANDOFF = Path('/data/WorldBridge4D-runs/h031-k5-gpu24-handoff-20260910')
OUTPUT = Path('/data/WorldBridge4D-runs/h031-k512-k5-mix50-155000-to170000-gpu24-20260910')
UUIDS = ['GPU-3bf40a3a-b0df-afe3-7ed5-83c630dabe7a', 'GPU-25430d5b-58a4-d1c1-701b-ade48355308f']

def health():
    raw = subprocess.check_output(['nvidia-smi', '-i', '2,4', '-q', '-x'], timeout=20)
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
    cfg = yaml.safe_load(Path(CONFIG).read_text()); validate_config(cfg, 2)
    old = yaml.safe_load(Path('configs/h031_k512_k11_mix50_prefix5_nostalltrace_to170000.yaml').read_text())
    assert {k for k in cfg.keys() | old.keys() if cfg.get(k) != old.get(k)} == {
        'native_kubric512_k11_trial', 'native_kubric512_k5_mix_resume', 'targets_per_source', 'tracking'}
    assert shutil.disk_usage(OUTPUT.parent).free >= 68719476736 + 38400000000
    if args.gate:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        subprocess.run([sys.executable, '-m', 'pytest', '-q',
                        'tests/test_native_k5_mix_resume.py', 'tests/test_native_k11.py',
                        'tests/test_native_k9_170k.py', 'tests/test_native_k9_mix_trial.py',
                        'tests/test_startup_preflight.py', 'tests/test_native512_long.py',
                        'tests/test_training256.py', 'tests/test_soft_torchrun.py'], check=True)
        review = json.loads(REVIEW.read_text())
        assert review['event'] == 'H031_PERIODIC_FULL_STATE_OK'
        assert review['checkpoint_sha256'] == SHA and review['global_step'] == 155000
        assert review['Adam_states'] == 193 and review['RNG_ranks'] == review['world_size'] == 2
        assert review['optimizer_steps_incremented'] == 2232
        assert SOURCE.is_file() and file_sha256(SOURCE) == SHA
        assert not OUTPUT.exists()
        xml = health()
        HANDOFF.mkdir(exist_ok=False)
        (HANDOFF / 'resume.pt').symlink_to(SOURCE)
        atomic_json(HANDOFF / 'train_status.json', dict(completed_steps=155000, world_size=2))
        (HANDOFF / 'gpu_health.xml').write_text(xml)
        report = dict(event='H031_K5_GPU24_GATE_OK', source=str(SOURCE), checkpoint_sha256=SHA,
                      config_sha256=file_sha256(CONFIG), resume_step=155000, target_step=170000,
                      seed=cfg['seed'], decoder_seed=cfg['decoder_seed'], physical_gpus=[2,4],
                      K=5, clips_per_update=8, pairs_per_update=40, unsaved_updates_to_replay=273,
                      fullstate_review=review, health_scope='NVML admission not exhaustive hardware certification')
        atomic_json(HANDOFF / 'complete.json', report)
        print(json.dumps(report), flush=True)
        return
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '2,4'
    assert os.environ['PYTHONFAULTHANDLER'] == '1'
    gate = json.loads((HANDOFF / 'complete.json').read_text())
    assert gate['checkpoint_sha256'] == SHA and gate['config_sha256'] == file_sha256(CONFIG)
    assert file_sha256(HANDOFF / 'resume.pt') == SHA
    assert not OUTPUT.exists()
    xml = health()
    (HANDOFF / 'gpu_health_at_launch.xml').write_text(xml)
    print(json.dumps(dict(event='H031_K5_GPU24_BEGIN', gate=gate,
                          change='K11_toK5_and_physical_GPU23_to24_fullstate_no_new_warmup',
                          caveat='K_changes_query_eligibility_and_RNG_not_exact_K11_replay')), flush=True)
    os.execv(sys.executable, [sys.executable, '-m', 'worldbridge.trainer.soft_torchrun',
        '--standalone', '--nproc-per-node=2', '--log-dir', str(OUTPUT)+'-elastic', '--tee', '3',
        'scripts/train.py', '--config', CONFIG, '--output-dir', str(OUTPUT),
        '--resume', str(HANDOFF / 'resume.pt'), '--startup-preflight-updates', '5',
        '--input-readiness', '--input-readiness-timeout-seconds', '900'])

if __name__ == '__main__':
    main()
