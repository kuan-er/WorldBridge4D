"""Conservative full-mode unfreeze from 183000: K512+DR512+PO256, 10k steps."""
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
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from worldbridge.data.cache.native import file_sha256
from worldbridge.trainer.config import validate_config
from worldbridge.utils.io import atomic_json

CONFIG = 'configs/h031_k512_dr512_full_k9_unfreeze_183000.yaml'
CONFIG_SHA = '7928fd682a0d6f0d5a0b90f25ed367fd5a08fe3a984089c36cb4a4feb730cbb3'
SOURCE_CONFIG = 'configs/h031_k512_b1_a4_k11_mix50_gpu67_to200000.yaml'
SOURCE_CONFIG_SHA = '6f8836acc0a70864e2961f5056b724af06ae3a791fc43f1939794a5974cb10c7'
STEP = 183000
SOURCE = Path('/data/WorldBridge4D-runs/h031-k512-b1-a4-k11-mix50-175374-to200000-gpu67-20260913/checkpoint-0183000.pt')
SHA = '164320e810e4bf8a2d68ed9e3d5962e87d05f69bb62499a1d456d614264ebb14'
HANDOFF = Path('/data/WorldBridge4D-runs/h031-k512-dr512-full-k9-unfreeze-lowlr-183000-gpu16-r3-handoff-20260919')
OUTPUT = Path('/data/WorldBridge4D-runs/h031-k512-dr512-full-k9-unfreeze-lowlr-183000-to193000-gpu16-r3-20260919')
UUIDS = ['GPU-e5923570-eb29-25bc-46f2-298f98a1706b', 'GPU-eb70590e-6978-b98c-eefc-0d158607d190']


def health():
    raw = subprocess.check_output(['nvidia-smi', '-i', '1,6', '-q', '-x'], timeout=20)
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
    restart_rates = cfg['lr_restart']['group_learning_rates']
    assert float(restart_rates['wan_backbone']) == 5e-7
    assert float(restart_rates['geometry_adapter']) == 3e-6
    assert int(cfg['lr_restart']['warmup_steps']) == 500
    assert not OUTPUT.exists()
    assert shutil.disk_usage(OUTPUT.parent).free >= 68719476736 + 38400000000
    if args.gate:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['PYTHONPATH'] = str(_ROOT / 'src') + (os.pathsep + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else '')
        subprocess.run([sys.executable, '-m', 'pytest', '-q',
            'tests/test_native512_full.py', 'tests/test_dr512_cache.py',
            'tests/test_native_b1_k11_200k.py', 'tests/test_native_b1_k9_200k.py',
            'tests/test_native_b2_k5_200k.py', 'tests/test_native_b2_k9_200k.py',
            'tests/test_native_k3_200k.py', 'tests/test_native_k3_mix_resume.py',
            'tests/test_native_k5_mix_resume.py', 'tests/test_native_k11.py',
            'tests/test_native_k9_170k.py', 'tests/test_native_k9_mix_trial.py',
            'tests/test_native512_capacity.py', 'tests/test_native512_k9.py',
            'tests/test_native512_long.py', 'tests/test_training256.py',
            'tests/test_startup_preflight.py', 'tests/test_soft_torchrun.py',
            'tests/test_fp32_master.py', 'tests/test_native_latents.py',
            'tests/test_native_rgb.py', 'tests/test_native_rgb_stream.py',
            'tests/test_native_gt_demand.py', 'tests/test_native_cache_pair.py'], check=True)
        assert SOURCE.is_file() and not SOURCE.is_symlink()
        HANDOFF.mkdir(exist_ok=False)
        subprocess.run([sys.executable, 'research/analysis/h031_periodic_checkpoint_review.py',
            '--checkpoint', str(SOURCE), '--step', str(STEP), '--config', SOURCE_CONFIG,
            '--report', str(HANDOFF / 'source_review.json')], check=True)
        review = json.loads((HANDOFF / 'source_review.json').read_text())
        assert review['checkpoint_sha256'] == SHA and review['global_step'] == STEP
        assert review['config_sha256'] == SOURCE_CONFIG_SHA
        assert review['Adam_states'] == 193 and review['RNG_ranks'] == review['world_size'] == 2
        # Finalize DR512 cache marker (bulk writes complete.json to OUTPUT; the
        # training adapter reads bulk_complete.json from the latent cache root).
        dr_out = Path('/data/WorldBridge4D-runs/h031-dr512-cache-20260919')
        dr_manifest = json.loads((dr_out / 'dynamic_replica.json').read_text())
        dr_complete = json.loads((dr_out / 'complete.json').read_text())
        assert dr_complete['manifest_sha256'] == dr_manifest['sha256']
        assert dr_complete['clips'] == len(dr_manifest['records']) == 6090
        dr_latent_root = Path('/data/WorldBridge4D-persistent/worldbridge4d-native-vae-v1/dynamic_replica') / dr_manifest['sha256']
        atomic_json(dr_latent_root / 'bulk_complete.json', {
            'manifest_sha256': dr_manifest['sha256'], 'processed': dr_complete['clips'],
            'all_existing_training_index_entries_verified': True, 'contract': dr_manifest['contract'],
        })
        dr_rgb_root = Path(cfg['datasets']['dynamic_replica']['native_rgb_root']) / 'dynamic_replica' / dr_manifest['sha256']
        atomic_json(dr_rgb_root / 'bulk_complete.json', {
            'manifest_sha256': dr_manifest['sha256'], 'processed': dr_complete['clips'],
            'requested': dr_complete['clips'], 'contract': 'native_rgb_uint8_snapshot_v1',
        })
        from worldbridge.data.datasets.native_dynamic_replica import NativeDynamicReplicaDataset
        dr = NativeDynamicReplicaDataset(cfg['datasets']['dynamic_replica'])
        assert len(dr) == 6090
        assert dr.rgb(0).shape == (21, 512, 512, 3)
        assert dr.clean_latent(0).shape == (16, 6, 64, 64)
        (HANDOFF / 'gpu_health.xml').write_text(health())
        (HANDOFF / 'resume.pt').symlink_to(SOURCE)
        atomic_json(HANDOFF / 'train_status.json', dict(completed_steps=STEP, world_size=2))
        import torch
        report = dict(event='H031_K512_DR512_FULL_K9_UNFREEZE_LOWLR_183000_GATE_OK',
            checkpoint_sha256=SHA, config_sha256=CONFIG_SHA, source=str(SOURCE),
            resume_step=STEP, target_step=193000, remaining_updates=193000 - STEP,
            B=1, A=4, K=9, world_size=2, clips_per_update=8, pairs_per_update=72,
            physical_gpus=[1, 6], trainable_mode=cfg['trainable_mode'],
            precision=cfg['precision'], fsdp_master_precision=cfg['fsdp_master_precision'],
            seed=cfg['seed'], decoder_seed=cfg['decoder_seed'], fullstate_review=review,
            python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
            platform=platform.platform(),
            scientific_delta='decoder_only->full unfreeze at183000; K512+DR512+PO256; filtered backbone optimizer; 500-step warmup to Wan5e-7/geometry3e-6/decoder3e-6/RGB3e-6',
            health_scope='UUID volatile ECC remap snapshot, not future hardware certification')
        atomic_json(HANDOFF / 'complete.json', report); print(json.dumps(report), flush=True)
        return
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1,6'
    os.environ['PYTHONFAULTHANDLER'] = '1'
    os.environ['PYTHONPATH'] = str(_ROOT / 'src') + (os.pathsep + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else '')
    gate = json.loads((HANDOFF / 'complete.json').read_text())
    assert gate['checkpoint_sha256'] == SHA and gate['config_sha256'] == CONFIG_SHA
    assert file_sha256(HANDOFF / 'resume.pt') == SHA
    (HANDOFF / 'gpu_health_at_launch.xml').write_text(health())
    print(json.dumps(dict(event='H031_K512_DR512_FULL_K9_UNFREEZE_LOWLR_183000_BEGIN', gate=gate)), flush=True)
    os.execv(sys.executable, [sys.executable, '-m', 'worldbridge.trainer.soft_torchrun',
        '--standalone', '--nproc-per-node=2', '--log-dir', str(OUTPUT) + '-elastic', '--tee', '3',
        'scripts/train.py', '--config', CONFIG, '--output-dir', str(OUTPUT),
        '--resume', str(HANDOFF / 'resume.pt'), '--startup-preflight-updates', '5'])


if __name__ == '__main__':
    main()
