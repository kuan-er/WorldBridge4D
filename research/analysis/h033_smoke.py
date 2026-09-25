"""H033 bounded GPU smoke: migrate from the H032 200k checkpoint and train 20 updates.

This is the code-path rehearsal for the real H033 launch. It exercises exactly
the machinery the endpoint run will use, only from the newest available parent:

* strict structural migration (drop ``camera_head.*`` params and Adam states,
  add the decoder-native camera parameters),
* the collapsed single-profile validator and the cycle-free data path across a
  full 20-update mixture cycle (Kubric512 + DR512 + PO256),
* the camera token + ray-field objective, BF16 compute with FP32 master weights,
  two-rank FSDP on the explicitly authorized GPUs, and checkpoint saving.

It is not a scientific result: 20 updates, W&B disabled, its own output dir.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET
import yaml
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h033_make_config import make_config
from worldbridge.data.cache.native import file_sha256
from worldbridge.trainer.config import validate_config
from worldbridge.utils.io import atomic_json

PARENT = Path('/data/WorldBridge4D-runs/h032-source-camera-ray-k10-to210000-20260924')
PARENT_STEP = 200000
PARENT_CHECKPOINT = PARENT/f'checkpoint-{PARENT_STEP:07d}.pt'
UPDATES = 20
END = PARENT_STEP + UPDATES
CONFIG = ROOT/f'configs/h033_smoke_{PARENT_STEP}_to_{END}.yaml'
OUTPUT = Path(f'/data/WorldBridge4D-runs/h033-smoke-gpu35-{PARENT_STEP}-20260925')
EXPECTATION = OUTPUT.parent/f'h033-smoke-parent-{PARENT_STEP}.json'
ALLOCATOR_KEYS = ('PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF', 'PYTORCH_NO_CUDA_MEMORY_CACHING')


def parent_state():
    import torch
    payload = torch.load(PARENT_CHECKPOINT, map_location='cpu', mmap=True, weights_only=True)
    state = payload['training_state']
    report = dict(step=int(state['global_step']), world_size=int(state['world_size']),
                  clips_seen={k: int(v) for k, v in state['clips_seen'].items()},
                  optimizer_states=len(payload['optimizer']['state']),
                  old_camera_tensors=sum(1 for key in payload['model'] if key.startswith('camera_head.')),
                  checkpoint_sha256=file_sha256(PARENT_CHECKPOINT))
    del payload
    return report


def configuration(parent):
    cfg = make_config(clips_seen=parent['clips_seen'])
    cfg['finetune_expected_global_step'] = PARENT_STEP
    cfg['max_steps'] = END
    cfg['checkpoint_steps'] = []
    cfg['tracking']['enabled'] = False
    return cfg


def health(gpu_ids, allow_degraded=False):
    raw = subprocess.check_output(['nvidia-smi', '-i', ','.join(gpu_ids), '-q', '-x'], timeout=20)
    gpus = ET.fromstring(raw).findall('gpu')
    uuids = [g.findtext('uuid') for g in gpus]
    if len(uuids) != 2:
        raise RuntimeError(f'expected two GPUs, got {uuids}')
    degraded = []
    for gpu in gpus:
        counters = {}
        for key in ('dram_uncorrectable', 'sram_uncorrectable_parity', 'sram_uncorrectable_secded'):
            counters[f'ecc_errors/volatile/{key}'] = gpu.findtext('ecc_errors/volatile/'+key)
        for key in ('remapped_row_pending', 'remapped_row_failure', 'remapped_row_unc'):
            counters[f'remapped_rows/{key}'] = gpu.findtext('remapped_rows/'+key)
        unhealthy = (counters['ecc_errors/volatile/dram_uncorrectable'] not in ('0', 'N/A')
                     or counters['ecc_errors/volatile/sram_uncorrectable_parity'] not in ('0', 'N/A')
                     or counters['ecc_errors/volatile/sram_uncorrectable_secded'] not in ('0', 'N/A')
                     or counters['remapped_rows/remapped_row_pending'] != 'No'
                     or counters['remapped_rows/remapped_row_failure'] != 'No')
        if unhealthy:
            degraded.append(dict(uuid=gpu.findtext('uuid'), counters=counters, unhealthy=unhealthy))
    if degraded and not allow_degraded:
        raise RuntimeError(f'degraded GPU refused by the health gate: {json.dumps(degraded)}')
    return uuids, raw.decode(), degraded


def smoke(gpu_ids, allow_degraded=False):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ','.join(gpu_ids), 'declare the exact GPU ids'
    assert PARENT_CHECKPOINT.is_file(), f'missing parent checkpoint {PARENT_CHECKPOINT}'
    if OUTPUT.exists():
        leftovers = sorted(str(path.name) for path in OUTPUT.glob('checkpoint-*.pt')) \
            + sorted(str(path.name) for path in OUTPUT.glob('latest.pt'))
        if leftovers:
            raise RuntimeError(f'{OUTPUT} already holds checkpoints from another attempt: {leftovers}')
        print(json.dumps(dict(event='H033_SMOKE_REUSE_EMPTY_OUTPUT', output=str(OUTPUT))), flush=True)
    parent = parent_state()
    assert parent['step'] == PARENT_STEP, parent
    assert parent['old_camera_tensors'] == 51, parent
    cfg = configuration(parent)
    validate_config(cfg, 2)
    CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False))
    atomic_json(EXPECTATION, parent)
    uuids, xml, degraded = health(gpu_ids, allow_degraded=allow_degraded)
    if degraded:
        print(json.dumps(dict(event='H033_SMOKE_DEGRADED_GPU_OVERRIDE', degraded=degraded,
                              note='operator explicitly accepted these GPUs; numeric results are not '
                                   'usable as quality or performance evidence')), flush=True)
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT/'gpu_health.xml').write_text(xml)
    os.environ['PYTHONPATH'] = str(ROOT/'src') + os.pathsep + os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONFAULTHANDLER'] = '1'
    os.environ['WANDB_MODE'] = 'disabled'
    argv = [sys.executable, '-m', 'worldbridge.trainer.soft_torchrun', '--standalone',
            '--nproc-per-node=2', '--log-dir', str(OUTPUT)+'-elastic', '--tee', '3',
            'scripts/train.py', '--config', str(CONFIG), '--output-dir', str(OUTPUT),
            '--checkpoint-dir', str(OUTPUT), '--finetune-from', str(PARENT_CHECKPOINT),
            '--stop-after-updates', str(UPDATES), '--disable-wandb',
            '--startup-preflight-updates', '5']
    print(json.dumps(dict(event='H033_SMOKE_BEGIN', physical_gpus=gpu_ids, gpu_uuids=uuids,
                          parent=parent, config=str(CONFIG), output=str(OUTPUT),
                          degraded_gpus=degraded,
                          config_sha256=file_sha256(CONFIG), argv=argv,
                          allocator_env={k: os.environ.get(k) for k in ALLOCATOR_KEYS})), flush=True)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: None)
    child = subprocess.Popen(argv, cwd=ROOT)
    code = child.wait()
    print(json.dumps(dict(event='H033_SMOKE_DONE', exit_code=code,
                          checkpoint=str(OUTPUT/f'checkpoint-{END:07d}.pt'),
                          exists=(OUTPUT/f'checkpoint-{END:07d}.pt').is_file())), flush=True)
    raise SystemExit(code if code >= 0 else 128 - code)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['smoke'])
    parser.add_argument('--gpus', required=True, help='exact authorized physical ids, e.g. 3,5')
    parser.add_argument('--allow-degraded-gpus', action='store_true',
                        help='explicit operator opt-in for GPUs the ECC gate rejects; logged loudly')
    args = parser.parse_args()
    ids = args.gpus.split(',')
    assert len(ids) == len(set(ids)) == 2 and all(value.isdigit() for value in ids), args.gpus
    smoke(ids, allow_degraded=args.allow_degraded_gpus)


if __name__ == '__main__':
    main()
