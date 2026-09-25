"""Full-state recovery; same W&B history; event-driven two-hour inspections."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from worldbridge.data.cache.native import file_sha256
from worldbridge.trainer.config import validate_config
from worldbridge.utils.io import atomic_json
from h031_k512_dr512_full_k9_unfreeze_183000 import health

BASE = 'configs/h031_k512_dr512_full_k9_unfreeze_183000.yaml'
CONFIG = 'configs/h031_k512_dr512_full_k9_resume191000_to200000.yaml'
SOURCE = Path('/data/WorldBridge4D-runs/h031-k512-dr512-full-k9-unfreeze-lowlr-183000-to193000-gpu16-r3-20260919/checkpoint-0191000.pt')
HANDOFF = Path('/data/WorldBridge4D-runs/h031-resume191000-to200000-gpu16-handoff-20260922')
OUTPUT = Path('/data/WorldBridge4D-runs/h031-resume191000-to200000-gpu16-20260922')
WANDB_ID = 'zypk1th1'
STEP = 191000
TARGET = 200000


def check_config():
    cfg = yaml.safe_load(Path(CONFIG).read_text())
    base = yaml.safe_load(Path(BASE).read_text())
    expected = yaml.safe_load(Path(BASE).read_text())
    expected['max_steps'] = TARGET
    expected['lr_restart']['end_step'] = TARGET
    expected['checkpoint_steps'] = list(range(192000, TARGET + 1, 1000))
    expected['checkpoint_every_after'] = 1000
    expected['tracking']['resume'] = 'must'
    assert cfg == expected
    assert cfg['lr_restart']['start_step'] == cfg['selected_checkpoint_step'] == 183000
    validate_config(cfg, 2)
    return cfg, base


def check_rng(rng):
    # FSDP saves capture_rng_state() without a local NumPy Generator, and
    # restores restore_rng_state(state) likewise. Only global NumPy is required.
    import torch
    assert {'python', 'numpy_global', 'torch_cpu', 'torch_cuda'} <= rng.keys()
    assert {'bit_generator', 'state', 'position', 'has_gauss', 'cached_gaussian'} <= rng['numpy_global'].keys()
    for key in ('torch_cpu', 'torch_cuda'):
        assert rng[key].dtype == torch.uint8 and rng[key].ndim == 1 and rng[key].numel() > 0


def check_payload(payload, base, step=STEP):
    import torch
    state = payload['training_state']
    assert payload['config'] == base
    assert state['global_step'] == step
    assert state['world_size'] == len(state['rng_states']) == 2
    assert state['dataset_cycle_offset'] == step % 20
    assert state['dataset_mix_phase_origin'] == 183000
    assert state['dataset_mix_counts'] == base['dataset_mix_counts']
    assert sum(state['clips_seen'].values()) == step * 8
    for rng in state['rng_states']:
        check_rng(rng)
    optimizer = payload['optimizer']
    assert len(optimizer['state']) == 1053
    groups = {g['name']: len(g['params']) for g in optimizer['param_groups']}
    assert groups == dict(wan_backbone=822, geometry_adapter=38, dense_decoder=132,
                          source_rgb_decay=18, source_rgb_no_decay=43)
    names = [n for g in optimizer['param_groups'] for n in g['params']]
    assert len(set(names)) == 1053 and set(names) == set(optimizer['state'])
    steps = {}
    for name, value in payload['model'].items():
        assert torch.isfinite(value).all(), name
    for name, values in optimizer['state'].items():
        for key in ('exp_avg', 'exp_avg_sq'):
            tensor = values[key]
            assert tensor.dtype == torch.float32 and torch.isfinite(tensor).all(), (name, key)
            assert tensor.shape == payload['model'][name].shape
        count = int(values['step'])
        assert count > 0
        steps[count] = steps.get(count, 0) + 1
    return dict(global_step=step, optimizer_states=1053, Adam_step_histogram=steps,
                optimizer_groups=groups, world_size=2, RNG_ranks=2, clips_seen=state['clips_seen'])


def gate():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    cfg, base = check_config()
    assert not HANDOFF.exists() and not OUTPUT.exists()
    assert shutil.disk_usage(OUTPUT.parent).free >= 68719476736 + 80000000000
    subprocess.run([sys.executable, '-m', 'pytest', '-q',
        'tests/test_resume191k.py', 'tests/test_native512_full.py',
        'tests/test_fp32_master.py', 'tests/test_soft_torchrun.py',
        'tests/test_bounded_resume_diagnostic.py'], check=True)
    assert SOURCE.is_file() and not SOURCE.is_symlink()
    source_sha = file_sha256(SOURCE)
    import torch
    torch.set_num_threads(2)
    payload = torch.load(SOURCE, map_location='cpu', mmap=True, weights_only=True)
    review = check_payload(payload, base)
    del payload
    import wandb
    remote = wandb.Api(timeout=60).run(f"{cfg['tracking']['entity']}/{cfg['tracking']['project']}/{WANDB_ID}")
    assert remote.id == WANDB_ID and remote.state in {'finished', 'failed', 'crashed'}
    # Keep existing191001..191168 metrics; replay locally, upload only new steps.
    last_logged = int(remote.summary['global_step'])
    assert last_logged == 191168, ('unexpected remote progress', last_logged)
    assert int(remote.summary.get('_step', last_logged)) <= last_logged
    temp = HANDOFF.with_name(HANDOFF.name + f'.tmp-{os.getpid()}')
    temp.mkdir()
    os.link(SOURCE, temp / 'resume.pt')
    assert (temp / 'resume.pt').stat().st_ino == SOURCE.stat().st_ino
    atomic_json(temp / 'train_status.json', {'completed_steps': STEP, 'world_size': 2})
    (temp / 'gpu_health.xml').write_text(health())
    report = dict(event='H031_RESUME191K_GATE_OK', checkpoint_sha256=source_sha,
        checkpoint_bytes=SOURCE.stat().st_size, source=str(SOURCE), review=review,
        config_sha256=file_sha256(CONFIG), base_config_sha256=file_sha256(BASE),
        resume_step=STEP, target_step=TARGET, physical_gpus=[1, 6],
        wandb_id=WANDB_ID, wandb_log_after_step=last_logged, seed=cfg['seed'],
        decoder_seed=cfg['decoder_seed'], python=sys.version, torch=torch.__version__,
        cuda=torch.version.cuda, platform=platform.platform(),
        allocator_env={k: os.environ.get(k) for k in ('PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF', 'PYTORCH_NO_CUDA_MEMORY_CACHING')},
        delta='extend existing hold to200k; checkpoint every1000/keep3; strict same W&B resume; unchanged science and allocator')
    atomic_json(temp / 'complete.json', report)
    os.replace(temp, HANDOFF)
    print(json.dumps(report), flush=True)


def train():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1,6'
    cfg, _ = check_config()
    report = json.loads((HANDOFF / 'complete.json').read_text())
    assert report['config_sha256'] == file_sha256(CONFIG)
    assert report['checkpoint_sha256'] == file_sha256(HANDOFF / 'resume.pt')
    assert report['allocator_env'] == {k: os.environ.get(k) for k in report['allocator_env']}
    (HANDOFF / 'gpu_health_at_launch.xml').write_text(health())
    OUTPUT.mkdir(exist_ok=False)
    (OUTPUT / 'wandb_run_id').write_text(WANDB_ID + '\n')
    os.environ['WANDB_MODE'] = 'online'
    os.environ['PYTHONFAULTHANDLER'] = '1'
    print(json.dumps(dict(event='H031_RESUME191K_BEGIN', gate=report)), flush=True)
    argv = [sys.executable, '-m', 'worldbridge.trainer.soft_torchrun',
        '--standalone', '--nproc-per-node=2', '--log-dir', str(OUTPUT) + '-elastic', '--tee', '3',
        'scripts/train.py', '--config', CONFIG, '--output-dir', str(OUTPUT),
        '--resume', str(HANDOFF / 'resume.pt'), '--startup-preflight-updates', '5',
        '--wandb-log-after-step', str(report['wandb_log_after_step'])]
    # PRL signals the whole process group. Remain alive while trainer checkpoints
    # and soft_torchrun reaps ranks; never terminate the supervisor before them.
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: None)
    child = subprocess.Popen(argv)
    stop = threading.Event()
    started = time.monotonic()
    def heartbeat():
        while not stop.wait(7200):
            print(json.dumps(dict(event='H031_TWO_HOUR_CHECK',
                elapsed_seconds=time.monotonic() - started, child_pid=child.pid)), flush=True)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        code = child.wait()
    finally:
        stop.set()
        thread.join()
    raise SystemExit(code if code >= 0 else 128 - code)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gate', action='store_true')
    p.add_argument('--review-endpoint', action='store_true')
    args = p.parse_args()
    os.environ['PYTHONPATH'] = str(ROOT / 'src') + (os.pathsep + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else '')
    if args.gate:
        gate()
    elif args.review_endpoint:
        import torch
        torch.set_num_threads(2)
        cfg, _ = check_config()
        path = OUTPUT / 'checkpoint-0200000.pt'
        report = check_payload(torch.load(path, map_location='cpu', mmap=True, weights_only=True), cfg, TARGET)
        report['checkpoint_sha256'] = file_sha256(path)
        atomic_json(OUTPUT / 'endpoint_review.json', report)
        print(json.dumps(report), flush=True)
    else:
        train()


if __name__ == '__main__':
    main()
