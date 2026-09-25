"""H033 launcher: decoder-native camera token + ray field, migrated from H032 210k.

Modes
-----
gate            CPU. Pin the parent endpoint, write the config with the parent's
                exact counters, run the suite and real GT ray/intrinsics probes.
capacity        GPU(1,6). Short bounded migration that proves the structural
                load, the new optimizer group and one completed update.
continue        GPU(1,6). Resume the reviewed capacity checkpoint to the endpoint.
review-capacity CPU. Verify the short checkpoint (finite model/Adam, increments).
review-endpoint CPU. Verify the endpoint checkpoint.

The parent trunk already contains H032's camera co-adaptation; only the deleted
``camera_head.*`` parameters and Adam states are dropped.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET
import numpy as np
import torch
import yaml
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h033_make_config import camera_parameter_count, make_config
from h031_resume191000_to200000 import check_rng
from worldbridge.data.cache.native import file_sha256
from worldbridge.trainer.config import validate_config
from worldbridge.trainer.schedulers import dataset_for_step
from worldbridge.utils.io import atomic_json

CONFIG = ROOT/'configs/h033_camera_query_ray_to210000.yaml'
PARENT = Path('/data/WorldBridge4D-runs/h032-source-camera-ray-k10-to210000-20260924')
PARENT_STEP = 202000
PARENT_CHECKPOINT = PARENT/f'checkpoint-{PARENT_STEP:07d}.pt'
HANDOFF = Path('/data/WorldBridge4D-runs/h033-camera-query-ray-handoff-20260925')
OUTPUT = Path('/data/WorldBridge4D-runs/h033-camera-query-ray-to210000-20260925')
EXPECTATION = HANDOFF/'parent_expectation.json'
ALLOCATOR_KEYS = ('PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF', 'PYTORCH_NO_CUDA_MEMORY_CACHING')
CAPACITY_STEP = PARENT_STEP + 50
CAMERA_TENSORS = camera_parameter_count(1536, 256, 256)


def expected_clips():
    if not EXPECTATION.is_file():
        return None
    return json.loads(EXPECTATION.read_text())['clips_seen']


def configuration():
    cfg = yaml.safe_load(CONFIG.read_text())
    assert cfg == make_config(clips_seen=expected_clips()), 'config is not the generator output'
    validate_config(cfg, 2)
    return cfg


def health(gpu_ids, expected_uuids=None):
    assert len(gpu_ids) == len(set(gpu_ids)) == 2
    raw = subprocess.check_output(['nvidia-smi', '-i', ','.join(gpu_ids), '-q', '-x'], timeout=20)
    gpus = ET.fromstring(raw).findall('gpu')
    uuids = [g.findtext('uuid') for g in gpus]
    assert len(uuids) == 2
    if expected_uuids is not None:
        assert uuids == expected_uuids
    for g in gpus:
        for key in ('dram_uncorrectable', 'sram_uncorrectable_parity', 'sram_uncorrectable_secded'):
            assert g.findtext('ecc_errors/volatile/'+key) == '0', (g.findtext('uuid'), key)
        for key in ('remapped_row_pending', 'remapped_row_failure'):
            assert g.findtext('remapped_rows/'+key) == 'No', (g.findtext('uuid'), key)
    return uuids, raw.decode()


def gate(gpu_ids):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    cfg = configuration()
    assert not HANDOFF.exists() and not OUTPUT.exists()
    assert PARENT_CHECKPOINT.is_file(), f'parent endpoint missing: {PARENT_CHECKPOINT}'
    assert shutil.disk_usage(OUTPUT.parent).free >= 68719476736 + 100000000000
    parent = torch.load(PARENT_CHECKPOINT, map_location='cpu', mmap=True, weights_only=True)
    state = parent['training_state']
    assert state['global_step'] == PARENT_STEP and state['world_size'] == 2
    for rng in state['rng_states']:
        check_rng(rng)
    parent_optimizer = len(parent['optimizer']['state'])
    parent_camera = sorted(n for n in parent['model'] if n.startswith('camera_head.'))
    assert parent_camera, 'parent checkpoint has no H032 camera head to drop'
    clips = {k: int(v) for k, v in state['clips_seen'].items()}
    sha = file_sha256(PARENT_CHECKPOINT)
    del parent
    HANDOFF.mkdir()
    atomic_json(EXPECTATION, dict(parent_step=PARENT_STEP, checkpoint=str(PARENT_CHECKPOINT),
                                  checkpoint_sha256=sha, clips_seen=clips,
                                  optimizer_states=parent_optimizer, dropped_camera_tensors=len(parent_camera)))
    CONFIG.write_text(yaml.safe_dump(make_config(clips_seen=clips), sort_keys=False))
    configuration()
    subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests'], check=True, cwd=ROOT)
    from worldbridge.models.factory import build_real_model
    model = build_real_model(cfg, 'cpu', load_wan_pretrained=False)
    payload = torch.load(PARENT_CHECKPOINT, map_location='cpu', mmap=True, weights_only=True)
    old, current = payload['model'], model.state_dict()
    fresh = set(current) - set(old)
    dropped = set(old) - set(current)
    assert fresh, 'migration must add fresh camera parameters'
    assert all(n.startswith(('decoder.camera_pose.', 'decoder.camera_rays.')) for n in fresh)
    assert all(n.startswith('camera_head.') for n in dropped)
    assert len(fresh) == 17 and sum(current[n].numel() for n in fresh) == CAMERA_TENSORS
    assert all(current[n].shape == v.shape for n, v in old.items() if n in current)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    old_state = set(payload['optimizer']['state'])
    assert old_state <= trainable and trainable - old_state == fresh
    assert all(n.startswith('camera_head.') for n in old_state - trainable)
    del model, payload, current, old
    import gc
    gc.collect()
    probes = ray_probes(cfg)
    uuids, xml = health(gpu_ids)
    os.link(PARENT_CHECKPOINT, HANDOFF/'resume.pt')
    atomic_json(HANDOFF/'train_status.json', dict(completed_steps=PARENT_STEP, world_size=2))
    report = dict(event='H033_GATE_OK', source_step=PARENT_STEP, target=210000,
                  checkpoint_sha256=sha, config_sha256=file_sha256(CONFIG),
                  gpu_ids=gpu_ids, gpu_uuids=uuids, fresh_tensors=len(fresh),
                  fresh_parameters=CAMERA_TENSORS, dropped_tensors=len(dropped),
                  parent_optimizer_states=parent_optimizer,
                  expected_optimizer_states=parent_optimizer - len(dropped) + len(fresh),
                  probes=probes, allocator_env={k: os.environ.get(k) for k in ALLOCATOR_KEYS},
                  behavior_diff=('camera readout moved into the decoder as one sequence token plus a '
                                 'diagonal-pair ray field; translation normalised by the measured '
                                 'camera-scale medians instead of the 5.62m point-cloud sigma'))
    assert all(v is None for v in report['allocator_env'].values())
    (HANDOFF/'gpu_health.xml').write_text(xml)
    atomic_json(HANDOFF/'complete.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'probes'}), flush=True)


def ray_probes(cfg):
    """Real GT probes for the new intrinsic path: rays, decode, source independence."""
    from worldbridge.data.factory import load_training_datasets
    from worldbridge.data.sampling import deterministic_sample_plan
    from worldbridge.models.camera import CameraOutput
    from worldbridge.trainer.camera_objective import (pixel_grid_coordinates, unit_rays_at,
                                                      unit_source_rays, validate_supervision_camera)
    datasets = load_training_datasets(cfg)
    probes = []
    for name, dataset in datasets.items():
        size = int(getattr(dataset, 'image_size', 256))
        grid = 64 if (name == 'kubric' and cfg.get('native_kubric512_full')) else 32
        step = next(s for s in range(PARENT_STEP, PARENT_STEP+20)
                    if dataset_for_step(s, cfg['seed'], cfg['dataset_mix_counts']) == name)
        index, _, _ = deterministic_sample_plan(dataset, name, cfg['seed'], step, 0, 0, 4)
        camera = dataset.supervision_camera(index)
        report = validate_supervision_camera(camera, size, size)
        K = torch.as_tensor(np.asarray(camera['intrinsics'], np.float32))[None]
        frame = int(np.random.default_rng([cfg['seed'], int(index)]).integers(21))
        ys, xs = pixel_grid_coordinates(grid, size, size, torch.device('cpu'), torch.float32)
        rays = unit_rays_at(K[:, frame], ys, xs)
        assert torch.allclose(rays.norm(dim=1), torch.ones(1, grid, grid), atol=1e-5)
        output = CameraOutput(torch.eye(3)[None, None, None].expand(1, 1, 3, 3),
                              torch.zeros(1, 1, 3), rays, torch.tensor([frame]))
        decoded = output.decode_pinhole(size, size)
        assert bool(decoded['valid']), f'{name}: GT rays must decode'
        focal_error = float((decoded['focal_x']/float(K[0, frame, 0, 0]) - 1).abs())
        focal_error_y = float((decoded['focal_y']/float(K[0, frame, 1, 1]) - 1).abs())
        principal_offset = float((decoded['principal_x'] - float(K[0, frame, 0, 2])).abs())
        principal_offset_y = float((decoded['principal_y'] - float(K[0, frame, 1, 2])).abs())
        assert focal_error < 1e-3 and focal_error_y < 1e-3, (name, focal_error, focal_error_y)
        dense = unit_source_rays(K, size, size)
        assert torch.allclose(dense[0, :, int(round(float(ys[0]))), int(round(float(xs[0])))],
                              rays[0, :, 0, 0], atol=5e-2)
        probe = dict(dataset=name, index=int(index), frame=frame, image_size=size, grid=grid,
                     focal_relative_error=[focal_error, focal_error_y],
                     decoded_principal_offset_px=[principal_offset, principal_offset_y],
                     principal_center_offset_px=report['principal_center_offset_px'],
                     invalid_pose_frames=report['invalid_pose_frames'])
        probes.append(probe)
        print(json.dumps(dict(event='H033_GT_RAY_PROBE', **probe)), flush=True)
    return probes


def review(step, capacity=False):
    cfg = configuration()
    path = OUTPUT/f'checkpoint-{step:07d}.pt'
    payload = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    assert payload['config'] == cfg
    state = payload['training_state']
    assert state['global_step'] == step
    assert state['world_size'] == len(state['rng_states']) == 2
    for rng in state['rng_states']:
        check_rng(rng)
    expectation = json.loads(EXPECTATION.read_text())
    parent = torch.load(HANDOFF/'resume.pt', map_location='cpu', mmap=True, weights_only=True)
    assert payload['coordinate_mean'] == parent['coordinate_mean']
    assert payload['coordinate_scale'] == parent['coordinate_scale']
    expected = {k: int(v) for k, v in expectation['clips_seen'].items()}
    for update in range(PARENT_STEP, step):
        expected[dataset_for_step(update, cfg['seed'], cfg['dataset_mix_counts'])] += 8
    assert state['clips_seen'] == expected, (state['clips_seen'], expected)
    old = parent['optimizer']['state']
    new = payload['optimizer']['state']
    camera_now = sorted(n for n in payload['model']
                        if n.startswith(('decoder.camera_pose.', 'decoder.camera_rays.')))
    assert len(camera_now) == 17
    dropped = set(old) - set(new)
    assert dropped and all(n.startswith('camera_head.') for n in dropped)
    assert set(new) - set(old) == set(camera_now)
    for n, value in payload['model'].items():
        assert torch.isfinite(value).all(), n
    for name, value in new.items():
        previous = int(old[name]['step']) if name in old else 0
        assert int(value['step']) == previous + step - PARENT_STEP, (name, value['step'])
        for key in ('exp_avg', 'exp_avg_sq'):
            assert value[key].dtype == torch.float32 and torch.isfinite(value[key]).all(), (name, key)
            assert value[key].shape == payload['model'][name].shape
    report = dict(event='H033_CHECKPOINT_VERIFIED', step=step, checkpoint=str(path),
                  checkpoint_sha256=file_sha256(path), optimizer_states=len(new),
                  retained_optimizer_states=len(old), camera_tensors=len(camera_now),
                  world_size=2, rng_ranks=2, clips_seen=expected)
    if capacity:
        os.link(path, HANDOFF/'capacity.pt')
        atomic_json(HANDOFF/'train_status.json', dict(completed_steps=step, world_size=2))
        atomic_json(HANDOFF/'capacity_complete.json', report)
    else:
        atomic_json(OUTPUT/'endpoint_review.json', report)
    print(json.dumps(report), flush=True)


def train(gpu_ids, capacity=False):
    cfg = configuration()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ','.join(gpu_ids)
    gate_report = json.loads((HANDOFF/'complete.json').read_text())
    assert gate_report['config_sha256'] == file_sha256(CONFIG) and gate_report['gpu_ids'] == gpu_ids
    assert gate_report['allocator_env'] == {k: os.environ.get(k) for k in ALLOCATOR_KEYS}
    _, xml = health(gpu_ids, gate_report['gpu_uuids'])
    (HANDOFF/('gpu_capacity.xml' if capacity else 'gpu_continue.xml')).write_text(xml)
    if capacity:
        source = HANDOFF/'resume.pt'
        sha = gate_report['checkpoint_sha256']
        assert not OUTPUT.exists()
        OUTPUT.mkdir()
        # No wandb_run_id file: H033 must not append a different architecture to
        # the H032/K10 W&B run. The group/tags below give it its own run.
        resume_args = ['--finetune-from', str(source), '--stop-after-updates', '50']
    else:
        review_report = json.loads((HANDOFF/'capacity_complete.json').read_text())
        assert review_report['step'] == CAPACITY_STEP
        source = HANDOFF/'capacity.pt'
        sha = review_report['checkpoint_sha256']
        resume_args = ['--resume', str(source)]
    assert file_sha256(source) == sha
    os.environ['PYTHONPATH'] = str(ROOT/'src') + os.pathsep + os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONFAULTHANDLER'] = '1'
    os.environ['WANDB_NAME'] = 'h033-decoder-camera-query-ray-to210k'
    argv = [sys.executable, '-m', 'worldbridge.trainer.soft_torchrun', '--standalone',
            '--nproc-per-node=2', '--log-dir', str(OUTPUT)+('-capacity-elastic' if capacity else '-elastic'),
            '--tee', '3', 'scripts/train.py', '--config', str(CONFIG), '--output-dir', str(OUTPUT),
            '--startup-preflight-updates', '5', *resume_args]
    print(json.dumps(dict(event='H033_GPU_BEGIN', capacity=capacity, physical_gpus=gpu_ids, argv=argv)), flush=True)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: None)
    child = subprocess.Popen(argv, cwd=ROOT)
    code = child.wait()
    raise SystemExit(code if code >= 0 else 128 - code)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['gate', 'capacity', 'continue',
                                                          'review-capacity', 'review-endpoint'])
    parser.add_argument('--gpus', help='exact two user-confirmed physical IDs')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode.startswith('review'):
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        review(CAPACITY_STEP if args.mode == 'review-capacity' else 210000,
               capacity=args.mode == 'review-capacity')
    else:
        assert args.gpus is not None
        ids = args.gpus.split(',')
        assert len(ids) == len(set(ids)) == 2 and all(i.isdigit() for i in ids)
        if args.mode == 'gate':
            gate(ids)
        else:
            train(ids, args.mode == 'capacity')


if __name__ == '__main__':
    main()
