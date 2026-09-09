"""Read-only full-state review of a published intermediate H031 checkpoint."""
import argparse
import json
import os
from pathlib import Path
import stat

import torch
import yaml

from worldbridge.data.cache.native import file_sha256
from worldbridge.trainer.schedulers import dataset_for_step
from worldbridge.utils.io import atomic_json

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--step', type=int, required=True)
p.add_argument('--config', required=True)
p.add_argument('--report', required=True)
a = p.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
checkpoint = Path(a.checkpoint)
report_path = Path(a.report)
assert not report_path.exists()
snapshot_stat = checkpoint.lstat()
assert stat.S_ISREG(snapshot_stat.st_mode)
cfg = yaml.safe_load(Path(a.config).read_text())
origin_path = Path(cfg['selected_checkpoint_path'])
origin_sha = '970ed406b28e4508a3ab206ff4492e0641cbe262f494e9dca3693b922f4b63f0'
assert file_sha256(origin_path) == origin_sha
print('PERIODIC_REVIEW_ORIGIN_SHA_OK', flush=True)
c = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
o = torch.load(origin_path, map_location='cpu', mmap=True, weights_only=True)
s = c['training_state']
origin = 152768
delta = a.step - origin
assert 0 < delta and a.step <= 170000
assert o['training_state']['global_step'] == origin
assert s['global_step'] == a.step and s['world_size'] == 2 and len(s['rng_states']) == 2
assert s['dataset_cycle_offset'] == a.step % 20
assert s['dataset_mix_phase_origin'] == origin
for key in ('dataset_mix_counts', 'lr_restart', 'targets_per_source', 'runtime_stall_traceback_seconds'):
    assert c['config'][key] == cfg[key], key
assert s['dataset_mix_counts'] == cfg['dataset_mix_counts']
assert set(c['model']) == set(o['model'])
assert all(v.shape == o['model'][k].shape for k, v in c['model'].items())
for k, v in c['model'].items():
    if k.startswith('decoder.'):
        assert v.dtype == torch.float32 and torch.isfinite(v).all(), k
current, baseline = c['optimizer']['state'], o['optimizer']['state']
assert len(current) == len(baseline) == 193 and set(current) == set(baseline)
for k, v in current.items():
    assert int(v['step']) == int(baseline[k]['step']) + delta, k
    assert v['exp_avg'].dtype == v['exp_avg_sq'].dtype == torch.float32, k
    assert torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all(), k
updates = {name: 0 for name in cfg['dataset_mix_counts']}
for step in range(origin, a.step):
    updates[dataset_for_step(step, cfg['seed'], cfg['dataset_mix_counts'])] += 1
for name, n in o['training_state']['clips_seen'].items():
    assert s['clips_seen'][name] == n + 8 * updates[name], name
print('PERIODIC_REVIEW_STATE_OK', flush=True)
sha = file_sha256(checkpoint)
after = checkpoint.lstat()
assert (snapshot_stat.st_dev, snapshot_stat.st_ino, snapshot_stat.st_size, snapshot_stat.st_mtime_ns) == (
    after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
report = dict(event='H031_PERIODIC_FULL_STATE_OK', checkpoint=str(checkpoint),
              checkpoint_sha256=sha, checkpoint_bytes=after.st_size, global_step=a.step,
              origin=str(origin_path), origin_sha256=origin_sha, phase_origin=origin,
              optimizer_steps_incremented=delta, Adam_states=193, world_size=2, RNG_ranks=2,
              dataset_updates=updates, clips_seen=s['clips_seen'],
              config=a.config, config_sha256=file_sha256(a.config), seed=cfg['seed'],
              scope='full state structural/finite/counter review; not RNG replay or quality evaluation',
              storage_caveat='checkpoint stays under normal keep-last-3 rolling retention; no hardlink pin')
atomic_json(report_path, report)
print(json.dumps(report), flush=True)
