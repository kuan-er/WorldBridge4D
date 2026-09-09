"""Endpoint integrity, not a quality verdict, for K9 + 50/25/25."""
import json
from pathlib import Path
import torch
import yaml
from worldbridge.data.cache.native import file_sha256
from worldbridge.trainer.schedulers import dataset_for_step
from worldbridge.utils.io import atomic_json

cfg = yaml.safe_load(Path('configs/h031_k512_k9_mix50_152768_to154768.yaml').read_text())
root = Path('/data/WorldBridge4D-runs/h031-k512-k9-mix50-152768-to154768-gpu23-20260909')
summary = json.loads((root / 'train_status.json').read_text())
assert summary['completed_steps'] == summary['execution_end'] == summary['target_steps'] == 154768
assert summary['diagnostic_stop_after_updates'] is None
assert not summary['quiesce_geometry_before_forward']
assert summary['world_size'] == 2 and summary['targets_per_source'] == 9
assert summary['dataset_mix_counts'] == cfg['dataset_mix_counts']
p = root / 'checkpoint-0154768.pt'
c = torch.load(p, map_location='cpu', mmap=True, weights_only=True)
o = torch.load(cfg['selected_checkpoint_path'], map_location='cpu', mmap=True, weights_only=True)
s = c['training_state']
assert s['global_step'] == 154768 and s['world_size'] == 2 and len(s['rng_states']) == 2
assert s['dataset_cycle_offset'] == 8 and s['dataset_mix_phase_origin'] == 152768
assert s['dataset_mix_counts'] == c['config']['dataset_mix_counts'] == cfg['dataset_mix_counts']
assert c['config']['targets_per_source'] == 9 and c['config']['lr_restart'] == cfg['lr_restart']
assert set(c['model']) == set(o['model'])
assert all(v.shape == o['model'][k].shape for k, v in c['model'].items())
a, b = c['optimizer']['state'], o['optimizer']['state']
assert len(a) == len(b) == 193 and set(a) == set(b)
assert all(int(a[k]['step']) == int(b[k]['step']) + 2000 for k in a)
assert all(v['exp_avg'].dtype == v['exp_avg_sq'].dtype == torch.float32 for v in a.values())
assert all(torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all() for v in a.values())
assert all(v.dtype == torch.float32 for k, v in c['model'].items() if k.startswith('decoder.'))
updates = {name: sum(dataset_for_step(i, cfg['seed'], cfg['dataset_mix_counts']) == name
                    for i in range(152768, 154768)) for name in cfg['dataset_mix_counts']}
assert list(updates.values()) == [1000, 500, 500]
for name, n in o['training_state']['clips_seen'].items():
    assert s['clips_seen'][name] == n + 8 * updates[name]
report = dict(event='H031_K9_MIX50_2000_FULL_STATE_OK', global_step=154768,
    origin_step=152768, updates=2000, B=1, A=4, K=9, world=2, pairs_per_update=72,
    dataset_updates=updates, Adam_states=193, all_Adam_steps_incremented=2000,
    RNG_ranks=2, checkpoint_sha256=file_sha256(p), training_summary=summary,
    quality_requires_fixed_heldout_validation=True, joint_K_and_mix_change=True)
atomic_json(root / 'full_state_review.json', report)
print(json.dumps(report), flush=True)
