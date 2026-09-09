"""Verify the170000 full-state endpoint independently; not a quality verdict."""
import json
from pathlib import Path
import torch
import yaml
from worldbridge.data.cache.native import file_sha256
from worldbridge.trainer.schedulers import dataset_for_step
from worldbridge.utils.io import atomic_json

cfg = yaml.safe_load(Path('configs/h031_k512_k9_mix50_prefix5_to170000.yaml').read_text())
root = Path('/data/WorldBridge4D-runs/h031-k512-k9-mix50-prefix5-to170000-gpu23-20260909')
summary = json.loads((root/'train_status.json').read_text())
assert summary['completed_steps'] == summary['execution_end'] == summary['target_steps'] == 170000
assert summary['diagnostic_stop_after_updates'] is None and not summary['quiesce_geometry_before_forward']
assert summary['world_size'] == 2 and summary['targets_per_source'] == 9
assert summary['dataset_mix_counts'] == cfg['dataset_mix_counts']
f = summary['startup_preflight']; assert f['requested_updates'] == 5
assert f['end'] == min(f['start']+5,170000) and f['execution_end'] == 170000
p = root/'checkpoint-0170000.pt'
c = torch.load(p,map_location='cpu',mmap=True,weights_only=True)
o = torch.load(cfg['selected_checkpoint_path'],map_location='cpu',mmap=True,weights_only=True)
s = c['training_state']; origin = 152768; delta = 170000-origin
assert s['global_step'] == 170000 and s['world_size'] == 2 and len(s['rng_states']) == 2
assert s['dataset_cycle_offset'] == 0 and s['dataset_mix_phase_origin'] == origin
assert s['dataset_mix_counts'] == c['config']['dataset_mix_counts'] == cfg['dataset_mix_counts']
assert c['config']['lr_restart'] == cfg['lr_restart'] and c['config']['targets_per_source'] == 9
assert set(c['model']) == set(o['model']) and all(v.shape == o['model'][k].shape for k,v in c['model'].items())
a,b = c['optimizer']['state'],o['optimizer']['state']
assert len(a) == len(b) == 193 and set(a) == set(b)
assert all(int(a[k]['step']) == int(b[k]['step'])+delta for k in a)
assert all(v['exp_avg'].dtype == v['exp_avg_sq'].dtype == torch.float32 for v in a.values())
assert all(torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all() for v in a.values())
assert all(v.dtype == torch.float32 for k,v in c['model'].items() if k.startswith('decoder.'))
updates = {name:sum(dataset_for_step(i,cfg['seed'],cfg['dataset_mix_counts']) == name
                   for i in range(origin,170000)) for name in cfg['dataset_mix_counts']}
assert sum(updates.values()) == delta
for name,n in o['training_state']['clips_seen'].items():
    assert s['clips_seen'][name] == n+8*updates[name]
report = dict(event='H031_PREFIX170K_FULL_STATE_OK', global_step=170000,
    phase_origin=origin, phase_updates=delta, dataset_updates=updates, Adam_states=193,
    all_Adam_steps_incremented=delta, RNG_ranks=2, checkpoint_sha256=file_sha256(p),
    training_summary=summary, quality_requires_fixed_heldout_validation=True)
atomic_json(root/'full_state_review.json',report)
print(json.dumps(report),flush=True)
