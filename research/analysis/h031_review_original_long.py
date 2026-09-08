"""Success-only full-state review of original-prefetch H031 step160010."""
import json
from pathlib import Path
import torch
from worldbridge.data.cache.native import file_sha256
from worldbridge.trainer.schedulers import dataset_for_step
from worldbridge.utils.io import atomic_json

root = Path('/data/WorldBridge4D-runs/h031-k512-k5-original-prefetch-150012-to160010-gpu23-20260908')
summary = json.loads((root / 'train_status.json').read_text())
assert summary['completed_steps'] == summary['execution_end'] == summary['target_steps'] == 160010
assert summary['diagnostic_stop_after_updates'] is None
assert not summary['quiesce_geometry_before_forward']
assert summary['world_size'] == 2 and summary['targets_per_source'] == 5
path = root / 'checkpoint-0160010.pt'
checkpoint = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
state = checkpoint['training_state']
assert state['global_step'] == 160010 and state['world_size'] == 2
assert len(state['rng_states']) == 2 and state['dataset_cycle_offset'] == 10
origin = torch.load('/data/WorldBridge4D-runs/h031-k512-po256-dr256-b1-a4-k5-dr-mmap-capacity-20260908/checkpoint-0150010.pt', map_location='cpu', mmap=True, weights_only=True)
resume = torch.load('/data/WorldBridge4D-runs/h031-k5-first2-diagnostic-gpu23-20260908/checkpoint-0150012.pt', map_location='cpu', mmap=True, weights_only=True)
assert set(checkpoint['model']) == set(origin['model'])
assert all(value.shape == origin['model'][key].shape for key, value in checkpoint['model'].items())
a, b, r = [item['optimizer']['state'] for item in (checkpoint, origin, resume)]
assert len(a) == len(b) == len(r) == 193 and set(a) == set(b) == set(r)
assert all(int(a[k]['step']) == int(b[k]['step']) + 10000 == int(r[k]['step']) + 9998 for k in a)
assert all(v['exp_avg'].dtype == v['exp_avg_sq'].dtype == torch.float32 for v in a.values())
assert all(v.dtype == torch.float32 for k, v in checkpoint['model'].items() if k.startswith('decoder.'))
for name, old_count in origin['training_state']['clips_seen'].items():
    expected = old_count + 8 * sum(dataset_for_step(i, 20260812) == name for i in range(150010, 160010))
    assert state['clips_seen'][name] == expected
report = dict(event='H031_ORIGINAL_PREFETCH_160010_FULL_STATE_OK', global_step=160010,
    updates_since150010=10000, updates_since150012=9998, B=1, A=4, K=5, world=2,
    optimizer_states=193, all_Adam_steps_incremented=10000, RNG_ranks=2,
    checkpoint_sha256=file_sha256(path), training_summary=summary,
    quality_requires_fixed_heldout_validation=True)
atomic_json(root / 'full_state_review.json', report)
print(json.dumps(report), flush=True)
