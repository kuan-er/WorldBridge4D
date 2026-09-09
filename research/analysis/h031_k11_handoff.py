"""Run only after old full-preflight workers terminate; publish verified resume alias."""
import json
import shutil
import time
from pathlib import Path
import torch
import yaml
from worldbridge.data.cache.native import file_sha256
from worldbridge.trainer.config import validate_config
from worldbridge.trainer.schedulers import dataset_for_step
from worldbridge.utils.io import atomic_json

config = 'configs/h031_k512_k11_mix50_prefix5_to170000.yaml'
cfg = yaml.safe_load(Path(config).read_text()); validate_config(cfg, 2)
old_root = Path('/data/WorldBridge4D-runs/h031-k512-k9-mix50-prefix5-to170000-gpu23-20260909')
fallback = Path('/data/WorldBridge4D-runs/h031-k512-k9-mix50-152768-to154768-gpu23-20260909/checkpoint-0152769.pt')
root = Path('/data/WorldBridge4D-runs/h031-k11-capacity-handoff-20260909')
origin = Path(cfg['selected_checkpoint_path'])
assert file_sha256(origin) == cfg['selected_checkpoint_sha256']
# The dependency guarantees the old writer has exited; no race with publication/pruning.
checkpoints = sorted(old_root.glob('checkpoint-*.pt'))
source = checkpoints[-1] if checkpoints else fallback
if source == fallback:
    assert file_sha256(source) == '91c1a06bea74f1cb304df38f7e9825e9db308022f35b555999bf96016dbe19e3'
sha = file_sha256(source) if source != origin else cfg['selected_checkpoint_sha256']
c = torch.load(source, map_location='cpu', mmap=True, weights_only=True)
o = torch.load(origin, map_location='cpu', mmap=True, weights_only=True)
s = c['training_state']; start = int(s['global_step']); delta = start - 152768
assert 0 <= delta < 170000 - 152768
assert s['world_size'] == 2 and len(s['rng_states']) == 2 and s['dataset_cycle_offset'] == start % 20
if source.parent == old_root:
    old_status = json.loads((old_root/'train_status.json').read_text())
    assert old_status['completed_steps'] == start
    assert c['config']['dataset_mix_counts'] == cfg['dataset_mix_counts']
    assert c['config']['targets_per_source'] == 9
    assert s['dataset_mix_phase_origin'] == 152768
assert set(c['model']) == set(o['model'])
assert all(v.shape == o['model'][k].shape for k,v in c['model'].items())
a,b = c['optimizer']['state'],o['optimizer']['state']
assert len(a) == len(b) == 193 and set(a) == set(b)
assert all(int(a[k]['step']) == int(b[k]['step']) + delta for k in a)
assert all(v['exp_avg'].dtype == v['exp_avg_sq'].dtype == torch.float32 for v in a.values())
assert all(torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all() for v in a.values())
assert all(v.dtype == torch.float32 for k,v in c['model'].items() if k.startswith('decoder.'))
for name,n in o['training_state']['clips_seen'].items():
    assert s['clips_seen'][name] == n + 8*sum(dataset_for_step(i,cfg['seed'],cfg['dataset_mix_counts']) == name
                                           for i in range(152768,start))
root.mkdir(parents=True, exist_ok=False)
(root/'resume.pt').symlink_to(source)
atomic_json(root/'train_status.json', dict(completed_steps=start, world_size=2))
report = dict(event='H031_K11_HANDOFF_OK', config_sha256=file_sha256(config),
    source=str(source), checkpoint_sha256=sha, start=start, end=170000,
    remaining_updates=170000-start, preserved_updates_since152768=delta,
    Adam_states=193, RNG_ranks=2, clips_seen=s['clips_seen'], startup_preflight_updates=5,
    CPU_regression_run='R-20260909060805-d6bbcd',
    fallback_config_sha256=file_sha256('configs/h031_k512_k9_mix50_prefix5_to170000.yaml'),
    policy='K11_native_complete_update_then_continue170k_else_capacity_failure_restore_this_K9_fullstate')
atomic_json(root/'complete.json', report)
# Admission waits on storage, not on a GPU lease; never delete inputs or lower reserve.
required_free = 68719476736 + 38400000000
free = shutil.disk_usage('/data/WorldBridge4D-runs').free
if free < required_free:
    print(json.dumps(dict(event='H031_K11_STORAGE_WAIT', free_bytes=free,
        required_bytes=required_free, deficit_bytes=required_free-free,
        checkpoint_sha256=sha, resume_step=start)), flush=True)
    deadline = time.monotonic() + 172800
    while shutil.disk_usage('/data/WorldBridge4D-runs').free < required_free:
        if time.monotonic() >= deadline:
            raise RuntimeError('storage admission timed out; no files removed or reserve reduced')
        time.sleep(5)
print(json.dumps(report), flush=True)
