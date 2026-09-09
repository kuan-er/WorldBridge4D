"""Launch the CPU-gated, full-state K9/50-25-25 2000-update trial."""
import json
import os
from pathlib import Path
import shutil
import sys
import yaml
from worldbridge.data.cache.native import file_sha256

config = 'configs/h031_k512_k9_mix50_152768_to154768.yaml'
cfg = yaml.safe_load(Path(config).read_text())
gate = json.loads(Path('/data/WorldBridge4D-runs/h031-k9-mix50-cpu-gate-20260909.json').read_text())
assert gate['event'] == 'H031_K9_MIX_CPU_OK'
assert gate['config_sha256'] == file_sha256(config)
assert gate['origin_sha256'] == cfg['selected_checkpoint_sha256']
assert gate['start'] == 152768 and gate['end'] == 154768 and len(gate['cases']) == 6
assert os.environ['CUDA_VISIBLE_DEVICES'] == '2,3'
assert shutil.disk_usage('/data/WorldBridge4D-runs').free >= 68719476736 + 38400000000
output = '/data/WorldBridge4D-runs/h031-k512-k9-mix50-152768-to154768-gpu23-20260909'
assert not Path(output).exists()
print(json.dumps(dict(event='H031_K9_MIX50_TRIAL_BEGIN', physical_gpus=[2, 3],
    config_sha256=gate['config_sha256'], origin_sha256=gate['origin_sha256'],
    start=152768, stop=154768, updates=2000, B=1, A=4, K=9, world=2,
    clips_per_update=8, pairs_per_update=72, counts_per20=cfg['dataset_mix_counts'],
    seed=cfg['seed'], LR='preserved150000_origin500warmup_now3e-6_hold',
    objective='unchanged_XYZ_cycle0_full_paths', execution='original_async_prefetch',
    joint_change='K_and_mix_together_no_separate_causal_attribution',
    page_faults='observation_not_failure', GT='verified_cache_or_original_raw_no_wait_full_bulk',
    missing_or_corrupt_inputs='fatal_no_subset_fallback', allocator='unchanged')), flush=True)
os.execv(sys.executable, [sys.executable, '-m', 'worldbridge.trainer.soft_torchrun',
    '--standalone', '--nproc-per-node=2', '--log-dir', output+'-elastic', '--tee', '3',
    'scripts/train.py', '--config', config, '--output-dir', output,
    '--resume', cfg['selected_checkpoint_path'], '--input-readiness',
    '--input-readiness-timeout-seconds', '900'])
