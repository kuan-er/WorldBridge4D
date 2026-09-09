"""Full-state K9/50-25-25 continuation to170000, explicit five-update preflight."""
import json
import os
from pathlib import Path
import shutil
import sys
import yaml
from worldbridge.data.cache.native import file_sha256
from worldbridge.trainer.config import validate_config

config = 'configs/h031_k512_k9_mix50_prefix5_to170000.yaml'
cfg = yaml.safe_load(Path(config).read_text()); validate_config(cfg, 2)
root = Path('/data/WorldBridge4D-runs/h031-k9-mix50-prefix5-to170000-handoff-20260909')
handoff = json.loads((root/'complete.json').read_text())
assert handoff['event'] == 'H031_PREFIX170K_HANDOFF_OK'
assert handoff['config_sha256'] == file_sha256(config)
assert handoff['end'] == 170000 and handoff['startup_preflight_updates'] == 5
assert file_sha256(root/'resume.pt') == handoff['checkpoint_sha256']
assert os.environ['CUDA_VISIBLE_DEVICES'] == '2,3'
assert shutil.disk_usage('/data/WorldBridge4D-runs').free >= 68719476736 + 38400000000
output = '/data/WorldBridge4D-runs/h031-k512-k9-mix50-prefix5-to170000-gpu23-20260909'
assert not Path(output).exists()
print(json.dumps(dict(event='H031_PREFIX170K_BEGIN', config_sha256=handoff['config_sha256'],
    resume_sha256=handoff['checkpoint_sha256'], resume_source=handoff['source'],
    start=handoff['start'], end=170000, remaining_updates=handoff['remaining_updates'],
    phase_origin=152768, physical_gpus=[2,3], B=1, A=4, K=9, world=2,
    counts_per20=cfg['dataset_mix_counts'], seed=cfg['seed'], startup_preflight_updates=5,
    LR='original150000_origin500warmup_now3e-6_hold_extended_to170000_no_restart',
    objective='unchanged_XYZ_cycle0_full_paths', execution='original_async_prefetch',
    payload_policy='prefix5_then_strict_actual_input_read_before_forward_no_subset_or_generation',
    allocator='unchanged', CPU_regression_run='R-20260909054559-8a9def')), flush=True)
os.execv(sys.executable,[sys.executable,'-m','worldbridge.trainer.soft_torchrun',
    '--standalone','--nproc-per-node=2','--log-dir',output+'-elastic','--tee','3',
    'scripts/train.py','--config',config,'--output-dir',output,
    '--resume',str(root/'resume.pt'),'--startup-preflight-updates','5',
    '--input-readiness','--input-readiness-timeout-seconds','900'])
