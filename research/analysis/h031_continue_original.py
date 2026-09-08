"""Resume the verified original async-prefetch branch, not the quiescent A/B."""
import json
import os
from pathlib import Path
import shutil
import sys

import yaml
from worldbridge.data.cache.native import file_sha256

CONFIG = 'configs/h031_k512_po256_dr256_k5_10k_ondemand.yaml'
CONFIG_SHA = '9e507e46ff25deb68f67c1848ded63eb9aeea2b7ae12f825e61dbfb66d9dcb4d'
ROOT = Path('/data/WorldBridge4D-runs/h031-k5-first2-diagnostic-gpu23-20260908')
RESUME_SHA = '5f6d329dd253044771686c976fc38ba16a99dec09888bbe53ed36c3734a5191b'
OUTPUT = '/data/WorldBridge4D-runs/h031-k512-k5-original-prefetch-150012-to160010-gpu23-20260908'
assert os.environ['CUDA_VISIBLE_DEVICES'] == '2,3'
assert file_sha256(CONFIG) == CONFIG_SHA
cfg = yaml.safe_load(Path(CONFIG).read_text())
report = json.loads((ROOT / 'full_state_review.json').read_text())
assert report['event'] == 'H031_FIRST2_DIAGNOSTIC_FULL_STATE_OK'
assert report['checkpoint_sha256'] == RESUME_SHA
assert report['all_Adam_steps_incremented'] == 2 and report['RNG_ranks'] == 2
assert shutil.disk_usage('/data/WorldBridge4D-runs').free >= 68719476736 + 38400000000
assert not Path(OUTPUT).exists()
print(json.dumps(dict(event='H031_ORIGINAL_PREFETCH_LONGTRAIN_BEGIN',
    physical_gpus=[2, 3], B=1, A=4, K=5, world=2, clips_per_update=8,
    pairs_per_update=40, resume=150012, endpoint=160010, remaining_updates=9998,
    seed=cfg['seed'], config_sha256=CONFIG_SHA, resume_sha256=RESUME_SHA,
    original150010_sha256=cfg['selected_checkpoint_sha256'],
    LR_start=150000, LR_warmup=500, LR_hold=3e-6,
    execution='original_async_prefetch_NO_quiescence_NO_diagnostic_stop',
    quiescent_state='not_used_strict_Adam_gate_failed',
    page_fault_and_stall_alarm='observation_only_not_automatic_failure_or_stop',
    full_horizon_payload_preflight='retained_may_take_tens_of_minutes',
    GT_bulk='paused_storage_guard_not_dependency',
    GT_mode='verified_cache_or_original_native_raw_full_corpus_not_subset',
    allocator_loss_and_cycle_paths='unchanged',
    historical_rank0_exit_cause='not_uniquely_determined_exit_audit_retained')),
    flush=True)
os.execv(sys.executable, [sys.executable, '-m', 'worldbridge.trainer.soft_torchrun',
    '--standalone', '--nproc-per-node=2', '--log-dir', OUTPUT+'-elastic', '--tee', '3',
    'scripts/train.py', '--config', CONFIG, '--output-dir', OUTPUT,
    '--resume', str(ROOT / 'checkpoint-0150012.pt'),
    '--input-readiness', '--input-readiness-timeout-seconds', '900'])
