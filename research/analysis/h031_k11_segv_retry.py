"""One full-state K11 retry; disable only periodic all-thread stack dumping."""
import json
import os
from pathlib import Path
import shutil
import sys
from h031_k11_segv_gate import CONFIG, ROOT, OUTPUT, HANDOFF
from worldbridge.data.cache.native import file_sha256

r = json.loads((ROOT/'complete.json').read_text())
assert r['event'] == 'H031_K11_SEGV_GATE_OK' and r['resume_step'] == 152774
assert file_sha256(CONFIG) == r['config_sha256']
assert file_sha256(HANDOFF/'resume.pt') == r['checkpoint_sha256']
assert os.environ['CUDA_VISIBLE_DEVICES'] == '2,3' and os.environ['PYTHONFAULTHANDLER'] == '1'
assert not OUTPUT.exists()
assert shutil.disk_usage(OUTPUT.parent).free >= 68719476736+38400000000
print(json.dumps(dict(event='H031_K11_SEGV_RETRY_BEGIN', gate=r,
    execution='same_K11_science_fullstate152774_no_new_warmup_original_async',
    caveat='86_old_completed_updates_unsaved_replayed_not_exact_wallclock_or_root_cause_proof',
    diagnostics='periodic_dump_disabled_fatal_faulthandler_rank_health_PRL_watchdog_retained')),flush=True)
os.execv(sys.executable,[sys.executable,'-m','worldbridge.trainer.soft_torchrun',
    '--standalone','--nproc-per-node=2','--log-dir',str(OUTPUT)+'-elastic','--tee','3',
    'scripts/train.py','--config',CONFIG,'--output-dir',str(OUTPUT),
    '--resume',str(HANDOFF/'resume.pt'),'--startup-preflight-updates','5',
    '--input-readiness','--input-readiness-timeout-seconds','900'])
