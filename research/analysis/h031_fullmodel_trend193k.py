"""Read-only CPU audit of current full-model training; no validation/convergence claim."""
import hashlib
import json
import math
import os
from pathlib import Path
import statistics as st
import sys
import wandb

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
OUT = Path('/data/WorldBridge4D-runs/h031-fullmodel-trend193k-20260923')
OUT.mkdir(exist_ok=False)
RUN = 'zhaigong2023-sjtu-hpc-center/worldbridge4d/zypk1th1'
KEYS = ['global_step', 'train/dataset', 'train/raw_epe_m', 'train/xyz_loss']
rows = [r for r in wandb.Api(timeout=60).run(RUN).scan_history(keys=KEYS, page_size=500)
        if 183000 < int(r['global_step']) <= 193000]
assert rows and all(math.isfinite(float(r[k])) for r in rows for k in KEYS)
raw = json.dumps(rows, sort_keys=True) + '\n'
(OUT / 'history_raw.json').write_text(raw)
unique, conflicts = {}, []
for row in rows:
    step = int(row['global_step'])
    if step in unique and any(unique[step][k] != row[k] for k in KEYS):
        conflicts.append(dict(step=step, first=unique[step], duplicate=row))
    else:
        unique[step] = row
(OUT / 'conflicts.json').write_text(json.dumps(conflicts, indent=2) + '\n')
assert not conflicts, 'Conflicting duplicate steps; no aggregate permitted'
assert max(unique) == 193000, '193000 endpoint missing'
report = dict(run=RUN, raw_rows=len(rows), unique_rows=len(unique),
              identical_duplicates=len(rows)-len(unique), min_step=min(unique), max_step=max(unique),
              raw_sha256=hashlib.sha256(raw.encode()).hexdigest(), python=sys.version,
              wandb_version=wandb.__version__, seed=None, windows=[],
              caveat='Descriptive logged-batch macro means, not matched samples, held-out validation, or proof of convergence. Fixed full-model phase; first 500 updates warm up. Replay through191168 suppressed on resume. No randomness or GPU computation in audit.')
for end in range(184000, 193001, 1000):
    lo = end - 999
    block = dict(start=lo, end=end, logged_updates=sum(lo <= s <= end for s in unique), datasets={})
    for i, name in enumerate(['kubric', 'pointodyssey', 'dynamic_replica']):
        rr = [r for s, r in sorted(unique.items()) if lo <= s <= end and int(r['train/dataset']) == i]
        if not rr:
            continue
        epe = [float(r['train/raw_epe_m']) for r in rr]
        block['datasets'][name] = dict(n=len(rr), EPE_mean=st.mean(epe), EPE_median=st.median(epe),
                                      EPE_std=st.stdev(epe) if len(epe)>1 else None,
                                      XYZ_mean=st.mean(float(r['train/xyz_loss']) for r in rr))
    report['windows'].append(block)
    print(json.dumps(dict(event='EPE_WINDOW', **block)), flush=True)
(OUT / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(event='FULLMODEL_TREND_OK', report=str(OUT / 'summary.json'))), flush=True)
