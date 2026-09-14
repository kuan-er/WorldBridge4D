"""CPU-only descriptive training-history audit; not matched validation or a convergence test."""
import hashlib
import json
import math
import os
from pathlib import Path
import statistics as st
import sys
import wandb

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
OUT=Path('/data/WorldBridge4D-runs/h031-epe-trend180k-audit-20260914')
OUT.mkdir(exist_ok=False)
RUN='zhaigong2023-sjtu-hpc-center/worldbridge4d/0myi2hf4'
OLD=Path('/data/WorldBridge4D-runs/h030-rgb1x-final-through155000-20260907/vs-bf16-norm30/summary.json')
keys=['global_step','train/dataset','train/raw_epe_m','train/xyz_loss']
run=wandb.Api(timeout=60).run(RUN)
rows=list(run.scan_history(keys=keys,page_size=500))
rows=[r for r in rows if 175374 < int(r['global_step']) <= 180000]
assert rows and all(math.isfinite(float(r[k])) for r in rows for k in keys)
assert len({int(r['global_step']) for r in rows}) == len(rows)
(OUT/'history.json').write_text(json.dumps(rows,sort_keys=True)+'\n')
names=['kubric','pointodyssey','dynamic_replica']
windows=[(175375,176000),(176001,177000),(177001,178000),(178001,179000),(179001,180000),
         (175375,176374),(175375,180000)]
report=dict(run=RUN,PRL_run='R-20260913031335-9eddf0',scope='through180000 only',python=sys.version,
    wandb_version=wandb.__version__,logged_updates=len(rows),min_step=min(r['global_step'] for r in rows),
    max_step=max(r['global_step'] for r in rows),windows=[],
    caveat='Batch-macro means over logged diagnostic updates, not every optimizer update or point-weighted pooled EPE. Unmatched samples/resolution/K versus historical256; no causal or held-out convergence claim.')
for lo,hi in windows:
    block=dict(start=lo,end=hi,datasets={})
    for i,name in enumerate(names):
        rr=[r for r in rows if lo<=r['global_step']<=hi and int(r['train/dataset'])==i]
        if not rr:continue
        vals=[float(r['train/raw_epe_m']) for r in rr]
        block['datasets'][name]=dict(logged_batches=len(vals),EPE_mean=st.mean(vals),
            EPE_median=st.median(vals),EPE_std=st.stdev(vals) if len(vals)>1 else None,
            XYZ_mean=st.mean(float(r['train/xyz_loss']) for r in rr))
    report['windows'].append(block)
    print(json.dumps(dict(event='EPE_WINDOW',**block)),flush=True)
old=json.loads(OLD.read_text())
report['historical_source']=str(OLD)
report['historical_sha256']=hashlib.sha256(OLD.read_bytes()).hexdigest()
report['historical256']=dict(baseline=old['baseline'],candidate=old['candidate'],windows=[
    dict(start=w['start'],end=w['end'],datasets={n:dict(
        logged_batches=v['paired_logged_batches'],EPE=v['raw_epe_m'],XYZ=v['xyz_loss'])
        for n,v in w['datasets'].items()}) for w in old['windows']])
report['history_sha256']=hashlib.sha256((OUT/'history.json').read_bytes()).hexdigest()
(OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(event='EPE_TREND_AUDIT_OK',report=str(OUT/'summary.json'),
    historical256=report['historical256'])),flush=True)
