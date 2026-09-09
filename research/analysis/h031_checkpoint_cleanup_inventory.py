"""Read-only checkpoint-space proposal: inode accounting, no payload reads/deletion."""
import json
import os
from pathlib import Path
import stat
from datetime import datetime, timezone

RUNS = Path('/data/WorldBridge4D-runs')
PERSISTENT = Path('/data/WorldBridge4D-persistent/checkpoints')
ACTIVE = RUNS/'h031-k512-k11-mix50-prefix5-to170000-gpu23-20260909'
KEEP = {
 'h030-fp32-cycle0-rgb1x-step155000-reference/checkpoint-0155000.pt': 'recent256_plain_RGB1x_reference',
 'h030-fp32-cycle0-rgb1x-boundary2x-step155000-reference/checkpoint-0155000.pt': 'recent256_boundary2x_tradeoff_reference',
 'h030-h027-140k-to-150k-cycle-all-norm30-gpu56-k13-b1-r1/checkpoint-0150000.pt': 'pre512_origin',
 'h030-151500-to-155k-edgecontrast0p01-gpu23-b2-k9-20260908/checkpoint-0154059.pt': 'protected_unfinished_0p01_endpoint_quality_unresolved',
 'h030-150k-to-155k-boundary2x-edgecontrast0p01-gpu23-b2-k15/checkpoint-0151500.pt': 'previously_protected_0p01_recovery',
 'h031-k512-po256-dr256-b1-a4-k15-live-ready-capacity-20260908/checkpoint-0150001.pt': 'previously_protected_capacity_origin',
 'h031-k512-po256-dr256-b1-a4-k5-dr-mmap-capacity-20260908/checkpoint-0150010.pt': 'protected_native512_phase_origin',
 'h031-k5-first2-diagnostic-gpu23-20260908/checkpoint-0150012.pt': 'protected_async_branch_origin',
 'h031-k5-first2-quiescent-diagnostic-gpu23-20260908/checkpoint-0150012.pt': 'protected_numerical_comparison_variant',
 'h031-k512-k5-original-prefetch-150012-to160010-gpu23-20260908/checkpoint-0152768.pt': 'ACTIVE_ENDPOINT_REVIEW_READS_THIS',
 'h031-k512-k9-mix50-152768-to154768-gpu23-20260909/checkpoint-0152769.pt': 'protected_previous_handoff',
 'h031-k512-k9-mix50-prefix5-to170000-gpu23-20260909/checkpoint-0152774.pt': 'ACTIVE_K11_RESUME_AND_K9_FALLBACK',
}
keep = {}
for rel, reason in KEEP.items():
    p = RUNS/rel
    st = p.stat(); keep[(st.st_dev, st.st_ino)] = reason
rows = {}
symlinks = []
for root in (RUNS, PERSISTENT):
    for directory, dirs, files in os.walk(root, followlinks=False):
        for filename in files:
            p = Path(directory)/filename
            if p.suffix not in ('.pt', '.pth', '.bin'):
                continue
            st = p.lstat()
            if stat.S_ISLNK(st.st_mode):
                symlinks.append(dict(path=str(p), target=str(p.resolve())))
                continue
            if not stat.S_ISREG(st.st_mode) or st.st_size < 1024**3:
                continue
            key = (st.st_dev, st.st_ino)
            r = rows.setdefault(key, dict(paths=[], bytes=st.st_size,
                allocated_bytes=st.st_blocks*512, nlink=st.st_nlink, reasons=[]))
            r['paths'].append(str(p))
            if PERSISTENT in p.parents: r['reasons'].append('persistent_checkpoint_archive_keep')
            if ACTIVE in p.parents: r['reasons'].append('ACTIVE_RUN_ALL_FILES_KEEP')
            if key in keep: r['reasons'].append(keep[key])
for s in symlinks:
    p = Path(s['target'])
    if p.exists():
        st = p.stat(); key = (st.st_dev,st.st_ino)
        if key in rows: rows[key]['reasons'].append('symlink_handoff_reference:'+s['path'])
for r in rows.values():
    if r['nlink'] > len(r['paths']): r['reasons'].append('unaccounted_external_hardlink_keep')
    r['reasons'] = sorted(set(r['reasons']))
    r['classification'] = 'KEEP' if r['reasons'] else 'CANDIDATE_ONLY_REQUIRES_USER_AND_DEPENDENCY_REVIEW'
    r['family'] = Path(r['paths'][0]).relative_to(RUNS).parts[0].split('-')[0] if r['paths'][0].startswith(str(RUNS)+'/') else 'persistent'
    r['GiB'] = r['allocated_bytes']/1024**3
candidates = [r for r in rows.values() if not r['reasons']]
protected = [r for r in rows.values() if r['reasons']]
summary = dict(timestamp=datetime.now(timezone.utc).isoformat(), metadata_only=True, deleted_files=0,
    checkpoint_unique_inodes=len(rows), checkpoint_unique_GiB=sum(r['GiB'] for r in rows.values()),
    keep_unique_inodes=len(protected), keep_GiB=sum(r['GiB'] for r in protected),
    candidates_unique_inodes=len(candidates), candidates_reclaimable_GiB=sum(r['GiB'] for r in candidates),
    candidates_by_family={f:dict(inodes=sum(r['family']==f for r in candidates),
        GiB=sum(r['GiB'] for r in candidates if r['family']==f)) for f in sorted({r['family'] for r in candidates})},
    caveat='Proposal only; remove all listed hardlinks of a candidate inode to reclaim space. Recheck active jobs/open files/symlinks before any deletion. Preserve small configs/logs/evaluation reports. No raw/cache deletion proposed.')
report = dict(summary=summary, protected=protected, candidates=candidates, symlinks=symlinks)
output = RUNS/'h031-checkpoint-cleanup-inventory-20260909.json'
output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(event='H031_CHECKPOINT_INVENTORY_OK',report=str(output),**summary)),flush=True)
