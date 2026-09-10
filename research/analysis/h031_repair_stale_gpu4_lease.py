"""Repair only the audited dead GPU4 lease's literal trailing backslash-n.
No lock removal, signals, device reset, admission bypass, or shared PRL code edit.
The existing PRL worker may then reclaim the dead-owner lease by normal policy.
"""
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from worldbridge.utils.io import atomic_json

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
lease = Path('/data/WorldBridge4D/.pi-research/runtime/gpu-leases/4')
p = lease / 'owner.json'
report = Path('/data/WorldBridge4D-runs/h031-gpu4-stale-lease-format-repair-20260910/complete.json')
assert not report.exists()
assert lease.resolve() == lease and not lease.is_symlink()
dirfd = os.open(lease, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
try:
    fd = os.open('owner.json', os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dirfd)
    try:
        before = os.fstat(fd)
        assert stat.S_ISREG(before.st_mode) and before.st_size < 4096
        raw = os.read(fd, 4096)
    finally:
        os.close(fd)
    assert raw.endswith(b'\\n')
    try:
        json.loads(raw)
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError('Only repair the known malformed legacy JSON')
    owner = json.loads(raw[:-2])
    assert owner == dict(run_id='R-20260823024636-331a8c', pid=3477772, gpu_id='4',
                         acquired_at='2026-08-23T02:46:37.204Z')
    assert not Path('/proc/3477772').exists(), 'Owner PID must be absent; no foreign control'
    fixed = raw[:-2] + b'\n'
    name = '.h031-dead-owner-format-repair.tmp'
    out = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=dirfd)
    try:
        os.write(out, fixed); os.fsync(out)
    finally:
        os.close(out)
    current = os.stat('owner.json', dir_fd=dirfd, follow_symlinks=False)
    assert (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) == (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    assert not Path('/proc/3477772').exists()
    os.replace(name, 'owner.json', src_dir_fd=dirfd, dst_dir_fd=dirfd)
    os.fsync(dirfd)
finally:
    os.close(dirfd)
result = dict(event='H031_STALE_GPU4_LEASE_FORMAT_REPAIRED', old_owner=owner,
              old_sha256=hashlib.sha256(raw).hexdigest(), new_sha256=hashlib.sha256(fixed).hexdigest(),
              change='literal trailing backslash-n to JSON whitespace; identity unchanged',
              worker_action='normal PRL pidAlive stale-owner reclamation; no manual lease deletion',
              no_process_signals=True, no_gpu_reset=True, admission_unchanged=True)
atomic_json(report, result)
print(json.dumps(result), flush=True)
