"""CPU-only authorized full-checkpoint transfer; no training or remote inference changes."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
SOURCE = Path('/data/WorldBridge4D-runs/h031-resume191000-to200000-gpu16-20260922/checkpoint-0193000.pt')
OUT = Path('/data/WorldBridge4D-runs/transfers/h031-step193000-to6024-20260923')
HOST = 'yejun@10.129.22.20'
DEST = '/home/yejun/data0/WorldBridge4D-inference/checkpoints'
SSH = ['ssh', '-p', '22', '-o', 'HostKeyAlias=[sy.irmv.top]:6024',
       '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
       '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=6']

def event(name, **kw):
    print(json.dumps(dict(event=name, **kw)), flush=True)

def remote(command):
    return subprocess.check_output(SSH + [HOST, command], text=True).strip()

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

# Same host verified via existing forwarded-port host key; fail before I/O if not writable.
remote(f'test -d {shlex.quote(DEST)} && test -w {shlex.quote(DEST)}')
OUT.mkdir(parents=True, exist_ok=True)
PIN = OUT / SOURCE.name
if not PIN.exists():
    os.link(SOURCE, PIN)  # atomic pin of published numbered checkpoint, not mutable latest.pt
st = PIN.stat()
assert st.st_size == 19363914955, 'Unexpected published checkpoint size'
event('TRANSFER_PINNED', source=str(SOURCE), pinned=str(PIN), bytes=st.st_size)
digest = sha(PIN)
assert PIN.stat().st_size == st.st_size and PIN.stat().st_mtime_ns == st.st_mtime_ns
final = f'{DEST}/{SOURCE.name}'
partial = f'{DEST}/.{SOURCE.name}.{digest}.partial'
q = shlex.quote
manifest = dict(source=str(SOURCE), pinned=str(PIN), bytes=st.st_size, sha256=digest,
                host=HOST, port=22, host_key_alias='[sy.irmv.top]:6024', destination=final, python=sys.version,
                seed=None, purpose='User-authorized inference checkpoint transfer; full training checkpoint unchanged')
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
event('TRANSFER_HASHED', **manifest)
remote(f'mkdir -p {q(DEST)}')
existing = remote(f'if test -e {q(final)}; then sha256sum -- {q(final)}; else echo ABSENT; fi')
if existing != 'ABSENT':
    assert existing.split()[0] == digest, 'Destination already exists with different bytes; refusing overwrite'
    event('TRANSFER_ALREADY_VERIFIED', destination=final, sha256=digest)
else:
    free = int(remote(f'df -B1 --output=avail {q(DEST)} | tail -1'))
    assert free > st.st_size + 64 * 1024**3, 'Insufficient remote free space including64GiB reserve'
    cmd = ['rsync', '-rt', '--partial', '--append-verify', '--timeout=120', '--info=progress2',
           '-e', shlex.join(SSH), str(PIN), f'{HOST}:{partial}']
    manifest['rsync_argv'] = cmd
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    event('TRANSFER_STARTED', destination=partial)
    subprocess.run(cmd, check=True)
    check = remote(f'sha256sum -- {q(partial)}')
    assert check.split()[0] == digest, 'Remote checksum mismatch; not publishing'
    # Hardlink publication is atomic and fails if another transfer created the final path.
    remote(f'ln -- {q(partial)} {q(final)} && rm -- {q(partial)}')
    check = remote(f'sha256sum -- {q(final)}')
    assert check.split()[0] == digest, 'Published checkpoint checksum mismatch'
manifest['completed_unix_seconds'] = time.time()
(OUT / 'complete.json').write_text(json.dumps(manifest, indent=2) + '\n')
event('TRANSFER_VERIFIED', destination=final, sha256=digest, bytes=st.st_size)
