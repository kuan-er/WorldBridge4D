"""Authorized internal-network transfer of H033 step204000 into Bridge4D_yj; CPU only."""
# Instance of h031_transfer193k_container.py for H033 step 204000 (user-authorized
# 2026-09-26, same container route and destination directory as step 193000).
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
PIN = Path('/data/WorldBridge4D-runs/transfers/h033-step204000-to6024-20260926/checkpoint-0204000.pt')
OUT = Path('/data/WorldBridge4D-runs/transfers/h033-step204000-to6024-container-20260926')
DIGEST = 'f45cf7987524b770054e455fe195e62e40cc2837303f4a1cf3b9c66d93078409'
SIZE = 19375053899
DEST = '/data/WorldBridge4D-inference/checkpoints/checkpoint-0204000.pt'
SSH = ['ssh', '-T', '-p', '22', '-o', 'HostKeyAlias=[sy.irmv.top]:6024', '-o', 'BatchMode=yes',
       '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15',
       '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=6', 'yejun@10.129.22.20']

def event(name, **kwargs):
    print(json.dumps(dict(event=name, **kwargs)), flush=True)

REMOTE = r'''
import hashlib, json, os, shutil, sys, time
from pathlib import Path
mode, destination, digest, size = sys.argv[1:5]
size = int(size)
final = Path(destination)
part = final.with_name('.' + final.name + '.' + digest + '.container.partial')
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()
assert final.parent.is_dir() and os.access(final.parent, os.W_OK)
if mode == 'prepare':
    if final.exists():
        assert final.stat().st_size == size and sha(final) == digest, 'Existing destination differs; no overwrite'
        print(json.dumps(dict(verified=True, offset=size)), flush=True)
    else:
        offset = part.stat().st_size if part.exists() else 0
        assert 0 <= offset <= size
        assert shutil.disk_usage(final.parent).free > size - offset + 64*1024**3
        print(json.dumps(dict(verified=False, offset=offset)), flush=True)
else:
    offset = int(sys.argv[5])
    start = time.monotonic()
    with part.open('ab') as f:
        assert f.tell() == offset, 'Partial file changed since prepare'
        remaining = size-offset
        while remaining:
            b = sys.stdin.buffer.read(min(8*1024**2, remaining))
            if not b:
                raise RuntimeError('Truncated stream; partial preserved')
            f.write(b)
            remaining -= len(b)
        assert not sys.stdin.buffer.read(1), 'Unexpected extra bytes'
        f.flush()
        os.fsync(f.fileno())
    transfer_seconds = time.monotonic() - start
    assert part.stat().st_size == size and sha(part) == digest, 'SHA mismatch; not publishing'
    os.chmod(part, 0o644)
    os.link(part, final)  # atomic no-clobber publication
    part.unlink()
    fd = os.open(final.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    print(json.dumps(dict(event='CONTAINER_TRANSFER_VERIFIED', path=str(final), bytes=size,
                         sha256=digest, streamed_bytes=size-offset,
                         receive_and_fsync_seconds=transfer_seconds,
                         total_receive_verify_seconds=time.monotonic()-start)), flush=True)
'''

# step 204000: H033 camera-query-ray route, 19375053899 bytes, shaped on the live
# H033 run output; source pinned by hardlink before launch.
def command(mode, *args):
    remote = ['docker', 'exec', '-i', 'Bridge4D_yj', '/opt/conda/bin/python3', '-c', REMOTE,
              mode, DEST, DIGEST, str(SIZE), *map(str, args)]
    return SSH + [shlex.join(remote)]

OUT.mkdir(exist_ok=False)
assert PIN.stat().st_size == SIZE
h = hashlib.sha256()
with PIN.open('rb') as f:
    for b in iter(lambda: f.read(8*1024**2), b''):
        h.update(b)
assert h.hexdigest() == DIGEST, 'Pinned source changed'
manifest = dict(source=str(PIN), sha256=DIGEST, bytes=SIZE, internal_host='10.129.22.20', port=22,
                container='Bridge4D_yj', container_destination=DEST,
                host_destination='/home/yejun/data0/WorldBridge4D-inference/checkpoints/checkpoint-0204000.pt',
                python=sys.version, seed=None, transport='SSH binary stream; SHA-verified atomic publication')
(OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
prepared = json.loads(subprocess.check_output(command('prepare'), text=True, timeout=180))
event('CONTAINER_TRANSFER_PREPARED', **prepared, **manifest)
if not prepared['verified']:
    proc = subprocess.Popen(command('receive', prepared['offset']), stdin=subprocess.PIPE)
    try:
        with PIN.open('rb') as f:
            f.seek(prepared['offset'])
            for b in iter(lambda: f.read(8*1024**2), b''):
                proc.stdin.write(b)
        proc.stdin.close()
        assert proc.wait(timeout=600) == 0, 'Remote receive/verify failed'
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=30)
manifest['completed_unix_seconds'] = time.time()
(OUT/'complete.json').write_text(json.dumps(manifest, indent=2)+'\n')
event('TRANSFER_COMPLETE', **manifest)
