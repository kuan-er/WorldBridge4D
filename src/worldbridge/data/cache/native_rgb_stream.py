"""Bounded local-only consumer of atomic RGB publications from a pinned producer."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time


class RGBStreamStopped(Exception):
    """Cooperative SIGTERM while awaiting the next publication."""


def checked_status(producer_status):
    status = producer_status()
    if status not in {'running', 'queued', 'succeeded'}:
        raise RuntimeError(f'RGB producer ended without success: {status}')
    return status


def wait_for_publication(path, *, producer_status, stopped, timeout=900, poll_seconds=0.5):
    deadline = time.monotonic() + timeout
    announced = False
    while True:
        if stopped():
            raise RGBStreamStopped()
        status = checked_status(producer_status)
        if Path(path).exists():
            return
        if status == 'succeeded':
            raise FileNotFoundError(f'successful RGB producer omitted {path}')
        if not announced:
            print(json.dumps({'event': 'NATIVE_RGB_WAIT', 'path': str(path),
                              'timeout_seconds': timeout}), flush=True)
            announced = True
        if time.monotonic() >= deadline:
            raise TimeoutError(f'no RGB publication within {timeout}s: {path}')
        time.sleep(poll_seconds)


def wait_for_success(*, producer_status, stopped, timeout=900, poll_seconds=0.5):
    deadline = time.monotonic() + timeout
    while True:
        if stopped():
            raise RGBStreamStopped()
        if checked_status(producer_status) == 'succeeded':
            return
        if time.monotonic() >= deadline:
            raise TimeoutError('RGB completion exists but producer did not succeed')
        time.sleep(poll_seconds)


def iter_stream(cache, indices, *, producer_status, stopped, timeout=900):
    for index in indices:
        try:
            wait_for_publication(cache.path(index), producer_status=producer_status,
                                 stopped=stopped, timeout=timeout)
        except RGBStreamStopped:
            return
        # Atomic filename excludes temporary files. Existing corrupt files fail immediately.
        yield from cache.iter_rgb([index])


def producer_probe(run_id, owner_session, rgb_root, config, preflight):
    """Bounded authoritative PRL read, cached10s; never touch a foreign producer."""
    script = """
import {readRun} from '/data/WorldBridge4D/.pi/npm/node_modules/pi-research-loop/dist/core.js';
const [id,owner,root,config,preflight]=process.argv.slice(1);
const r=readRun(id,'/data/WorldBridge4D'); const a=r.command.argv;
const arg=k=>a[a.indexOf(k)+1];
if(r.owner?.session_id!==owner || !a.includes('worldbridge.data.commands.extract_native_rgb')
  || arg('--stage')!=='bulk' || arg('--rgb-root')!==root
  || arg('--config')!==config || arg('--preflight')!==preflight)
  throw Error('RGB producer identity/owner/argv mismatch');
console.log(JSON.stringify({status:r.status}));
"""
    previous = None
    observed_at = float('-inf')
    def probe():
        nonlocal previous, observed_at
        if time.monotonic() - observed_at >= 10:
            result = subprocess.run(['/opt/node-v22.19.0-linux-x64/bin/node', '--input-type=module',
                                     '-e', script, run_id, owner_session, str(rgb_root), config,
                                     str(preflight)], check=True, capture_output=True, text=True, timeout=10)
            previous = json.loads(result.stdout)['status']
            observed_at = time.monotonic()
        return previous
    return probe
