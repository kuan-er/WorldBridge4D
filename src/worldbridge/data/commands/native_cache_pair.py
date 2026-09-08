"""Two dataset workers under ONE PRL GPU6 lease; never an unmanaged GPU launch."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def supervise(commands: dict[str, list[str]]) -> int:
    """Forward soft stops, stop siblings on failure, and reap every child."""
    children: dict[str, subprocess.Popen] = {}
    stopping = False
    failed = False

    def stop(_sig=None, _frame=None):
        nonlocal stopping
        stopping = True
        for child in children.values():
            if child.poll() is None:
                try:
                    child.terminate()
                except ProcessLookupError:
                    pass

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for name, command in commands.items():
            if stopping:
                break
            children[name] = subprocess.Popen(command)  # inherit PRL group and logs
            print(json.dumps({'event': 'NATIVE_PAIR_WORKER_STARTED', 'dataset': name,
                              'pid': children[name].pid, 'argv': command}), flush=True)
        while children:
            for name, child in list(children.items()):
                code = child.poll()
                if code is None:
                    continue
                print(json.dumps({'event': 'NATIVE_PAIR_WORKER_EXIT', 'dataset': name,
                                  'exit_code': code}), flush=True)
                del children[name]
                if code != 0:
                    failed = True
                    stop()
            if children:
                time.sleep(0.2)
        if failed or stopping:
            print('NATIVE_PAIR_PARTIAL_STOP', flush=True)
            return 3 if stopping and not failed else 1
        print('NATIVE_PAIR_BULK_OK', flush=True)
        return 0
    finally:
        if children:
            stop()
            for child in children.values():
                child.wait()  # no timeout escalation or SIGKILL
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main() -> int:
    import yaml
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--preflight-root', type=Path, required=True)
    p.add_argument('--cpu-gate', required=True)
    p.add_argument('--owner-session', required=True, help='Owner of unchanged historical CPU gate')
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == cfg['cache_gpu'] == '6'
    # Combined admission: K8GiB (observed allocated4.38/process~6.2) + DR32 + safety4.
    # Native workers retain their own CPU/code/config/manifest/VAE/space checks.
    free = int(subprocess.check_output([
        'nvidia-smi', '-i', '6', '--query-gpu=memory.free', '--format=csv,noheader,nounits',
    ], text=True).strip())
    assert free >= 45056, f'combined GPU6 admission requires45056MiB, free={free}'
    for dataset in ('kubric', 'dynamic_replica'):
        ready = json.loads((args.preflight_root / dataset / 'ready.json').read_text())
        digest = ready['datasets'][dataset]['manifest_sha256']
        report = json.loads((Path(cfg['cache_root']) / dataset / digest / 'smoke_complete.json').read_text())
        assert report['manifest_sha256'] == digest and report['processed'] == report['requested'] == 3
        assert report['repeat_max_abs_error'] == 0
        assert report['peak_cuda_GiB'] <= (6 if dataset == 'kubric' else 30)
    commands = {dataset: [sys.executable, '-u', '-m',
        'worldbridge.data.commands.precompute_native_latents', '--config', args.config,
        '--preflight', str(args.preflight_root / dataset), '--cpu-gate', args.cpu_gate,
        '--owner-session', args.owner_session, '--dataset', dataset, '--stage', 'bulk',
    ] for dataset in ('kubric', 'dynamic_replica')}
    return supervise(commands)


if __name__ == '__main__':
    sys.exit(main())
