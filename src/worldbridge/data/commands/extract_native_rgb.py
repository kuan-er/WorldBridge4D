"""CPU-only Kubric RGB extraction; resume verified publications without NAS replay."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import time

import yaml

from worldbridge.data.cache.native import NativeLatentCache, file_sha256
from worldbridge.data.cache.native_rgb import NativeRGBCache
from worldbridge.data.native_inputs import iter_rgb, _check_source
from worldbridge.utils.io import atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--preflight', type=Path, required=True)
    p.add_argument('--rgb-root', type=Path, required=True)
    p.add_argument('--stage', choices=['smoke', 'bulk'], required=True)
    args = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'extraction must be CPU-only'
    cfg = yaml.safe_load(Path(args.config).read_text())
    ready = json.loads((args.preflight / 'ready.json').read_text())
    assert file_sha256(args.config) == ready['config_sha256']
    manifest = json.loads((args.preflight / 'kubric.json').read_text())
    assert manifest['dataset'] == 'kubric'
    assert manifest['sha256'] == ready['datasets']['kubric']['manifest_sha256']
    cache = NativeRGBCache(args.rgb_root, manifest)
    latents = NativeLatentCache(cfg['cache_root'], 'kubric', manifest['sha256'],
                               tuple(manifest['latent_shape']), ready['vae_sha256'])
    samples = {row['index']: row for row in ready['datasets']['kubric']['samples']}
    indices = list(samples) if args.stage == 'smoke' else list(range(len(manifest['records'])))
    cache.root.mkdir(parents=True, exist_ok=True)
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    begin = time.monotonic()
    summary = {'manifest_sha256': manifest['sha256'], 'requested': len(indices),
               'processed': 0, 'written': 0, 'reused': 0, 'latent_identity_checks': 0,
               'stage': args.stage, 'dataset': 'kubric', 'training_ready': False,
               'rgb_root': str(args.rgb_root), 'seed': cfg['seed']}
    def publish(index, identity, written):
        if index in samples:
            assert identity['rgb_sha256'] == samples[index]['rgb_sha256']
            assert identity['rgb_shape'] == samples[index]['rgb_shape']
        if latents.path(index).exists():
            latents.read(index, manifest['records'][index]['clip_id'], identity)
            summary['latent_identity_checks'] += 1
        summary['processed'] += 1
        summary['written' if written else 'reused'] += 1
        summary['elapsed_seconds'] = time.monotonic() - begin
        atomic_json(cache.root / f'{args.stage}_progress.json', summary)
        print(json.dumps({'event': 'NATIVE_RGB_PROGRESS', 'index': index, **summary}), flush=True)
    missing = []
    for index in indices:
        if stopping:
            break
        if cache.path(index).exists():
            rgb, identity = cache.read(index)
            publish(index, identity, False)
            del rgb
        else:
            missing.append(index)
    payload_bytes = len(missing) * 21 * 512 * 512 * 3
    # Budget remaining extraction plus safety; never remove old caches to make space.
    assert shutil.disk_usage(cache.root).free >= cfg['minimum_free_bytes'] + int(payload_bytes * 1.2)
    if not stopping:
        for row, rgb, identity in iter_rgb(manifest, missing):
            if stopping:
                break
            _check_source(manifest, row['path'])  # post-decode check before atomic publication
            assert shutil.disk_usage(cache.root).free >= cfg['minimum_free_bytes'] + 2 * rgb.nbytes
            written = cache.write(row['index'], rgb, identity)
            publish(row['index'], identity, written)
    if stopping:
        atomic_json(cache.root / f'{args.stage}_partial_stop.json', summary)
        print('NATIVE_RGB_PARTIAL_STOP', flush=True)
        return 3
    assert summary['processed'] == len(indices)
    if args.stage == 'bulk':
        assert {f.name for f in cache.root.glob('rgb_*.safetensors')} == {cache.path(i).name for i in indices}
    atomic_json(cache.root / f'{args.stage}_complete.json', summary)
    print(f'NATIVE_RGB_{args.stage.upper()}_OK', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
