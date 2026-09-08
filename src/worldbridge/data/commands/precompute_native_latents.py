"""Single explicitly leased GPU: native FP32 VAE smoke or full existing-index cache."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

from worldbridge.data.cache.native import NativeLatentCache, file_sha256
from worldbridge.data.native_inputs import iter_rgb, validate_manifest
from worldbridge.models.wan import WanVAEEncoder
from worldbridge.utils.io import atomic_json

AUDITED_PATHS = [
    'src/worldbridge/data/cache/native.py', 'src/worldbridge/data/native_inputs.py',
    'src/worldbridge/data/commands/precompute_native_latents.py', 'src/worldbridge/models/wan.py',
    'research/analysis/h031_native_preflight.py', 'tests/test_native_latents.py',
    'configs/h031_native_cache.yaml',
    'src/worldbridge/data/cache/native_rgb.py',
    'src/worldbridge/data/commands/extract_native_rgb.py',
    'tests/test_native_rgb.py', 'tests/test_native_rgb_stream.py',
    'src/worldbridge/data/cache/native_rgb_stream.py',
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--preflight', type=Path, required=True)
    p.add_argument('--cpu-gate', required=True)
    p.add_argument('--owner-session', required=True)
    p.add_argument('--dataset', choices=['kubric', 'dynamic_replica'], required=True)
    p.add_argument('--stage', choices=['smoke', 'bulk'], required=True)
    p.add_argument('--rgb-root', type=Path, help='Verified local RGB snapshot; no raw-source fallback')
    p.add_argument('--rgb-producer-run', help='Stream local atomic publications from this owned PRL Run')
    p.add_argument('--rgb-wait-timeout', type=float, default=900)
    args = p.parse_args()
    assert args.rgb_wait_timeout > 0
    assert not args.rgb_producer_run or args.rgb_root is not None
    config = yaml.safe_load(Path(args.config).read_text())
    meta = yaml.safe_load((Path('/data/WorldBridge4D/runs') / args.cpu_gate / 'run.yaml').read_text())
    assert meta['status'] == 'succeeded' and meta['owner']['session_id'] == args.owner_session
    subprocess.run(['git', 'diff', '--exit-code', meta['git']['commit'], 'HEAD', '--', *AUDITED_PATHS], check=True)
    ready = json.loads((args.preflight / 'ready.json').read_text())
    assert ready['config_sha256'] == file_sha256(args.config)
    assert ready['vae_sha256'] == file_sha256(config['vae_checkpoint'])
    manifest = json.loads((args.preflight / f'{args.dataset}.json').read_text())
    validate_manifest(manifest)
    assert manifest['sha256'] == ready['datasets'][args.dataset]['manifest_sha256']
    assert manifest['vae_sha256'] == ready['vae_sha256']
    local_rgb = None
    producer_status = None
    if args.rgb_root is not None:
        from worldbridge.data.cache.native_rgb import NativeRGBCache
        assert args.dataset == 'kubric', 'local migration is authorized for Kubric only'
        local_rgb = NativeRGBCache(args.rgb_root, manifest)
        if args.rgb_producer_run:
            from worldbridge.data.cache.native_rgb_stream import producer_probe, checked_status
            producer_status = producer_probe(args.rgb_producer_run, args.owner_session,
                                             args.rgb_root, args.config, args.preflight)
            checked_status(producer_status)
        else:
            local_rgb.require_complete()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == config['cache_gpu']
    assert all(not os.environ.get(k) for k in ['PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF', 'PYTORCH_NO_CUDA_MEMORY_CACHING'])
    assert torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    cache = NativeLatentCache(config['cache_root'], args.dataset, manifest['sha256'],
                             tuple(manifest['latent_shape']), ready['vae_sha256'])
    indices = ([x['index'] for x in ready['datasets'][args.dataset]['samples']] if args.stage == 'smoke'
               else list(range(len(manifest['records']))))
    if args.stage == 'bulk':
        smoke = json.loads((cache.root / 'smoke_complete.json').read_text())
        assert smoke['manifest_sha256'] == manifest['sha256'] and smoke['repeat_max_abs_error'] == 0
        if local_rgb is not None:
            assert smoke.get('rgb_input_root') == str(args.rgb_root.resolve())
            if producer_status is not None:
                assert smoke.get('rgb_producer_run') == args.rgb_producer_run
    stopping = False
    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    torch.cuda.reset_peak_memory_stats()
    encoder = WanVAEEncoder(config['vae_checkpoint'], device='cuda:0', dtype=torch.float32,
                            expected_shape=tuple(manifest['latent_shape']))
    assert not encoder.vae.use_tiling and not encoder.vae.use_slicing
    assert all(p.dtype == torch.float32 and not p.requires_grad for p in encoder.parameters())
    print(json.dumps({'event': 'NATIVE_VAE_LOADED', 'dataset': args.dataset, 'stage': args.stage,
                      'physical_gpu': config['cache_gpu'], 'seed': config['seed'],
                      'vae_sha256': ready['vae_sha256'], 'native_hw': manifest['native_hw'],
                      'posterior': 'mean', 'dtype': 'float32', 'tiling': False,
                      'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                      'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                      'cudnn_benchmark': torch.backends.cudnn.benchmark}), flush=True)
    begin = time.monotonic()
    summary = {'manifest_sha256': manifest['sha256'], 'stage': args.stage, 'dataset': args.dataset,
               'requested': len(indices), 'processed': 0, 'written': 0, 'reused': 0,
               'repeat_max_abs_error': None, 'training_ready': False,
               'rgb_input_root': str(args.rgb_root.resolve()) if local_rgb is not None else None,
               'rgb_producer_run': args.rgb_producer_run}
    if producer_status is not None:
        from worldbridge.data.cache.native_rgb_stream import iter_stream
        inputs = iter_stream(local_rgb, indices, producer_status=producer_status,
                             stopped=lambda: stopping, timeout=args.rgb_wait_timeout)
    else:
        inputs = local_rgb.iter_rgb(indices) if local_rgb is not None else iter_rgb(manifest, indices)
    input_begin = time.monotonic()
    for row, rgb, identity in inputs:
        input_seconds = time.monotonic() - input_begin
        work_begin = time.monotonic()
        i = row['index']
        if stopping:
            atomic_json(cache.root / f'{args.stage}_partial_stop.json', summary)
            print('NATIVE_CACHE_PARTIAL_STOP', flush=True)
            return 3
        if (args.stage == 'bulk' and cache.path(i).exists()):
            cache.read(i, row['clip_id'], identity)
            summary['reused'] += 1
        else:
            tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2)[None]
            with torch.inference_mode():
                latent = encoder(tensor).float().cpu().numpy()[0]
                if args.stage == 'smoke' and summary['processed'] == 0:
                    repeated = encoder(tensor).float().cpu().numpy()[0]
                    summary['repeat_max_abs_error'] = float(np.max(np.abs(latent - repeated)))
                    assert np.array_equal(latent, repeated), 'native posterior-mean repeat differs'
                    del repeated
            parent = cache.root
            while not parent.exists():
                parent = parent.parent
            assert shutil.disk_usage(parent).free >= int(config['minimum_free_bytes']) + 2 * latent.nbytes
            if cache.write(i, row['clip_id'], latent, identity):
                summary['written'] += 1
            else:
                summary['reused'] += 1
            del tensor, latent
        summary['last_clip_seconds'] = {'read_decode_hash_rgb': input_seconds,
                                        'latent_read_or_encode_write': time.monotonic() - work_begin}
        summary['processed'] += 1
        summary['elapsed_seconds'] = time.monotonic() - begin
        summary['peak_cuda_GiB'] = torch.cuda.max_memory_allocated() / 2**30
        atomic_json(cache.root / f'{args.stage}_progress.json', summary)
        print(json.dumps({'event': 'NATIVE_CACHE_PROGRESS', 'index': i, **summary}), flush=True)
        input_begin = time.monotonic()
    if stopping:
        atomic_json(cache.root / f'{args.stage}_partial_stop.json', summary)
        print('NATIVE_CACHE_PARTIAL_STOP', flush=True)
        return 3
    assert summary['processed'] == len(indices)
    if args.stage == 'bulk':
        if producer_status is not None:
            from worldbridge.data.cache.native_rgb_stream import (
                wait_for_publication, wait_for_success, RGBStreamStopped,
            )
            try:
                kwargs = dict(producer_status=producer_status, stopped=lambda: stopping,
                              timeout=args.rgb_wait_timeout)
                wait_for_publication(local_rgb.root / 'bulk_complete.json', **kwargs)
                local_rgb.require_complete()
                wait_for_success(**kwargs)
            except RGBStreamStopped:
                atomic_json(cache.root / 'bulk_partial_stop.json', summary)
                print('NATIVE_CACHE_PARTIAL_STOP', flush=True)
                return 3
        expected = {cache.path(i).name for i in indices}
        assert {p.name for p in cache.root.glob('latent_*.safetensors')} == expected
        # Every cache was verified against freshly decoded or SHA-verified staged RGB.
        summary['all_existing_training_index_entries_verified'] = True
    atomic_json(cache.root / f'{args.stage}_complete.json', summary)
    print(f'NATIVE_CACHE_{args.stage.upper()}_OK {args.dataset}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
