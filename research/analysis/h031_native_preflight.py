"""CPU source/shape/storage gate for native VAE caches; no training claim."""
from __future__ import annotations

import argparse
import json
import importlib.metadata
from pathlib import Path
import shutil

import numpy as np
import torch
import yaml

from worldbridge.data.cache.native import file_sha256
from worldbridge.data.native_inputs import build_manifest, iter_rgb
from worldbridge.utils.io import atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert not torch.cuda.is_initialized()
    assert not args.output.exists(), 'fresh CPU preflight output required'
    config = yaml.safe_load(Path(args.config).read_text())
    assert config['transform'] == 'identity_no_resize_crop_pad_or_temporal_resampling'
    inventory = json.loads(Path(config['source_inventory']).read_text())
    vae_sha = file_sha256(config['vae_checkpoint'])
    report = {'config_sha256': file_sha256(args.config), 'vae_sha256': vae_sha,
              'source_inventory_sha256': file_sha256(config['source_inventory']),
              'seed': config['seed'],
              'environment': {p: importlib.metadata.version(p) for p in ['torch','diffusers','numpy','pillow','tensorflow-cpu','safetensors']},
              'datasets': {}, 'PO_blocked': config['blocked']['pointodyssey'],
              'three_dataset_cache_complete': False, 'training_ready': False}
    estimated = 0
    for name in config['datasets']:
        manifest = build_manifest(config, name, vae_sha)
        old = inventory['datasets'][name]
        assert old['index_sha256'] == manifest['index_sha256']
        assert old['clips'] == len(manifest['records'])
        indices = sorted({0, len(manifest['records']) // 2, len(manifest['records']) - 1})
        samples = []
        for row, rgb, identity in iter_rgb(manifest, indices):
            previous = next(x for x in old['preselected_first_middle_last'] if x['index'] == row['index'])
            assert list(rgb.shape) == previous['shape']
            if name == 'kubric':
                # Independent legacy TF PNG decoder vs new sequential PIL decoder.
                assert identity['rgb_sha256'] == previous['RGB_sha256']
            else:
                assert file_sha256(row['paths'][0]) == previous['first_rgb_sha256']
            samples.append({'index': row['index'], 'clip_id': row['clip_id'], **identity})
        payload = int(np.prod(manifest['latent_shape'])) * 4 * len(manifest['records'])
        estimated += payload + 4096 * len(manifest['records'])
        atomic_json(args.output / f'{name}.json', manifest)
        report['datasets'][name] = {'manifest_sha256': manifest['sha256'], 'clips': len(manifest['records']),
                                    'shape': manifest['latent_shape'], 'samples': samples,
                                    'estimated_latent_bytes_without_metadata': payload}
        print(json.dumps({'event': 'NATIVE_CPU_DATASET_READY', 'dataset': name, **report['datasets'][name]}), flush=True)
    root = Path(config['cache_root'])
    assert root.is_absolute() and not root.resolve().is_relative_to(Path.cwd().resolve())
    parent = root
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    assert free >= int(config['minimum_free_bytes']) + int(estimated * 1.2), 'native cache storage admission failed'
    report.update(estimated_bytes_with_metadata=estimated, data_free_bytes=free,
                  required_free_bytes=int(config['minimum_free_bytes']) + int(estimated * 1.2))
    assert not torch.cuda.is_initialized()
    atomic_json(args.output / 'ready.json', report)
    print('NATIVE_VAE_CPU_PREFLIGHT_OK', flush=True)


if __name__ == '__main__':
    main()
