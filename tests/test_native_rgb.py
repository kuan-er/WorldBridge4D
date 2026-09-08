import json

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file

from worldbridge.data.cache.native import CONTRACT, file_sha256, json_hash, rgb_identity
from worldbridge.data.cache.native_rgb import NativeRGBCache


@pytest.fixture
def staged(tmp_path):
    index = tmp_path / 'index.jsonl'
    index.write_text('{}\n')
    manifest = {'contract': CONTRACT, 'dataset': 'kubric', 'native_hw': [16, 32],
                'latent_shape': [16, 6, 2, 4], 'frames': 21, 'tiling': False,
                'transform': 'identity_no_resize_crop_pad_or_temporal_resampling',
                'index_path': str(index), 'index_sha256': file_sha256(index),
                'records': [{'index': i, 'clip_id': f'clip{i}', 'row_sha256': str(i) * 64,
                             'path': '/NAS_MUST_NOT_BE_OPENED/movi.tfrecord'} for i in range(2)]}
    manifest['sha256'] = json_hash(manifest)
    cache = NativeRGBCache(tmp_path / 'rgb', manifest)
    rgb = np.random.default_rng(20260908).integers(256, size=(21, 16, 32, 3), dtype=np.uint8)
    return cache, rgb


def test_roundtrip_resume_and_no_raw_access(staged, monkeypatch):
    from worldbridge.data import native_inputs
    cache, rgb = staged
    identity = rgb_identity(rgb)
    assert cache.write(0, rgb, identity)
    assert not cache.write(0, rgb, identity)
    monkeypatch.setattr(native_inputs, '_check_source', lambda *_: pytest.fail('NAS accessed'))
    monkeypatch.setattr(native_inputs, 'iter_rgb', lambda *_: pytest.fail('raw reader accessed'))
    row, actual, digest = next(cache.iter_rgb([0]))
    assert row['index'] == 0 and digest == identity
    np.testing.assert_array_equal(actual, rgb)


def test_missing_never_falls_back(staged):
    cache, _ = staged
    with pytest.raises(FileNotFoundError):
        next(cache.iter_rgb([0]))


def test_refuse_new_payload(staged):
    cache, rgb = staged
    cache.write(0, rgb, rgb_identity(rgb))
    changed = rgb.copy()
    changed[0, 0, 0, 0] ^= 1
    with pytest.raises(ValueError, match='overwrite'):
        cache.write(0, changed, rgb_identity(changed))
    np.testing.assert_array_equal(cache.read(0)[0], rgb)


@pytest.mark.parametrize('corruption', ['payload', 'identity', 'shape', 'dtype'])
def test_detect_corruption(staged, corruption):
    cache, rgb = staged
    cache.write(0, rgb, rgb_identity(rgb))
    with safe_open(str(cache.path(0)), framework='np') as f:
        meta = f.metadata()
    if corruption == 'payload':
        rgb[0, 0, 0, 0] ^= 1
    elif corruption == 'identity':
        meta['index'] = '1'
    elif corruption == 'shape':
        rgb = rgb[:, :8]
    else:
        rgb = rgb.astype(np.float32)
    save_file({'rgb': rgb.copy()}, str(cache.path(0)), metadata=meta)
    with pytest.raises(ValueError):
        cache.read(0)


def test_complete_requires_every_publication(staged):
    cache, rgb = staged
    cache.write(0, rgb, rgb_identity(rgb))
    with pytest.raises(FileNotFoundError):
        cache.require_complete()
    marker = cache.root / 'bulk_complete.json'
    report = {'manifest_sha256': cache.manifest['sha256'], 'requested': 2, 'processed': 2}
    marker.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='coverage'):
        cache.require_complete()
    cache.write(1, rgb, rgb_identity(rgb))
    cache.require_complete()
    report['manifest_sha256'] = '0' * 64
    marker.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='not complete'):
        cache.require_complete()


@pytest.mark.parametrize('index', [-1, 2, True, '0'])
def test_invalid_index(staged, index):
    cache, _ = staged
    with pytest.raises(ValueError):
        cache.path(index)


def test_wrong_input_hash_rejected(staged):
    cache, rgb = staged
    identity = rgb_identity(rgb)
    identity['rgb_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='publication'):
        cache.write(0, rgb, identity)
    assert not cache.path(0).exists()


def test_local_index_mutation_rejected(staged):
    cache, _ = staged
    from pathlib import Path
    Path(cache.manifest['index_path']).write_text('changed')
    with pytest.raises(ValueError, match='index changed'):
        NativeRGBCache(cache.root, cache.manifest)
