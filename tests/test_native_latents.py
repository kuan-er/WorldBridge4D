from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file

from worldbridge.data.cache.native import NativeLatentCache, file_sha256, json_hash, latent_shape, rgb_identity
from worldbridge.data.cache.latent import LazyLatentCache
from worldbridge.data.native_inputs import build_manifest, external_paths, iter_rgb, validate_manifest


@pytest.mark.parametrize('hw,expected', [((512,512),(16,6,64,64)), ((720,1280),(16,6,90,160)),
                                          ((256,256),(16,6,32,32)), ((16,32),(16,6,2,4))])
def test_native_shape(hw, expected):
    assert latent_shape(*hw) == expected


@pytest.mark.parametrize('h,w,t', [(0,16,21), (-16,16,21), (540,960,21), (512,511,21),
                                   (512,512,20), (512,512,25), (8,16,21), (512.0,512,21), (True,16,21)])
def test_no_implicit_resize_pad_or_temporal_change(h,w,t):
    with pytest.raises(ValueError, match='no implicit transform'):
        latent_shape(h,w,t)


def cache(tmp):
    return NativeLatentCache(tmp, 'kubric', 'a'*64, (16,6,2,4), 'b'*64)


def example():
    rgb = np.arange(21*16*32*3, dtype=np.uint8).reshape(21,16,32,3)
    latent = np.arange(16*6*2*4, dtype=np.float32).reshape(16,6,2,4)
    return rgb, latent


def test_native_cache_roundtrip_idempotence_and_separate_namespace(tmp_path):
    c = cache(tmp_path); rgb, value = example(); identity = rgb_identity(rgb)
    assert c.write(0, 'clip0', value, identity)
    original = c.path(0).read_bytes()
    assert np.array_equal(c.read(0, 'clip0', identity), value)
    assert not c.write(0, 'clip0', value.copy(), identity)
    assert c.path(0).read_bytes() == original
    assert not list(c.root.glob('*.tmp.safetensors'))
    assert c.root != NativeLatentCache(tmp_path, 'kubric', 'c'*64, c.shape, 'b'*64).root
    with pytest.raises(ValueError, match='overwrite'):
        c.write(0, 'clip0', value+1, identity)
    assert c.path(0).read_bytes() == original


@pytest.mark.parametrize('change', ['clip', 'rgb', 'vae', 'shape', 'manifest'])
def test_native_identity_fail_closed(tmp_path, change):
    c = cache(tmp_path); rgb, value = example(); identity = rgb_identity(rgb)
    c.write(0, 'clip0', value, identity)
    clip = 'clip0'
    if change == 'clip': clip = 'different'
    if change == 'rgb': identity = {**identity, 'rgb_sha256': 'd'*64}
    if change == 'vae': c.fixed['vae_sha256'] = 'd'*64
    if change == 'manifest': c.fixed['manifest_sha256'] = 'd'*64
    if change == 'shape': c.shape = (16,6,4,2)
    with pytest.raises(ValueError):
        c.read(0, clip, identity)


@pytest.mark.parametrize('change', ['fp16', 'nan', 'shape', 'rgb_shape', 'rgb_hash'])
def test_invalid_native_publication_leaves_no_tensor(tmp_path, change):
    c = cache(tmp_path); rgb, value = example(); identity = rgb_identity(rgb)
    if change == 'fp16': value = value.astype(np.float16)
    if change == 'nan': value[0,0,0,0] = np.nan
    if change == 'shape': value = value[..., :1]
    if change == 'rgb_shape': identity['rgb_shape'] = [21,256,256,3]
    if change == 'rgb_hash': identity['rgb_sha256'] = 'not_a_hash'
    with pytest.raises(ValueError): c.write(0, 'clip0', value, identity)
    assert not c.path(0).exists()


def test_tensor_corruption_and_legacy_256_refused(tmp_path):
    c = cache(tmp_path); rgb, value = example(); identity = rgb_identity(rgb)
    c.write(0, 'clip0', value, identity)
    with safe_open(str(c.path(0)), framework='np') as h: metadata = h.metadata()
    save_file({'latent': value+1}, str(c.path(0)), metadata=metadata)
    with pytest.raises(ValueError, match='checksum'): c.read(0, 'clip0')
    legacy = LazyLatentCache(c.root, 'kubric', expected_shape=c.shape)
    with pytest.raises(RuntimeError, match='contract'): legacy.read(0, 'clip0')


@pytest.mark.parametrize('kind', ['float', 'time', 'channel'])
def test_native_rgb_contract(kind):
    rgb, _ = example()
    if kind == 'float': rgb = rgb.astype(np.float32)
    if kind == 'time': rgb = rgb[:20]
    if kind == 'channel': rgb = rgb[..., :1]
    with pytest.raises(ValueError): rgb_identity(rgb)


def test_noncontiguous_rgb_hash_is_content_based():
    rgb, _ = example()
    reversed_rgb = rgb[:, :, ::-1]
    assert not reversed_rgb.flags.c_contiguous
    assert rgb_identity(reversed_rgb) == rgb_identity(np.ascontiguousarray(reversed_rgb))


def external_fixture(tmp):
    index = tmp/'train.jsonl'; frames = []
    for i in range(21):
        path = tmp/'raw/train'/f'{i}.png'; path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((16,32,3), i, np.uint8)).save(path)
        frames.append({'rgb': f'{i}.png'})
    row = {'index':0, 'clip_id':'one', 'stride':1, 'frames':frames}
    index.write_text(json.dumps(row)+'\n')
    cfg = {'seed':20260908, 'vae_checkpoint':'fake', 'datasets':{'dynamic_replica':{
        'raw_root':str(tmp/'raw'), 'index':str(index), 'native_hw':[16,32]}}}
    return cfg, row


def test_native_external_manifest_exact_frames_no_resize_and_mutation_guards(tmp_path):
    cfg, _ = external_fixture(tmp_path)
    m = build_manifest(cfg, 'dynamic_replica', 'a'*64)
    validate_manifest(m)
    (row, rgb, identity), = list(iter_rgb(m, [0]))
    assert identity == rgb_identity(rgb)
    assert rgb.shape == (21,16,32,3)
    for i in range(21): assert (rgb[i] == i).all()
    for indices in ([0,0], [-1], [1]):
        with pytest.raises(ValueError): list(iter_rgb(m, indices))
    m['frames'] = 25
    with pytest.raises(ValueError, match='checksum'): validate_manifest(m)
    m['sha256'] = json_hash({k:v for k,v in m.items() if k != 'sha256'})
    with pytest.raises(ValueError, match='transform'): validate_manifest(m)


def test_native_source_change_and_missing_rgb_no_cache_fallback(tmp_path):
    cfg, _ = external_fixture(tmp_path)
    m = build_manifest(cfg, 'dynamic_replica', 'a'*64)
    path = Path(m['records'][0]['paths'][0]); path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='source file changed'): list(iter_rgb(m, [0]))
    path.unlink()
    with pytest.raises(FileNotFoundError): build_manifest(cfg, 'dynamic_replica', 'a'*64)


def test_filtered_legacy_row_ids_keep_training_list_position_and_original_bytes(tmp_path):
    cfg, row = external_fixture(tmp_path)
    rows = [{**row, 'index':182}, {**row, 'index':6761, 'clip_id':'two'}]
    index = Path(cfg['datasets']['dynamic_replica']['index'])
    index.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    original = index.read_bytes()
    m = build_manifest(cfg, 'dynamic_replica', 'a'*64)
    assert [(r['index'],r['source_row_index']) for r in m['records']] == [(0,182),(1,6761)]
    assert [r['row_sha256'] for r in m['records']] == [json_hash(r) for r in rows]
    assert [r['index'] for r,_,_ in iter_rgb(m, [0,1])] == [0,1]
    assert index.read_bytes() == original


def test_native_index_change_rejected(tmp_path):
    cfg, _ = external_fixture(tmp_path)
    m = build_manifest(cfg, 'dynamic_replica', 'a'*64)
    Path(m['index_path']).write_text('[]\n')
    with pytest.raises(ValueError, match='index changed'): validate_manifest(m)


def test_native_paths_stride_time_and_root_contract(tmp_path):
    cfg, row = external_fixture(tmp_path)
    with pytest.raises(ValueError, match='stride1'): external_paths('dynamic_replica', tmp_path, {**row,'stride':2})
    with pytest.raises(ValueError, match='21'): external_paths('dynamic_replica', tmp_path, {**row,'frames':row['frames'][:20]})
    evil = {**row, 'frames':[{'rgb':'../../escape.png'}]*21}
    with pytest.raises(ValueError, match='escapes'): external_paths('dynamic_replica', tmp_path, evil)
    po = {'source_scene':'/old/PointOdyssey/train/ani', 'start':42, 'stride':1}
    assert external_paths('pointodyssey', tmp_path, po) == [tmp_path/'train/ani/rgbs'/f'rgb_{42+j:05d}.jpg' for j in range(21)]
