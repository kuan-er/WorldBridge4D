"""CPU-only real native GT staging and full-resume identity gate for10-update admission."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch
import yaml

from worldbridge.data.cache.native import file_sha256, rgb_identity
from worldbridge.data.cache.native_rgb import NativeRGBCache
from worldbridge.data.datasets.movif256 import MOViF256Dataset
from worldbridge.data.geometry import CameraModel, GeometryBuilder
from worldbridge.data.movif import MOViFDataset
from worldbridge.data.native_inputs import _check_source, decode_kubric_rgb
from worldbridge.data.sampling import deterministic_sample_plan
from worldbridge.trainer.config import validate_config
from worldbridge.trainer.schedulers import dataset_for_step
from worldbridge.utils.io import atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--reuse-staged-geometry', action='store_true')
    args = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    cfg = yaml.safe_load(Path(args.config).read_text()); validate_config(cfg, 2)
    values = cfg['datasets']['kubric']
    manifest = json.loads(Path(values['native_manifest']).read_text())
    rgb_cache = NativeRGBCache(values['native_rgb_root'], manifest); rgb_cache.require_complete()
    root = Path(values['native_geometry_root']); root.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(root).free > 68719476736 + 4 * 2**30
    plans = []
    for step in range(150000, cfg['max_steps']):
        if dataset_for_step(step, cfg['seed']) != 'kubric':
            continue
        for rank in range(2):
            for slot in range(4):
                index, source, _ = deterministic_sample_plan(manifest['records'], 'kubric', cfg['seed'], step, slot, rank, 4)
                # slots_per_rank=4 is identical to the old B2/A2 control.
                plans.append({'step': step, 'rank': rank, 'slot': slot, 'index': index, 'source_seed': source})
    assert plans
    groups = defaultdict(dict)
    for index in sorted({v['index'] for v in plans}):
        row = manifest['records'][index]
        groups[row['path']][row['local_record']] = row
    tf = MOViFDataset._tf(); tf.config.set_visible_devices([], 'GPU')
    native = MOViFDataset.__new__(MOViFDataset)
    native.clip_length = 21; native.clip_start = 0; native.seed = cfg['seed']
    report = {'manifest_sha256': manifest['sha256'], 'config_sha256': file_sha256(args.config),
              'start_step': 150000, 'end_step': cfg['max_steps'], 'plans': plans, 'entries': {},
              'geometry_transform': 'native512_no_resize_original_radial_depth_and_validity',
              'training_ready': False, 'capacity_input_gate': True}
    if args.reuse_staged_geometry:
        previous = json.loads((root / 'ready.json').read_text())
        for key in ('manifest_sha256','config_sha256','start_step','end_step','plans','geometry_transform'):
            assert previous[key] == report[key]
        assert set(previous['entries']) == {str(v['index']) for v in plans}
        for entry in previous['entries'].values():
            assert file_sha256(root / entry['file']) == entry['sha256']
        report = previous
        groups.clear()
        print('NATIVE_GT_SNAPSHOT_REUSED_SHA_VERIFIED', flush=True)
    for path, wanted in sorted(groups.items()):
        _check_source(manifest, path)
        options = tf.data.Options(); options.threading.private_threadpool_size = 1
        options.threading.max_intra_op_parallelism = 1
        data = tf.data.TFRecordDataset([path], num_parallel_reads=1).with_options(options)
        for local, raw_tensor in enumerate(data.take(max(wanted) + 1)):
            if local not in wanted:
                continue
            raw = bytes(raw_tensor.numpy()); row = wanted[local]; index = row['index']
            _, identity = rgb_cache.read(index)
            assert rgb_identity(decode_kubric_rgb(raw)) == identity
            sample = native._decode(raw, row['raw_index'], decode_rgb=False)
            assert sample.depth.shape == (21, 512, 512) and sample.segmentation.shape == (21, 512, 512)
            assert sample.clip_start == 0
            camera = CameraModel(512, 512, sample.focal_length, sample.sensor_width)
            coords = np.array([[0,0], [511,511], [255,255], [128,384], [400,200]])
            world = camera.backproject_pixels(sample.depth[0, coords[:,1], coords[:,0]], coords,
                                              sample.camera_positions[0], sample.camera_quaternions[0])
            uv, _, _ = camera.project(world, sample.camera_positions[0], sample.camera_quaternions[0])
            error = float(np.max(np.abs(uv - coords))); assert error < 1e-6
            xyz, _, valid = GeometryBuilder(sample).trajectory(0, coords, 'source', compute_visibility=False)
            expected = camera.world_to_camera(world, sample.camera_positions[0], sample.camera_quaternions[0])
            assert np.allclose(xyz[:,0], expected, atol=1e-5)
            _check_source(manifest, path)
            destination = root / f'geom_{index:08d}.npz'
            temporary = destination.with_suffix('.tmp.npz')
            np.savez(temporary, **asdict(sample))
            with temporary.open('rb') as f: os.fsync(f.fileno())
            os.replace(temporary, destination)
            loaded = MOViF256Dataset._load_compact_sample(destination)
            np.testing.assert_array_equal(loaded.depth, sample.depth)
            np.testing.assert_array_equal(loaded.depth_valid, sample.depth_valid)
            report['entries'][str(index)] = {'file': destination.name, 'sha256': file_sha256(destination),
                                              'RGB_sha256': identity['rgb_sha256'], 'roundtrip_px': error}
            print(json.dumps({'event':'NATIVE_GT_STAGED', 'index':index, 'count':len(report['entries']),
                              'roundtrip_px':error}), flush=True)
        _check_source(manifest, path)
    assert len(report['entries']) == len({x['index'] for x in plans})
    # Strict source checkpoint admission, full model/optimizer retained, not weights-only.
    origin = Path(cfg['selected_checkpoint_path'])
    assert file_sha256(origin) == cfg['selected_checkpoint_sha256']
    payload = torch.load(origin, map_location='cpu', mmap=True, weights_only=True)
    state = payload['training_state']
    assert state['global_step'] == 150000 and state['world_size'] == 2 and len(state['rng_states']) == 2
    assert len(payload['optimizer']['state']) == 193
    report['origin_sha256'] = cfg['selected_checkpoint_sha256']
    report['origin_state'] = {k:v for k,v in state.items() if k != 'rng_states'}
    report['optimizer_states'] = len(payload['optimizer']['state'])
    # Compare all decoder names/shapes without allocating194M real CPU parameters.
    from worldbridge.models.decoder import DenseQueryDecoder
    with torch.device('meta'):
        decoder = DenseQueryDecoder(num_frames=21, latent_shape=(512,21,32,32), query_dim=1536,
            embedding_dim=768, num_layers=5, num_heads=12, upsample_channels=(1536,768,384,192),
            output_size=(256,256), query_grid_size=32, structured_motion_slots=8,
            structured_local_queries=True, source_rgb_pyramid=True, source_rgb_channels=(32,64,128),
            source_rgb_fusion_32=True, pre_attention_rgb_query=True, native_512=True)
    actual = {k:tuple(v.shape) for k,v in decoder.state_dict().items()}
    expected = {k.removeprefix('decoder.'):tuple(v.shape) for k,v in payload['model'].items() if k.startswith('decoder.')}
    assert actual == expected
    report['decoder_checkpoint_names_shapes_exact'] = True
    atomic_json(root / 'ready.json', report)
    # Exercise the real prefetch planner, GT selection, caches and text on both ranks,
    # including PO/DR256; no GPU and no missing input fallback/generation permitted.
    from worldbridge.data.factory import load_training_datasets
    from worldbridge.trainer.lazy_vae import set_lazy_vae_identity
    from worldbridge.trainer.batching import GeometryPrefetcher
    datasets = load_training_datasets(cfg)
    set_lazy_vae_identity(datasets, file_sha256(cfg['vae_checkpoint']))
    for rank in range(2):
        prefetch = GeometryPrefetcher(datasets, seed=cfg['seed'], rank=rank, accumulation=4,
            microbatch_per_gpu=1, targets_per_source=15, use_source_rgb=True, cycle_enabled=True,
            cycle_dataset_names=('kubric','pointodyssey','dynamic_replica'), start_step=150000,
            target_steps=cfg['max_steps'], depth=1, workers=2)
        try:
            prefetch.refill()
            for step in range(150000,cfg['max_steps']):
                planned = prefetch.pop(step)
                for future in planned.geometry_futures:
                    (index, source, xyz, valid, rgb, visible, camera, _, _), _ = future.result()
                    hw = 512 if planned.dataset_name == 'kubric' else 256
                    assert xyz.shape == (21,3,hw,hw) and valid.shape == visible.shape == (21,hw,hw)
                    assert rgb.shape == (hw,hw,3) and rgb.dtype == np.uint8
                    assert valid.reshape(21,-1).any(axis=1).sum() >= 15
                    latent = planned.dataset.clean_latent(index)
                    assert latent.shape == (16,6,hw//8,hw//8) and np.isfinite(latent).all()
                print(json.dumps({'event':'NATIVE_REAL_BATCH_GATE', 'rank':rank,'step':step,
                                  'dataset':planned.dataset_name,'clips':4,'K':15}),flush=True)
        finally:
            prefetch.close()
    atomic_json(root / 'cpu_complete.json', report)
    print('NATIVE512_CAPACITY_CPU_OK', flush=True)


if __name__ == '__main__': main()
