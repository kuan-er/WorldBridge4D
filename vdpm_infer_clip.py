from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

REPO = Path('/data/WorldBridge4D-inference/repos/vdpm')
WB = Path('/data/WorldBridge4D')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(WB / 'src'))

SOURCES = [5, 10, 15, 20]
T = 21


def load_rgb(dataset: str, index: int):
    if dataset == 'kubric':
        from worldbridge.data import MOViFDataset
        ds = MOViFDataset('/dataset/nas0/yejun/MOVi-F/512x512', split='validation', clip_length=21, clip_start=0)
        sample = ds[index]
        return [x for x in sample.rgb], {'clip_id': f'kubric/validation/{index:06d}', 'native_size': [sample.height, sample.width]}
    if dataset == 'pointodyssey':
        root = Path('/dataset/data/preprocessed_256_three_dataset_v1/metadata/pointodyssey_worldbridge4d_v1')
        from worldbridge.pointodyssey import PointOdysseyDataset
        ds = PointOdysseyDataset(root, split='validation', image_size=256, raw_root='/dataset/nas0/PointOdyssey')
        row = ds.rows[index]
        scene = Path(row['source_scene'])
        start = int(row['start'])
        frames = []
        for j in range(T):
            with Image.open(scene / 'rgbs' / f'rgb_{start+j:05d}.jpg') as im:
                frames.append(np.asarray(im.convert('RGB'), dtype=np.uint8))
        return frames, {'clip_id': row['clip_id'], 'native_size': [frames[0].shape[0], frames[0].shape[1]]}
    if dataset == 'dynamic_replica':
        root = Path('/dataset/data/preprocessed_256_three_dataset_v1/metadata/dynamic_stereo_worldbridge4d_v1')
        from worldbridge.dynamic_replica import DynamicReplicaDataset
        ds = DynamicReplicaDataset(root, split='validation', image_size=256, raw_root='/dataset/data/Dynamic_dataset/dynamic_stereo')
        row = ds.rows[index]
        stream = str(row['stream'])
        start = int(row['start'])
        stream_root = Path('/dataset/data/Dynamic_dataset/dynamic_stereo/train') / stream
        frames = []
        for j in range(T):
            rel = row['frames'][j]['rgb']
            with Image.open(stream_root / Path(rel).name) as im:
                frames.append(np.asarray(im.convert('RGB'), dtype=np.uint8))
        return frames, {'clip_id': row['clip_id'], 'native_size': [frames[0].shape[0], frames[0].shape[1]]}
    raise ValueError(dataset)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['kubric', 'pointodyssey', 'dynamic_replica'])
    ap.add_argument('--index', type=int, required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    from dpm.model import VDPM
    from visualise import preprocess_images

    frames, meta = load_rgb(args.dataset, args.index)
    cfg = OmegaConf.create({'model': OmegaConf.load(REPO / 'configs' / 'model' / 'dpm.yaml')})
    device = torch.device(args.device)
    model = VDPM(cfg).to(device)
    state = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    del state

    images = preprocess_images(frames, mode='crop')
    native_shape = list(images.shape[1:])
    images = images.to(device)
    start = time.perf_counter()
    with torch.inference_mode():
        result = model.inference(None, images=images.unsqueeze(0))
    elapsed = time.perf_counter() - start

    # V-DPM returns [condition/dynamic time, reference frame, H, W, 3].
    # The saved protocol layout is [source, target, H, W, 3].
    all_xyz = torch.cat([x['pts3d'].detach().float().cpu() for x in result['pointmaps']], dim=0).numpy()
    all_conf = torch.cat([x['conf'].detach().float().cpu() for x in result['pointmaps']], dim=0).numpy()
    xyz = all_xyz[:, SOURCES].transpose(1, 0, 2, 3, 4)
    conf = all_conf[:, SOURCES].transpose(1, 0, 2, 3)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, xyz=xyz, confidence=conf, sources=np.asarray(SOURCES), targets=np.arange(T),
                        dataset=args.dataset, clip_index=np.int64(args.index))
    summary = {
        **meta, 'dataset': args.dataset, 'index': args.index, 'sources': SOURCES,
        'targets': list(range(T)), 'input_shape_chw': native_shape,
        'saved_xyz_shape': list(xyz.shape), 'saved_confidence_shape': list(conf.shape),
        'elapsed_seconds': elapsed,
        'cuda_peak_gib': torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else None,
        'checkpoint': str(Path(args.checkpoint).resolve()),
    }
    out.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print('VDPM_CLIP_INFERENCE_OK', flush=True)


if __name__ == '__main__':
    main()
