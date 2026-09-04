from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = Path(os.environ.get('WORLDBRIDGE4D_INFERENCE_ROOT', '/data/WorldBridge4D-inference'))
REPO = EXTERNAL_ROOT / 'repos' / 'vdpm'
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

SOURCES = [5, 10, 15, 20]
T = 21


def preprocess_images_official(images_np):
    """V-DPM visualise.py preprocessing without importing its cv2 GUI."""
    if not images_np:
        raise ValueError('at least one image is required')
    images = []
    shapes = set()
    for image_np in images_np:
        img = Image.fromarray(image_np)
        if img.mode == 'RGBA':
            background = Image.new('RGBA', img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        img = img.convert('RGB')
        width, height = img.size
        target = 518
        new_width = target
        new_height = round(height * (new_width / width) / 14) * 14
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        tensor = torch.from_numpy(np.asarray(img, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
        if new_height > target:
            start_y = (new_height - target) // 2
            tensor = tensor[:, start_y:start_y + target, :]
        shapes.add(tuple(tensor.shape[1:]))
        images.append(tensor)
    if len(shapes) != 1:
        raise ValueError(f'official preprocessing produced inconsistent shapes: {shapes}')
    return torch.stack(images)


def load_rgb(dataset: str, index: int):
    if dataset == 'kubric':
        from worldbridge.data import MOViFDataset
        ds = MOViFDataset('/dataset/nas0/yejun/MOVi-F/512x512', split='validation', clip_length=21, clip_start=0)
        sample = ds[index]
        return [x for x in sample.rgb], {'clip_id': f'kubric/validation/{index:06d}', 'native_size': [sample.height, sample.width]}
    if dataset == 'pointodyssey':
        root = Path('/dataset/data/preprocessed_256_three_dataset_v1/metadata/pointodyssey_worldbridge4d_v1')
        from worldbridge.pointodyssey import PointOdysseyDataset
        ds = PointOdysseyDataset(root, split='validation', image_size=256)
        row = ds.rows[index]
        scene = Path('/dataset/nas0/PointOdyssey') / Path(row['source_scene']).relative_to('/dataset/PointOdyssey')
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
            with Image.open(stream_root / 'images' / Path(rel).name) as im:
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
    from omegaconf import OmegaConf

    frames, meta = load_rgb(args.dataset, args.index)
    cfg = OmegaConf.create({'model': OmegaConf.load(REPO / 'configs' / 'model' / 'dpm.yaml')})
    device = torch.device(args.device)
    model = VDPM(cfg).to(device)
    state = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    del state

    images = preprocess_images_official(frames)
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
    # Explicit protocol outputs: all diagonal pointmaps and source=0 tracking
    # are retained in addition to all four arbitrary source trajectories.
    pointmap = all_xyz[np.arange(T), np.arange(T)]
    pointmap_conf = all_conf[np.arange(T), np.arange(T)]
    first_frame = all_xyz[:, 0]
    first_frame_conf = all_conf[:, 0]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, xyz=xyz, confidence=conf, sources=np.asarray(SOURCES), targets=np.arange(T),
        pointmap=pointmap, pointmap_confidence=pointmap_conf,
        first_frame=first_frame, first_frame_confidence=first_frame_conf,
        dataset=args.dataset, clip_index=np.int64(args.index),
    )
    summary = {
        **meta, 'dataset': args.dataset, 'index': args.index, 'sources': SOURCES,
        'targets': list(range(T)), 'input_shape_chw': native_shape,
        'saved_xyz_shape': list(xyz.shape), 'saved_confidence_shape': list(conf.shape),
        'saved_pointmap_shape': list(pointmap.shape), 'saved_first_frame_shape': list(first_frame.shape),
        'elapsed_seconds': elapsed,
        'cuda_peak_gib': torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else None,
        'checkpoint': str(Path(args.checkpoint).resolve()),
    }
    out.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print('VDPM_CLIP_INFERENCE_OK', flush=True)


if __name__ == '__main__':
    main()
