from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = Path(os.environ.get('WORLDBRIDGE4D_INFERENCE_ROOT', '/data/WorldBridge4D-inference'))
FOURRC = EXTERNAL_ROOT / 'repos' / '4rc'
sys.path.insert(0, str(FOURRC))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

SOURCES = [5, 10, 15, 20]
T = 21

# Reuse the audited RGB-only dataset adapters from the VDPM setup. They return
# raw native RGB arrays; 4RC's own load_images performs the official resize/crop.
from vdpm_infer_clip import load_rgb  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['kubric', 'pointodyssey', 'dynamic_replica'])
    ap.add_argument('--index', type=int, required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    from arc.models.arc import Arc
    from arc.dust3r.inference_multiview import inference
    from arc.dust3r.utils.image import load_images

    frames, meta = load_rgb(args.dataset, args.index)
    if len(frames) != T:
        raise ValueError(f'expected {T} frames, got {len(frames)}')
    device = torch.device(args.device)
    checkpoint_dir = Path(args.checkpoint).resolve().parent
    model = Arc.from_pretrained(str(checkpoint_dir)).to(device).eval()

    with tempfile.TemporaryDirectory(prefix='4rc_rgb_') as td:
        paths = []
        for i, frame in enumerate(frames):
            arr = np.asarray(frame)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            Image.fromarray(arr[..., :3], mode='RGB').save(Path(td) / f'{i:05d}.png')
            paths.append(str(Path(td) / f'{i:05d}.png'))
        # This is the official 4RC loader: long edge 512, patch size 14,
        # preserving aspect ratio/cropping only to patch-compatible dimensions.
        imgs = load_images(paths, size=512, verbose=True, patch_size=14)

    query = torch.tensor(SOURCES, dtype=torch.long)
    for img in imgs:
        img['track_query_idx'] = query
    start = time.perf_counter()
    with torch.inference_mode():
        output, profiling = inference(
            imgs, model, device, dtype='bf16-mixed', profiling=True,
            verbose=True, use_center_as_anchor=False,
        )
    elapsed = time.perf_counter() - start

    preds = output['preds']
    # inference_multiview returns a list over target views. Each track_multi is
    # [B, Q, H, W, 3] and is already converted to the predicted world frame by
    # Arc._postprocess_output.
    tracks, track_confs, pointmaps, point_confs = [], [], [], []
    for pred in preds:
        tracks.append(pred['track_multi'][0].detach().float().cpu().numpy())
        track_confs.append(pred['conf_track_multi'][0].detach().float().cpu().numpy())
        pointmaps.append(pred['pts'][0].detach().float().cpu().numpy())
        point_confs.append(pred['conf'][0].detach().float().cpu().numpy())
    tracks = np.stack(tracks, axis=1)  # [Q, T, H, W, 3]
    track_confs = np.stack(track_confs, axis=1)  # [Q, T, H, W]
    pointmaps = np.stack(pointmaps, axis=0)  # [T, H, W, 3]
    point_confs = np.stack(point_confs, axis=0)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, xyz=tracks, confidence=track_confs,
        sources=np.asarray(SOURCES), targets=np.arange(T),
        pointmap=pointmaps, pointmap_confidence=point_confs,
        first_frame=tracks[0], first_frame_confidence=track_confs[0],
        dataset=args.dataset, clip_index=np.int64(args.index),
    )
    summary = {
        **meta, 'dataset': args.dataset, 'index': args.index,
        'sources': SOURCES, 'targets': list(range(T)),
        'input_shape_hw': list(imgs[0]['img'].shape[-2:]),
        'saved_xyz_shape': list(tracks.shape),
        'saved_pointmap_shape': list(pointmaps.shape),
        'saved_first_frame_shape': list(tracks[0].shape),
        'elapsed_seconds': elapsed,
        'profiling': profiling,
        'cuda_peak_gib': torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else None,
        'checkpoint': str(Path(args.checkpoint).resolve()), 'status': 'succeeded',
    }
    out.with_suffix('.json').write_text(json.dumps(summary, indent=2, default=str) + '\n')
    print(json.dumps(summary, indent=2, default=str))
    print('4RC_CLIP_INFERENCE_OK', flush=True)


if __name__ == '__main__':
    main()
