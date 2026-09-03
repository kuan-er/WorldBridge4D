from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Importing this module also supplies the audited RGB-only dataset adapters.
from importlib.util import module_from_spec, spec_from_file_location

HERE = Path(__file__).parent
spec = spec_from_file_location('vdpm_infer_clip', HERE / 'vdpm_infer_clip.py')
vdpm = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(vdpm)

FOURRC = Path('/data/WorldBridge4D-inference/repos/4rc')
SOURCES = [5, 10, 15, 20]
T = 21


def infer_clip(frames, model, device, load_images, inference):
    with tempfile.TemporaryDirectory(prefix='4rc_rgb_') as td:
        paths = []
        for i, frame in enumerate(frames):
            arr = np.asarray(frame)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            Image.fromarray(arr[..., :3]).convert('RGB').save(Path(td) / f'{i:05d}.png')
            paths.append(str(Path(td) / f'{i:05d}.png'))
        imgs = load_images(paths, size=512, verbose=False, patch_size=14)
    query = torch.tensor(SOURCES, dtype=torch.long)
    for img in imgs:
        img['track_query_idx'] = query
    with torch.inference_mode():
        output, profiling = inference(
            imgs, model, device, dtype='bf16-mixed', profiling=True,
            verbose=False, use_center_as_anchor=False,
        )
    preds = output['preds']
    tracks, track_confs, pointmaps, point_confs = [], [], [], []
    for pred in preds:
        tracks.append(pred['track_multi'][0].float().cpu().numpy())
        track_confs.append(pred['conf_track_multi'][0].float().cpu().numpy())
        pointmaps.append(pred['pts'][0].float().cpu().numpy())
        point_confs.append(pred['conf'][0].float().cpu().numpy())
    tracks = np.stack(tracks, axis=1)
    track_confs = np.stack(track_confs, axis=1)
    pointmaps = np.stack(pointmaps, axis=0)
    point_confs = np.stack(point_confs, axis=0)
    return tracks, track_confs, pointmaps, point_confs, list(imgs[0]['img'].shape[-2:]), profiling


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['kubric', 'pointodyssey', 'dynamic_replica'])
    ap.add_argument('--start', type=int, required=True)
    ap.add_argument('--stop', type=int, required=True, help='exclusive')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output-root', required=True)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    from arc.models.arc import Arc
    from arc.dust3r.inference_multiview import inference
    from arc.dust3r.utils.image import load_images

    device = torch.device(args.device)
    checkpoint_dir = Path(args.checkpoint).resolve().parent
    model = Arc.from_pretrained(str(checkpoint_dir)).to(device).eval()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    progress = root / 'progress.jsonl'
    for index in range(args.start, args.stop):
        out = root / f'{index:06d}.npz'
        summary_path = out.with_suffix('.json')
        if out.is_file() and summary_path.is_file():
            continue
        started = time.perf_counter()
        try:
            frames, meta = vdpm.load_rgb(args.dataset, index)
            tracks, track_confs, pointmaps, point_confs, input_hw, profiling = infer_clip(
                frames, model, device, load_images, inference)
            np.savez_compressed(
                out, xyz=tracks.astype(np.float32), confidence=track_confs.astype(np.float32),
                sources=np.asarray(SOURCES), targets=np.arange(T),
                pointmap=pointmaps.astype(np.float32), pointmap_confidence=point_confs.astype(np.float32),
                first_frame=tracks[0].astype(np.float32), first_frame_confidence=track_confs[0].astype(np.float32),
                dataset=args.dataset, clip_index=np.int64(index),
            )
            record = {
                **meta, 'dataset': args.dataset, 'index': index, 'sources': SOURCES,
                'targets': list(range(T)), 'input_shape_hw': input_hw,
                'saved_xyz_shape': list(tracks.shape), 'saved_pointmap_shape': list(pointmaps.shape),
                'saved_first_frame_shape': list(tracks[0].shape),
                'elapsed_seconds': time.perf_counter() - started,
                'model_profiling': profiling, 'cuda_peak_gib': torch.cuda.max_memory_allocated(device) / 2**30,
                'checkpoint': str(Path(args.checkpoint).resolve()), 'status': 'succeeded',
            }
        except Exception as exc:
            record = {'dataset': args.dataset, 'index': index, 'status': 'failed',
                      'error_type': type(exc).__name__, 'error': str(exc),
                      'elapsed_seconds': time.perf_counter() - started}
        summary_path.write_text(json.dumps(record, indent=2, default=str) + '\n')
        with progress.open('a') as f:
            f.write(json.dumps(record, default=str) + '\n')
            f.flush()
        print(json.dumps(record, default=str), flush=True)
    print('4RC_DATASET_WORKER_DONE', flush=True)


if __name__ == '__main__':
    main()
