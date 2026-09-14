from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

EXTERNAL_ROOT = Path(os.environ.get('WORLDBRIDGE4D_INFERENCE_ROOT', '/data/WorldBridge4D-inference'))
REPO = EXTERNAL_ROOT / 'repos' / 'vdpm'
SCRIPT = Path(__file__).with_name('vdpm_infer_clip.py')
spec = importlib.util.spec_from_file_location('vdpm_infer_clip', SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['kubric', 'pointodyssey', 'dynamic_replica'])
    ap.add_argument('--start', type=int, required=True)
    ap.add_argument('--stop', type=int, required=True, help='exclusive')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output-root', required=True)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    if args.stop <= args.start:
        raise ValueError('stop must be greater than start')

    from dpm.model import VDPM
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({'model': OmegaConf.load(REPO / 'configs' / 'model' / 'dpm.yaml')})
    device = torch.device(args.device)
    model = VDPM(cfg).to(device)
    state = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    del state

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
            frames, meta = mod.load_rgb(args.dataset, index)
            images = mod.preprocess_images_official(frames).to(device)
            with torch.inference_mode():
                result = model.inference(None, images=images.unsqueeze(0))
            all_xyz = torch.cat([x['pts3d'].detach().float().cpu() for x in result['pointmaps']], dim=0).numpy()
            all_conf = torch.cat([x['conf'].detach().float().cpu() for x in result['pointmaps']], dim=0).numpy()
            xyz = all_xyz[:, mod.SOURCES].transpose(1, 0, 2, 3, 4)
            conf = all_conf[:, mod.SOURCES].transpose(1, 0, 2, 3)
            pointmap = all_xyz[np.arange(mod.T), np.arange(mod.T)]
            pointmap_conf = all_conf[np.arange(mod.T), np.arange(mod.T)]
            first_frame = all_xyz[:, 0]
            first_frame_conf = all_conf[:, 0]
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, xyz=xyz, confidence=conf, sources=np.asarray(mod.SOURCES), targets=np.arange(mod.T),
                                pointmap=pointmap, pointmap_confidence=pointmap_conf,
                                first_frame=first_frame, first_frame_confidence=first_frame_conf,
                                dataset=args.dataset, clip_index=np.int64(index))
            record = {
                **meta, 'dataset': args.dataset, 'index': index, 'sources': mod.SOURCES,
                'targets': list(range(mod.T)), 'input_shape_chw': list(images.shape[1:]),
                'saved_xyz_shape': list(xyz.shape), 'saved_pointmap_shape': list(pointmap.shape),
                'saved_first_frame_shape': list(first_frame.shape),
                'elapsed_seconds': time.perf_counter() - started,
                'cuda_peak_gib': torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else None,
                'checkpoint': str(Path(args.checkpoint).resolve()), 'status': 'succeeded',
            }
            summary_path.write_text(json.dumps(record, indent=2) + '\n')
        except Exception as exc:
            record = {'dataset': args.dataset, 'index': index, 'status': 'failed',
                      'error_type': type(exc).__name__, 'error': str(exc),
                      'elapsed_seconds': time.perf_counter() - started}
            summary_path.write_text(json.dumps(record, indent=2) + '\n')
        with progress.open('a') as f:
            f.write(json.dumps(record) + '\n')
            f.flush()
        print(json.dumps(record), flush=True)
    print('VDPM_DATASET_WORKER_DONE', flush=True)


if __name__ == '__main__':
    main()
