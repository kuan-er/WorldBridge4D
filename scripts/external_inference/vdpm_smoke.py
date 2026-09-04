from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--frames', type=int, default=5)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    from dpm.model import VDPM
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({'model': OmegaConf.load(repo / 'configs' / 'model' / 'dpm.yaml')})
    device = torch.device(args.device)
    model = VDPM(cfg).to(device)
    state = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=True)
    model.eval()
    images = torch.rand(1, args.frames, 3, 518, 518, device=device)
    with torch.inference_mode():
        result = model.inference(None, images=images)
    summary = {
        'frames': args.frames,
        'pointmap_count': len(result['pointmaps']),
        'pointmap_shapes': [list(x['pts3d'].shape) for x in result['pointmaps']],
        'confidence_shapes': [list(x['conf'].shape) for x in result['pointmaps']],
        'pose_shape': list(result['pose_enc'].shape),
        'missing': list(missing),
        'unexpected': list(unexpected),
        'cuda_peak_gib': torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else None,
    }
    print(json.dumps(summary, indent=2))
    print('VDPM_REAL_CHECKPOINT_SMOKE_OK', flush=True)


if __name__ == '__main__':
    main()
