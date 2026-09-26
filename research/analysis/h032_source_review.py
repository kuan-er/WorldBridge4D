"""CPU-only strict review and immutable hardlink pin of the196k source."""
import json
import os
from pathlib import Path
import sys
import platform
import yaml
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from h031_resume191000_to200000 import check_payload
from worldbridge.data.cache.native import file_sha256
from worldbridge.utils.io import atomic_json
SOURCE = Path('/data/WorldBridge4D-runs/h031-resume191000-to200000-gpu16-20260922/checkpoint-0196000.pt')
OUT = Path('/data/WorldBridge4D-runs/h032-camera-ray-source196000-20260924')

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(2)
    cfg = yaml.safe_load((ROOT / 'configs/h031_k512_dr512_full_k9_resume191000_to200000.yaml').read_text())
    assert not OUT.exists()
    sha = file_sha256(SOURCE)
    payload = torch.load(SOURCE, map_location='cpu', mmap=True, weights_only=True)
    review = check_payload(payload, cfg, step=196000)
    review.update(coordinate_mean=payload['coordinate_mean'], coordinate_scale=payload['coordinate_scale'])
    assert all(float(x) > 0 for x in payload['coordinate_scale'])
    OUT.mkdir()
    os.link(SOURCE, OUT / 'resume.pt')
    atomic_json(OUT / 'train_status.json', {'completed_steps':196000, 'world_size':2})
    report = dict(event='H032_SOURCE_OK', source=str(SOURCE), checkpoint_sha256=sha,
                  checkpoint_bytes=SOURCE.stat().st_size, review=review,
                  python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                  platform=platform.platform(), code_baseline='ef5fc4634c7518cf295e70a3e56942bf98cdc016')
    atomic_json(OUT / 'complete.json', report)
    print(json.dumps(report), flush=True)
if __name__ == '__main__':
    main()
