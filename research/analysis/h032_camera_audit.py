"""Read-only CPU calibration audit; no RGB/VAE/checkpoint generation."""
import json
import os
from pathlib import Path
import sys
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from worldbridge.data.factory import load_training_datasets
from worldbridge.trainer.camera_objective import validate_supervision_camera
from worldbridge.utils.io import atomic_json
OUT=Path('/data/WorldBridge4D-runs/h032-camera-calibration-audit-perframe-20260924')

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    cfg=yaml.safe_load((ROOT/'configs/h031_k512_dr512_full_k9_resume191000_to200000.yaml').read_text())
    datasets=load_training_datasets(cfg)
    results={}
    for name,dataset in datasets.items():
        # Kubric intrinsics are scalar clip metadata, therefore constant by
        # construction. Verify three actual native samples; external K is
        # per-frame, so audit every indexed PO/DR clip rather than assuming.
        indices=sorted({0,len(dataset)//2,len(dataset)-1}) if name=='kubric' else range(len(dataset))
        size=getattr(dataset,'image_size',256)
        maxima={}; failures=[]; variable_clips=0
        for i in indices:
            camera=dataset.supervision_camera(i)
            try:
                values=validate_supervision_camera(camera,size,size)
                for k,v in values.items(): maxima[k]=max(maxima.get(k,0),v)
                variable_clips += int(values['focal_relative_drift'] > 1e-5)
            except ValueError as e:
                if len(failures)<10: failures.append(dict(index=i,error=str(e)))
            if i%1000==0: print(json.dumps(dict(event='H032_CAMERA_AUDIT_PROGRESS',dataset=name,index=i)),flush=True)
        results[name]=dict(checked=len(indices),corpus=len(dataset),maxima=maxima,first_failures=failures,
                           variable_focal_clips=variable_clips,
                           scope='scalar_contract_plus_three_native_samples' if name=='kubric' else 'all_indexed_cameras')
        print(json.dumps(dict(event='H032_CAMERA_AUDIT_DATASET',dataset=name,**results[name])),flush=True)
    OUT.mkdir(exist_ok=False)
    atomic_json(OUT/'report.json',results)
    assert not any(v['first_failures'] for v in results.values()), 'camera assumptions fail; inspect report, do not train'
    atomic_json(OUT/'complete.json',dict(event='H032_CAMERA_AUDIT_OK',intrinsics_mode='per_frame_source_independent',results=results))
    print('H032_CAMERA_AUDIT_OK',flush=True)
if __name__=='__main__': main()
