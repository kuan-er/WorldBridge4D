#!/usr/bin/env python3
"""Promote a core Dynamic Replica cache after its real-Wan gates pass."""
from __future__ import annotations
import datetime as dt, hashlib, json
from pathlib import Path
import jsonschema

ROOT=Path(__file__).resolve().parents[1]
def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  while b:=f.read(8<<20): h.update(b)
 return h.hexdigest()

def main()->None:
 root=Path('/data/WorldBridge4D-persistent/datasets/dynamic_stereo_worldbridge4d_v1')
 manifest_path=root/'manifest.json'; report_path=root/'audit/validation_report.json'; smoke_path=root/'audit/real_wan_smoke.json'
 manifest=json.loads(manifest_path.read_text()); report=json.loads(report_path.read_text()); smoke=json.loads(smoke_path.read_text())
 if smoke.get('status')!='pass' or not smoke.get('loss_decreased'): raise RuntimeError('real-Wan/tiny-overfit gate did not pass')
 report['gates']['tiny_overfit']={'pass':True,'losses':smoke['losses'],'loss_decreased':True}
 report['gates']['real_gradient_smoke']={'pass':True,'steps':smoke['steps'],'global_steps':smoke['global_steps'],'finite_gradients':all(smoke['finite_gradients']),'wan_gradient':smoke['wan_gradient'],'adapter_gradient':smoke['adapter_gradient'],'decoder_gradient':smoke['decoder_gradient'],'peak_cuda_memory_gib':smoke['peak_cuda_memory_gib']}
 required=['split_leakage','temporal','geometry','coordinate_statistics','wan_determinism','tiny_overfit','real_gradient_smoke']
 passed=all(bool(report['gates'][key].get('pass')) for key in required)
 if not passed: raise RuntimeError(f'not all protocol gates passed: {report["gates"]}')
 report['status']='validated'; report['formal_training_allowed']=True; report['completed_at_utc']=dt.datetime.now(dt.timezone.utc).isoformat(); report_path.write_text(json.dumps(report,indent=2)+'\n')
 for item in manifest['artifacts']:
  if item['kind']=='validation_report': item['bytes']=report_path.stat().st_size; item['sha256']=sha(report_path)
 manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
 schema=json.loads((ROOT/'docs/dataset_manifest_v1.schema.json').read_text()); jsonschema.validate(manifest,schema)
 state_path=root/'PREPROCESSING_STATE.json'; state=json.loads(state_path.read_text()); state['status']='validated'; state['completed_at_utc']=report['completed_at_utc']; state_path.write_text(json.dumps(state,indent=2)+'\n')
 marker={'status':'validated','formal_training_allowed':True,'manifest_sha256':sha(manifest_path),'validation_report_sha256':sha(report_path),'completed_at_utc':report['completed_at_utc']}; (root/'CACHE_COMPLETE.json').write_text(json.dumps(marker,indent=2)+'\n'); print(json.dumps(marker,indent=2))
if __name__=='__main__': main()
