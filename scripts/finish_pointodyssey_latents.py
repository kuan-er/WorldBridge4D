#!/usr/bin/env python3
"""Finish a PointOdyssey cache after the CPU index/stat phase."""
from __future__ import annotations
import argparse, datetime as dt, json, platform, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from scripts.preprocess_pointodyssey import (build_latents, file_artifact, rejected_scenes, sha256)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root',type=Path,default=Path('/dataset/PointOdyssey'))
    ap.add_argument('--output-root',type=Path,required=True)
    ap.add_argument('--wan-checkpoint',type=Path,default=Path('/dataset/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth'))
    ap.add_argument('--device',default='cuda:0'); ap.add_argument('--vae-batch-size',type=int,default=8); ap.add_argument('--shard-size',type=int,default=256)
    ap.add_argument('--source-sha256',default='')
    a=ap.parse_args(); out=a.output_root
    if (out/'manifest.json').exists(): raise SystemExit(f'already finalized: {out}')
    rows={}
    for split in ('train','validation','test'):
        p=out/'splits'/f'{split}.jsonl'
        rows[split]=[json.loads(x) for x in p.read_text().splitlines() if x]
    ordered=rows['train']+rows['validation']+rows['test']
    arts=[]
    for split in ('train','validation','test'):
        for kind,rel,fmt in [('split_index',f'splits/{split}.jsonl','jsonl'),('sample_shard',f'samples/{split}.jsonl','jsonl')]:
            p=out/rel; d=file_artifact(kind,split,p,out); arts.append(d)
    stats=out/'stats/coordinate_stats_train_source.npz'; arts.append(file_artifact('coordinate_stats',None,stats,out))
    arts.extend(build_latents(ordered,out,a.wan_checkpoint,a.device,a.shard_size,a.vae_batch_size))
    report={'protocol':'worldbridge4d.dataset.v1','status':'not_validated','gates':{},'accepted_clips':{k:len(v) for k,v in rows.items()},'rejected_scenes':sum((rejected_scenes(a.data_root,s) for s in ('train','val','test')),[]),'notes':['Latents are posterior means from the native WanVAEEncoder.','Formal gates still require validation and consumer smoke.']}
    rp=out/'audit/validation_report.json'; rp.parent.mkdir(parents=True,exist_ok=True); rp.write_text(json.dumps(report,indent=2)+'\n'); arts.append(file_artifact('validation_report',None,rp,out))
    src=a.source_sha256 or '0'*64
    commit='0'*40
    try:
        import subprocess; commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    except Exception: pass
    import torch
    manifest={'protocol':'worldbridge4d.dataset.v1','dataset':{'id':'PointOdyssey','version':'official-local-release','source_uri':str(a.data_root),'source_sha256':src,'license':None},'clip':{'frames':21,'height':128,'width':128,'channels':3,'rgb_dtype':'uint8','temporal_policy':'ordered_no_padding_no_interpolation','fps':24.0,'default_stride':1},'camera':{'intrinsics':'per_frame_3x3','pose':'camera_to_world_4x4','optical_axis':'-z','image_axes':'u_right_v_down','pixel_center':'integer_uv','world_units':'meters','depth_convention':'z_meters'},'geometry':{'annotation_mode':'dense_xyz','coordinate_frame':'source_camera','validity_semantics':'valid_not_visibility_occluded_valid_supervised','dense_xyz_storage_dtype':'float32','visibility_available':True},'wan_latent':{'model':'Wan2.1-T2V-1.3B-VAE','checkpoint_sha256':sha256(a.wan_checkpoint),'posterior':'mean','normalization':'native_wan_channel_mean_std','dtype':'float32','shape':[16,6,16,16]},'splits':{s:{'clips':len(rows[s]),'parents':len({r['parent_id'] for r in rows[s]}),'index_path':f'splits/{s}.jsonl','index_sha256':sha256(out/f'splits/{s}.jsonl')} for s in rows},'artifacts':arts,'producer':{'git_commit':commit,'argv':sys.argv,'seed':2029,'created_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps({'cache':str(out),'clips':{s:len(v) for s,v in rows.items()},'latent_shards':len(arts)-7},indent=2))
if __name__=='__main__': main()
