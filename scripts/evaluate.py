#!/usr/bin/env python3
"""Chunked evaluation of reconstruction and tracking query families."""
from __future__ import annotations
import argparse,json,os,pathlib,sys,time
import numpy as np
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from worldbridge.data import MOViFDataset
from worldbridge.metrics import update_metrics,finalize_metrics
from worldbridge.models import WorldLatentModel
from worldbridge.pipeline import encode_sample

def make_model(cfg):
 m=cfg["model"]
 return WorldLatentModel(cfg["mode"],latent_channels=m["latent_channels"],latent_time=m["latent_time"],latent_height=m["latent_height"],latent_width=m["latent_width"],trajectory_channels=m["trajectory_channels"],trajectory_hidden=m["trajectory_hidden"],trajectory_time_dim=m.get("trajectory_time_dim",8),reconstruction_channels=m.get("reconstruction_channels",32),context_hidden=m["context_hidden"],decoder_hidden=m["decoder_hidden"])

def decode_update(model,z,geom,store,group,source,target,uv,x,visible,valid,device,chunk):
 n=len(source); H,W=geom.sample.height,geom.sample.width
 for start in range(0,n,chunk):
  sl=slice(start,min(start+chunk,n)); uv01=uv[sl].astype(np.float32); uv01[:,0]/=max(W-1,1); uv01[:,1]/=max(H-1,1)
  st=torch.from_numpy(source[sl].astype(np.float32))[None].to(device); tt=torch.from_numpy(target[sl].astype(np.float32))[None].to(device); ut=torch.from_numpy(uv01)[None].to(device)
  pred=model.denormalize_coordinates(model.decoder(z,st,ut,tt,geom.sample.num_frames))[0].float().cpu().numpy()
  reproj=np.empty(len(pred),np.float32)
  for target_frame in np.unique(target[sl]):
   ii=np.flatnonzero(target[sl]==target_frame); puv,_,_=geom.project_trajectory(pred[ii],int(target_frame)); tuv,_,_=geom.project_trajectory(x[sl][ii],int(target_frame)); reproj[ii]=np.linalg.norm(puv-tuv,axis=-1)
  update_metrics(store,pred,x[sl],valid[sl],visible[sl],reproj,group)
 return n

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--checkpoint",required=True); ap.add_argument("--split",default="validation"); ap.add_argument("--data-root"); ap.add_argument("--max-examples",type=int,default=None); ap.add_argument("--pixel-stride",type=int,default=16); ap.add_argument("--query-chunk",type=int,default=8192); ap.add_argument("--output")
 args=ap.parse_args(); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); ckpt=torch.load(args.checkpoint,map_location=device,weights_only=False); cfg=ckpt["config"]; data=cfg["data"]
 root=args.data_root or data["root"]
 ds=MOViFDataset(root,args.split,data["clip_length"],data.get("clip_start",0),args.max_examples,cfg.get("seed",0))
 tracking=data.get("tracking", cfg.get("tracking", {}))
 run=None
 if cfg.get("tracking", {}).get("enabled", False):
  import wandb
  run=wandb.init(project=os.getenv("WANDB_PROJECT", tracking.get("project", "worldbridge4d")), entity=os.getenv("WANDB_ENTITY", tracking.get("entity")), group=os.getenv("WANDB_GROUP", tracking.get("group", "worldbridge4d-full-vs-compact")), job_type="evaluate", name=os.getenv("WANDB_NAME", f"{cfg['mode']}-{args.split}-eval-{os.getenv('PRL_RUN_ID', 'local')}"), tags=list(tracking.get("tags", []))+[cfg["mode"], args.split, "full-movi-f"], config={"checkpoint": args.checkpoint, "split": args.split, "max_examples": args.max_examples, "pixel_stride": args.pixel_stride, "query_chunk": args.query_chunk})
  print(f"WANDB_RUN_URL: {run.url}", flush=True)
 model=make_model(cfg).to(device); model.load_state_dict(ckpt["model"]); model.eval(); store={}; total_queries=0
 if device.type=="cuda": torch.cuda.reset_peak_memory_stats()
 started=time.perf_counter()
 with torch.no_grad(),torch.amp.autocast("cuda",enabled=device.type=="cuda"):
  for sample in ds:
   z,geom=encode_sample(model,sample,device,cfg["train"].get("trajectory_block_size",16384),data["depth_tolerance"],data["depth_relative_tolerance"])
   T,H,W=sample.num_frames,sample.height,sample.width; flat=np.arange(0,H*W,args.pixel_stride,dtype=np.int64); uv=np.stack([flat%W,flat//W],-1); n=len(uv)
   # All reconstruction source frames, t=s.
   for s in range(T):
    x,m,v=geom.trajectory(s,uv); source=np.full(n,s,np.int64); target=source.copy(); row=np.arange(n)
    total_queries+=decode_update(model,z,geom,store,"reconstruction",source,target,uv,x[row,target],m[row,target],v[row,target],device,args.query_chunk)
   # First-frame anchor, every target frame.
   x,m,v=geom.trajectory(0,uv); source=np.zeros(n*T,np.int64); target=np.tile(np.arange(T),n); uv_all=np.repeat(uv,T,axis=0)
   total_queries+=decode_update(model,z,geom,store,"first_frame_tracking",source,target,uv_all,x.reshape(-1,3),m.reshape(-1),v.reshape(-1),device,args.query_chunk)
   if cfg["mode"]=="full":
    # Every legal (s,t), evaluated on a deterministic spatial grid.
    for s in range(T):
     x,m,v=geom.trajectory(s,uv); source=np.full(n*T,s,np.int64); target=np.tile(np.arange(T),n); uv_all=np.repeat(uv,T,axis=0)
     total_queries+=decode_update(model,z,geom,store,"arbitrary_all_st",source,target,uv_all,x.reshape(-1,3),m.reshape(-1),v.reshape(-1),device,args.query_chunk)
     ids=sample.segmentation[s,uv[:,1],uv[:,0]]; first=np.full(n,T,np.int64)
     for k in np.unique(ids):
      if k>0 and k<=sample.num_instances:
       seen=np.flatnonzero(sample.instance_visibility[k-1]>0); first[ids==k]=seen[0] if len(seen) else T
     late=(first>0)&(first<T)&(s>=first)
     if late.any():
      lx=x[late]; lm=m[late]; lv=v[late]; luv=uv[late]; nn=len(luv)
      source_l=np.full(nn*T,s,np.int64); target_l=np.tile(np.arange(T),nn)
      total_queries+=decode_update(model,z,geom,store,"late_appearing_objects",source_l,target_l,np.repeat(luv,T,0),lx.reshape(-1,3),lm.reshape(-1),lv.reshape(-1),device,args.query_chunk)
    # Reproducible random arbitrary-source sample in addition to all s,t.
    rng=np.random.default_rng(cfg.get("seed",0)+31); nq=min(4096,H*W); source=rng.integers(T,size=nq); target=rng.integers(T,size=nq); rflat=rng.integers(H*W,size=nq); ruv=np.stack([rflat%W,rflat//W],-1); x,m,v=geom.query(source,ruv); row=np.arange(nq)
    total_queries+=decode_update(model,z,geom,store,"arbitrary_random",source,target,ruv,x[row,target],m[row,target],v[row,target],device,args.query_chunk)
 elapsed=time.perf_counter()-started
 result={"mode":cfg["mode"],"split":args.split,"examples":len(ds),"pixel_stride":args.pixel_stride,"all_st":cfg["mode"]=="full","metrics":finalize_metrics(store),"parameters":sum(p.numel() for p in model.parameters()),"peak_gpu_memory_mb":torch.cuda.max_memory_allocated()/2**20 if device.type=="cuda" else 0,"inference_queries_per_second":total_queries/elapsed,"elapsed_seconds":elapsed,"query_count":total_queries,"device":str(device),"compact_capability_note":("Compact is evaluated only for reconstruction and source-frame-0 tracking; arbitrary-source tracking is not a native claim." if cfg["mode"]=="compact" else None)}
 if run is not None:
  run.summary.update({f"eval/{k}": v for k,v in result["metrics"].items()})
  result["wandb_url"]=run.url
  run.finish()
 print(json.dumps(result,indent=2)); output=pathlib.Path(args.output or pathlib.Path(args.checkpoint).with_name(f"evaluation_{args.split}.json")); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,indent=2)); print(f"EVALUATION_OK: {output}")
if __name__=="__main__":main()
