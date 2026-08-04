#!/usr/bin/env python3
"""Train Compact or Full geometry-supervised deterministic 4D latent baseline."""
from __future__ import annotations
import argparse, json, os, pathlib, subprocess, sys, time
import numpy as np
import torch
import yaml

ROOT=pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from worldbridge.data import MOViFDataset
from worldbridge.losses import balanced_query_loss
from worldbridge.models import WorldLatentModel
from worldbridge.pipeline import encode_sample, query_tensors, sample_balanced_queries, train_coordinate_stats


def load_config(path):
    cfg=yaml.safe_load(pathlib.Path(path).read_text()); cfg["config_path"]=str(path); return cfg

def make_model(cfg):
    m=cfg["model"]
    return WorldLatentModel(cfg["mode"],latent_channels=m["latent_channels"],latent_time=m["latent_time"],
        latent_height=m["latent_height"],latent_width=m["latent_width"],trajectory_channels=m["trajectory_channels"],
        trajectory_hidden=m["trajectory_hidden"],trajectory_time_dim=m.get("trajectory_time_dim",8),
        reconstruction_channels=m.get("reconstruction_channels",32),context_hidden=m["context_hidden"],decoder_hidden=m["decoder_hidden"])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--output-dir"); ap.add_argument("--steps",type=int); ap.add_argument("--resume")
    args=ap.parse_args(); cfg=load_config(args.config)
    if args.output_dir: cfg["output_dir"]=args.output_dir
    if args.steps is not None: cfg["train"]["steps"]=args.steps
    seed=int(cfg.get("seed",0)); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    device=torch.device("cuda" if torch.cuda.is_available() and cfg["train"].get("device","auto")!="cpu" else "cpu")
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats()
    data=cfg["data"]
    ds=MOViFDataset(data["root"],"train",data["clip_length"],data.get("clip_start",0),data["max_examples"],seed)
    samples=[ds[i] for i in range(len(ds))]
    mean,scale=train_coordinate_stats(samples[:data.get("stats_examples",len(samples))],data["depth_tolerance"],data["depth_relative_tolerance"])
    model=make_model(cfg).to(device); model.set_coordinate_stats(torch.from_numpy(mean),torch.from_numpy(scale))
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(cfg["train"]["learning_rate"]),weight_decay=float(cfg["train"].get("weight_decay",0)))
    start_step=0
    if args.resume:
        ckpt=torch.load(args.resume,map_location=device,weights_only=False); model.load_state_dict(ckpt["model"]); optimizer.load_state_dict(ckpt["optimizer"]); start_step=int(ckpt["step"])
    amp=bool(cfg["train"].get("amp",True) and device.type=="cuda")
    scaler=torch.amp.GradScaler("cuda",enabled=amp); accum=int(cfg["train"].get("gradient_accumulation",1))
    num_queries=int(cfg["train"]["queries_per_step"]); block=int(cfg["train"].get("trajectory_block_size",16384))
    fixed_queries=bool(cfg["train"].get("fixed_queries",False)); steps=int(cfg["train"]["steps"])
    first_loss=None; last_loss=None; optimizer.zero_grad(set_to_none=True); total_queries=0; started=time.perf_counter()
    model.train()
    for step in range(start_step,steps):
        sample=samples[step%len(samples)]
        qseed=seed+1000+(step%len(samples) if fixed_queries else step*7919)
        with torch.amp.autocast("cuda",enabled=amp):
            z,geom=encode_sample(model,sample,device,block,data["depth_tolerance"],data["depth_relative_tolerance"])
            query=sample_balanced_queries(geom,cfg["mode"],num_queries,np.random.default_rng(qseed))
            source,target,uv01,target_x,valid,groups=query_tensors(query,device)
            pred_norm=model.decoder(z,source,uv01,target,sample.num_frames)
            pred=model.denormalize_coordinates(pred_norm)[0]
            loss,parts=balanced_query_loss(pred,target_x,valid,groups)
            scaled_loss=loss/accum
        scaler.scale(scaled_loss).backward()
        if (step+1)%accum==0 or step+1==steps:
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["train"].get("gradient_clip",1.0))
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        value=float(loss.detach()); first_loss=value if first_loss is None else first_loss; last_loss=value; total_queries+=num_queries
        if step==start_step or (step+1)%int(cfg["train"].get("log_every",5))==0 or step+1==steps:
            print(json.dumps({"step":step+1,"loss":value,"groups":parts,"latent_shape":list(z.shape),"valid_queries":int(valid.sum()),"device":str(device)}),flush=True)
    elapsed=time.perf_counter()-started; ratio=float(last_loss/max(first_loss,1e-12))
    out=pathlib.Path(cfg["output_dir"]); out.mkdir(parents=True,exist_ok=True)
    git_commit=os.getenv("PRL_GIT_COMMIT") or subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    summary={"mode":cfg["mode"],"seed":seed,"steps":steps,"first_loss":first_loss,"final_loss":last_loss,"loss_ratio":ratio,
             "elapsed_seconds":elapsed,"train_queries_per_second":total_queries/elapsed,"peak_gpu_memory_mb":(torch.cuda.max_memory_allocated()/2**20 if device.type=="cuda" else 0),
             "parameters":sum(p.numel() for p in model.parameters()),"coordinate_mean":mean.tolist(),"coordinate_scale":scale.tolist(),
             "device":str(device),"torch":torch.__version__,"git_commit":git_commit,"prl_run_id":os.getenv("PRL_RUN_ID"),"command":sys.argv,"config":cfg}
    checkpoint={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"step":steps,"config":cfg,"summary":summary}
    ckpt_path=out/"checkpoint.pt"; torch.save(checkpoint,ckpt_path)
    # Verify save/restore immediately with a separately constructed model.
    restored=make_model(cfg).to(device); restored.load_state_dict(torch.load(ckpt_path,map_location=device,weights_only=False)["model"])
    summary["checkpoint_restore_verified"]=True
    (out/"summary.json").write_text(json.dumps(summary,indent=2)); pathlib.Path("artifacts").mkdir(exist_ok=True); pathlib.Path("artifacts/checkpoint.complete").write_text(str(ckpt_path)+"\n")
    print(json.dumps(summary,sort_keys=True),flush=True); print(f"CHECKPOINT_CREATED: {ckpt_path}",flush=True)
    required=cfg["train"].get("require_loss_ratio")
    if required is not None and ratio>float(required): raise RuntimeError(f"overfit loss ratio {ratio:.4f} > required {required}")

if __name__=="__main__": main()
