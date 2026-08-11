#!/usr/bin/env python3
"""Protocol v1 real-Wan two-step smoke for the PointOdyssey adapter."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from worldbridge.pointodyssey import PointOdysseyDataset
from worldbridge.dense4d import masked_pair_smooth_l1
from worldbridge.dense4d_runtime import build_real_model, parameter_groups

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache-root',type=Path,required=True); ap.add_argument('--wan-root',type=Path,required=True); ap.add_argument('--condition',type=Path,required=True); ap.add_argument('--device',default='cuda'); ap.add_argument('--index',type=int,default=6521); ap.add_argument('--source',type=int,default=0); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--wandb-dir',type=Path,required=True); a=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
    torch.cuda.set_device(torch.device(a.device)); torch.manual_seed(2029); np.random.seed(2029)
    ds=PointOdysseyDataset(a.cache_root,'train'); i=int(a.index); source=int(a.source)
    with np.load(a.cache_root/'stats/coordinate_stats_train_source.npz') as z: mean=z['mean'].astype(np.float32); scale=z['scale'].astype(np.float32)
    xyz,valid,_=ds.source_all_targets_with_visibility(i,source)
    target=((xyz-mean[None,:,None,None])/np.maximum(scale[None,:,None,None],1e-6)).astype(np.float32)
    clean=torch.from_numpy(ds.clean_latent(i))[None]
    source_t=torch.full((1,21),source,dtype=torch.long); target_t=torch.arange(21,dtype=torch.long)[None]
    config={'wan_root':str(a.wan_root),'empty_text_condition':str(a.condition),'clip_length':21,'image_size':128,'precision':'bf16','backbone_readout':'wan_hidden_structured','wan_hidden_layers':[13,14,15],'geometry_dim':64,'geometry_spatial_size':16,'geometry_num_heads':4,'geometry_clean_skip':True,'motion_slots':4,'structured_local_queries':True,'query_dim':128,'embedding_dim':64,'num_cross_attn_layers':1,'num_heads':4,'upsample_channels':[128,64,32,16],'trainable_mode':'full','gradient_checkpointing':True,'seed':2029,'decoder_seed':424242,'learning_rate':1e-4,'backbone_learning_rate':1e-5,'geometry_learning_rate':1e-4,'weight_decay':0.0}
    model=build_real_model(config,a.device); groups=parameter_groups(model,config); opt=torch.optim.AdamW(groups,weight_decay=0.0)
    import wandb
    a.wandb_dir.mkdir(parents=True,exist_ok=True)
    run=wandb.init(project='worldbridge4d',name='pointodyssey-protocol-v1-real-wan-smoke',mode='offline',dir=str(a.wandb_dir),config=config)
    clean=clean.to(a.device,dtype=torch.bfloat16); source_t=source_t.to(a.device); target_t=target_t.to(a.device); target=torch.from_numpy(target)[None].to(a.device,dtype=torch.bfloat16); valid_t=torch.from_numpy(valid)[None].to(a.device)
    losses=[]; finite_grad=[]
    for step in range(1,3):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16): pred,_,_=model(clean,source_t,target_t)
        loss=masked_pair_smooth_l1(pred.float(),target.float(),valid_t,beta=0.05)
        if not torch.isfinite(loss): raise RuntimeError(f'non-finite loss at step {step}')
        loss.backward()
        grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        ok=bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads)
        if not ok: raise RuntimeError(f'non-finite/missing gradients at step {step}')
        opt.step(); losses.append(float(loss.detach())); finite_grad.append(ok)
        payload={'global_step':step,'smoke/loss':losses[-1]}; run.log(payload,step=step); print(json.dumps(payload),flush=True)
    wan_gradient=any(p.grad is not None for p in model.backbone.mapping.parameters() if p.requires_grad)
    adapter_gradient=any(p.grad is not None for p in model.backbone.adapter_parameters if p.requires_grad)
    decoder_gradient=any(p.grad is not None for p in model.decoder.parameters() if p.requires_grad)
    run.finish()
    result={'status':'pass','steps':2,'global_steps':[1,2],'wandb_mode':'offline','wandb_global_steps':[1,2],'losses':losses,'finite_gradients':finite_grad,'clean_latent_shape':list(clean.shape[1:]),'prediction_shape':list(pred.shape),'valid_points':int(valid_t.sum()),'wan_gradient':wan_gradient,'adapter_gradient':adapter_gradient,'decoder_gradient':decoder_gradient,'peak_cuda_memory_gib':torch.cuda.max_memory_allocated()/2**30}
    if not (wan_gradient and adapter_gradient and decoder_gradient): raise RuntimeError('Wan/adapter/decoder gradient gate failed')
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
