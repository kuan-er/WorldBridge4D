#!/usr/bin/env python3
"""Resolve radial-vs-z depth using independent native forward flow on static background."""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np
import tensorflow as tf
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/"src"))
from worldbridge.geometry import CameraModel

def arr(ex,key,kind="float_list",dtype=np.float64): return np.asarray(getattr(ex.features.feature[key],kind).value,dtype=dtype)
def png(ex,key,t,dtype,channels=1): return tf.io.decode_png(ex.features.feature[key].bytes_list.value[t],channels=channels,dtype=dtype).numpy()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--data-root",required=True); args=ap.parse_args()
 version=next(p for p in sorted((pathlib.Path(args.data_root)/"128x128").glob("*")) if p.is_dir()); path=sorted(version.glob("movi_f-train.tfrecord-*"))[0]
 ex=tf.train.Example.FromString(bytes(next(iter(tf.data.TFRecordDataset([str(path)]))).numpy()))
 H=int(arr(ex,"metadata/height","int64_list")[0]); W=int(arr(ex,"metadata/width","int64_list")[0]); T=int(arr(ex,"metadata/num_frames","int64_list")[0])
 dr=arr(ex,"metadata/depth_range"); fr=arr(ex,"metadata/forward_flow_range")
 depths=[]; segs=[]
 for t in (0,1):
  raw=png(ex,"depth",t,tf.uint16)[...,0].astype(np.float64); depths.append(dr[0]+raw/65535*(dr[1]-dr[0])); segs.append(png(ex,"segmentations",t,tf.uint8)[...,0])
 flow_raw=arr(ex,"forward_flow","int64_list").reshape(T,H,W,2)[0]
 flow=fr[0]+flow_raw/65535*(fr[1]-fr[0])
 cp=arr(ex,"camera/positions").reshape(T,3)[:2]; cq=arr(ex,"camera/quaternions").reshape(T,4)[:2]
 f=float(arr(ex,"camera/focal_length")[0]); sensor=float(arr(ex,"camera/sensor_width")[0]); cam=CameraModel(H,W,f,sensor)
 vv,uu=np.meshgrid(np.arange(H),np.arange(W),indexing="ij"); uv=np.stack([uu.ravel(),vv.ravel()],-1); bg=(segs[0].ravel()==0); uv=uv[bg][::8]; fl=flow.reshape(-1,2)[bg][::8]; d0=depths[0].ravel()[bg][::8]
 rows=[]
 for swap in (False,True):
  fxy=fl[:,::-1] if swap else fl
  for signx in (-1,1):
   for signy in (-1,1):
    target=uv+fxy*np.array([signx,signy]); q=np.floor(target+.5).astype(int); inside=(q[:,0]>=0)&(q[:,0]<W)&(q[:,1]>=0)&(q[:,1]<H)
    same=np.zeros(len(q),bool); same[inside]=segs[1][q[inside,1],q[inside,0]]==0; keep=inside&same
    if keep.sum()<50: continue
    for radial in (False,True):
     c=CameraModel(H,W,f,sensor,depth_is_euclidean=radial)
     w0=c.backproject_pixels(d0[keep],uv[keep],cp[0],cq[0]); d1=depths[1][q[keep,1],q[keep,0]]; w1=c.backproject_pixels(d1,q[keep],cp[1],cq[1]); e=np.linalg.norm(w0-w1,axis=-1)
     rows.append({"flow_swap":swap,"flow_sign":[signx,signy],"depth":"radial" if radial else "z","n":int(keep.sum()),"median_world_error":float(np.median(e)),"mean":float(np.mean(e)),"p95":float(np.percentile(e,95))})
 print(json.dumps(sorted(rows,key=lambda x:x["median_world_error"])[:16],indent=2)); print("DEPTH_AUDIT_OK")
if __name__=="__main__":main()
