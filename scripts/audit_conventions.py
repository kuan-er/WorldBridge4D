#!/usr/bin/env python3
"""Resolve quaternion direction/order against independent native MOVi metadata."""
from __future__ import annotations
import argparse, json, pathlib
import numpy as np
import tensorflow as tf


def arr(ex,key,kind="float_list",dtype=np.float64):
    return np.asarray(getattr(ex.features.feature[key],kind).value,dtype=dtype)

def Rq(q, order):
    if order=="wxyz": w,x,y,z=np.moveaxis(q,-1,0)
    else: x,y,z,w=np.moveaxis(q,-1,0)
    n=np.sqrt(w*w+x*x+y*y+z*z); w,x,y,z=w/n,x/n,y/n,z/n
    R=np.empty(q.shape[:-1]+(3,3))
    R[...,0,0]=1-2*(y*y+z*z); R[...,0,1]=2*(x*y-z*w); R[...,0,2]=2*(x*z+y*w)
    R[...,1,0]=2*(x*y+z*w); R[...,1,1]=1-2*(x*x+z*z); R[...,1,2]=2*(y*z-x*w)
    R[...,2,0]=2*(x*z-y*w); R[...,2,1]=2*(y*z+x*w); R[...,2,2]=1-2*(x*x+y*y)
    return R

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-root",required=True); args=ap.parse_args()
    version=next(p for p in sorted((pathlib.Path(args.data_root)/"128x128").glob("*")) if p.is_dir())
    path=sorted(version.glob("movi_f-train.tfrecord-*"))[0]
    raw=next(iter(tf.data.TFRecordDataset([str(path)]))).numpy(); ex=tf.train.Example.FromString(bytes(raw))
    T=int(arr(ex,"metadata/num_frames","int64_list")[0]); N=int(arr(ex,"metadata/num_instances","int64_list")[0])
    H=int(arr(ex,"metadata/height","int64_list")[0]); W=int(arr(ex,"metadata/width","int64_list")[0])
    f=float(arr(ex,"camera/focal_length")[0]); sensor=float(arr(ex,"camera/sensor_width")[0]); fx=f/sensor*W
    cp=arr(ex,"camera/positions").reshape(T,3); cq=arr(ex,"camera/quaternions").reshape(T,4)
    ip=arr(ex,"instances/positions").reshape(N,T,3); iq=arr(ex,"instances/quaternions").reshape(N,T,4)
    image=arr(ex,"instances/image_positions").reshape(N,T,2)
    bbox=arr(ex,"instances/bboxes_3d").reshape(N,T,8,3)
    camera=[]
    for order in ("wxyz","xyzw"):
      S=Rq(cq,order)
      for stored in ("camera_to_world","world_to_camera"):
        A=S if stored=="camera_to_world" else np.swapaxes(S,-1,-2)
        cam=np.einsum("ntj,tji->nti",ip-cp[None],A)
        for forward in (-1,1):
          z=forward*cam[...,2]; good=np.abs(z)>1e-8
          u=fx*cam[...,0]/np.where(good,z,1)+(W-1)/2
          # Local camera y points up for both forward signs.
          v=(H-1)/2-fx*cam[...,1]/np.where(good,z,1)
          pred=np.stack([u/W,v/H],-1)
          for perm in (False,True):
            target=image[...,::-1] if perm else image
            mae=float(np.mean(np.abs(pred[good]-target[good])))
            camera.append({"order":order,"stored":stored,"forward":f"{forward:+d}z","native_xy_swapped":perm,"normalized_mae":mae})
    object_rows=[]
    for order in ("wxyz","xyzw"):
      S=Rq(iq,order)
      for stored in ("local_to_world","world_to_local"):
        A=S if stored=="local_to_world" else np.swapaxes(S,-1,-2)
        local=np.einsum("ntkj,ntji->ntki",bbox-ip[:,:,None],A)
        # Rigid local bbox corners should be temporally constant. Corner ordering is native and stable.
        mean_local=local.mean(axis=1,keepdims=True)
        error=np.linalg.norm(local-mean_local,axis=-1)
        object_rows.append({"order":order,"stored":stored,"local_bbox_temporal_mean_error":float(error.mean()),"p95":float(np.percentile(error,95))})
    print("CAMERA_CANDIDATES")
    print(json.dumps(sorted(camera,key=lambda x:x["normalized_mae"])[:10],indent=2))
    print("OBJECT_CANDIDATES")
    print(json.dumps(sorted(object_rows,key=lambda x:x["local_bbox_temporal_mean_error"]),indent=2))
    print("CONVENTION_AUDIT_OK")
if __name__=="__main__": main()
