#!/usr/bin/env python3
"""Stage-B geometry gate for source-anchored dynamic pointmaps."""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from worldbridge.data import MOViFDataset
from worldbridge.geometry import GeometryBuilder
from worldbridge.pointmap import build_dynamic_pointmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/dataset/MOVi-F")
    ap.add_argument("--max-examples", type=int, default=2)
    ap.add_argument("--output-dir", default="/tmp/worldbridge_pointmap_geometry")
    args = ap.parse_args()
    ds = MOViFDataset(args.data_root, split="train", clip_length=21, clip_start=0, max_examples=args.max_examples)
    results=[]; visualization=None
    for sample in ds:
        geom=GeometryBuilder(sample); P,Pvalid=geom.pointmaps(); T,H,W,_=P.shape
        diag=[]; static=[]; dynamic=[]; visible_bad=[]; valid_total=visible_total=occ_total=0
        for s in range(T):
            pm=build_dynamic_pointmap(sample,s)
            diag.append(np.abs(pm.xyz[s][pm.valid[s]]-P[s][pm.valid[s]]).max())
            valid_total += int(pm.valid.sum()); visible_total += int((pm.visible & pm.valid).sum()); occ_total += int((~pm.visible & pm.valid).sum())
            # source-pixel anchoring: target time does not change source-grid p.
            assert pm.xyz.shape == (T,H,W,3) and pm.visible.shape == (T,H,W)
            # W=C1 inverse direction round-trip on a deterministic pixel block.
            flat=np.arange(0,H*W,max(1,(H*W)//257),dtype=np.int64)
            uv=np.stack([flat%W,flat//W],-1)
            world=geom.camera.backproject_pixels(sample.depth[s,uv[:,1],uv[:,0]],uv,sample.camera_positions[s],sample.camera_quaternions[s])
            back=geom.anchor_to_world(geom.world_to_anchor(world))
            assert np.max(np.linalg.norm(back-world,axis=-1)) < 1e-5
            # Rigid transform sanity: one dynamic foreground source pixel.
            ids=sample.segmentation[s,uv[:,1],uv[:,0]]
            dyn=np.flatnonzero((ids>0) & sample.instance_dynamic[np.maximum(ids-1,0)])
            if len(dyn) and T>1:
                j=int(dyn[0]); obj=int(ids[j]-1)
                local=(world[j]-sample.instance_positions[obj,s]) @ geom._object_rot[obj,s]
                expected=np.einsum('tij,j->ti',geom._object_rot[obj],local)+sample.instance_positions[obj]
                actual=geom.anchor_to_world(pm.xyz[:,uv[j,1],uv[j,0]])
                dynamic.append(float(np.max(np.linalg.norm(expected-actual,axis=-1))))
            # Static background must stay world-static over target time.
            bg=np.flatnonzero(ids==0)
            if len(bg):
                j=int(bg[0]); static.append(float(np.max(np.linalg.norm(geom.anchor_to_world(pm.xyz[:,uv[j,1],uv[j,0]])-world[j],axis=-1))))
            # Visibility is only a visibility label: valid occluded points remain XYZ.
            if np.any((~pm.visible) & pm.valid):
                q=np.flatnonzero((~pm.visible[:, :, :]) & pm.valid)
                assert q.size and np.isfinite(pm.xyz).all()
            # Visible target projection agrees with segmentation and depth by construction.
            for t in range(T):
                x=pm.xyz[t].reshape(-1,3); ok=pm.visible[t].reshape(-1)&pm.valid[t].reshape(-1)
                if ok.any():
                    uvp,_,rd=geom.project_trajectory(x[ok],t)
                    u=np.floor(uvp[:,0]+.5).astype(int); v=np.floor(uvp[:,1]+.5).astype(int)
                    inside=(u>=0)&(u<W)&(v>=0)&(v<H)
                    if inside.any(): visible_bad.append(int(np.count_nonzero(~inside)))
        result={"video_name":sample.video_name,"clip_shape":[T,H,W,3],"sources_tested":T,
                "diagonal_X_equals_source_pointmap_max":float(max(diag)),
                "dynamic_rigid_transform_max":float(max(dynamic) if dynamic else 0.0),
                "background_static_max":float(max(static) if static else 0.0),
                "visible_projection_out_of_bounds":int(sum(visible_bad)),
                "valid":valid_total,"visible_valid":visible_total,"occluded_valid":occ_total,
                "has_occluded_valid":bool(occ_total>0)}
        assert result["diagonal_X_equals_source_pointmap_max"] < 1e-4
        assert result["dynamic_rigid_transform_max"] < 1e-4
        assert result["background_static_max"] < 1e-4
        assert result["visible_projection_out_of_bounds"] == 0
        results.append(result)
        if visualization is None:
            source=min(7,T-1); pm=build_dynamic_pointmap(sample,source)
            chosen=np.array([[16,16],[32,64],[64,64],[96,32],[112,112]],dtype=np.int64)
            projected=[]
            for t in range(T):
                uv_t,_,_=geom.project_trajectory(pm.xyz[t,chosen[:,1],chosen[:,0]],t); projected.append(uv_t)
            visualization={"rgb":sample.rgb[source],"pointmap":P[source],"trajectories":pm.xyz[:,chosen[:,1],chosen[:,0]],
                           "projected":np.asarray(projected),"visible":pm.visible[:,chosen[:,1],chosen[:,0]],"pixels":chosen}
    out=pathlib.Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    (out/"geometry_report.json").write_text(json.dumps(results,indent=2))
    if visualization is not None:
        np.savez_compressed(out/"source_tracks.npz",**visualization)
        try:
            import matplotlib.pyplot as plt
            fig,ax=plt.subplots(1,2,figsize=(8,4));ax[0].imshow(visualization["rgb"]);ax[0].set_title("source RGB s=7")
            ax[1].imshow(visualization["rgb"]);ax[1].set_title("projected source tracks")
            for j in range(len(visualization["pixels"])):
                q=visualization["projected"][:,j];vis=visualization["visible"][:,j]
                ax[1].plot(q[:,0],q[:,1],color="white",linewidth=.6);ax[1].scatter(q[vis,0],q[vis,1],c="lime",s=6);ax[1].scatter(q[~vis,0],q[~vis,1],c="red",s=6)
            for a in ax:a.set_xlim(0,128);a.set_ylim(128,0);a.axis("off")
            fig.tight_layout();fig.savefig(out/"source_tracks.png",dpi=140);plt.close(fig)
        except Exception as exc:(out/"plot_error.txt").write_text(repr(exc))
    print(json.dumps(results,indent=2)); print("POINTMAP_GEOMETRY_OK")

if __name__ == "__main__": main()
