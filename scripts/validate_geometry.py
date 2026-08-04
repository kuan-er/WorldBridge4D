#!/usr/bin/env python3
"""Run bounded MOVi-F geometry checks and save one diagnostic figure."""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from worldbridge.data import MOViFDataset
from worldbridge.geometry import GeometryBuilder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output-dir", default="/tmp/worldbridge_geometry")
    ap.add_argument("--source", type=int, default=0)
    ap.add_argument("--pixel-block", type=int, default=4096)
    args = ap.parse_args()
    ds = MOViFDataset(args.data_root, split="train", clip_length=21, clip_start=0, max_examples=1)
    sample = ds[0]
    geom = GeometryBuilder(sample)
    P, V = geom.pointmaps()
    T,H,W,_ = P.shape
    # Projection round-trip on every pointmap pixel, in bounded frame chunks.
    vv, uu = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    uv = np.stack([uu.reshape(-1), vv.reshape(-1)], axis=-1)
    roundtrip = []
    radial = []
    for t in range(T):
        projected, depth_z, radial_depth = geom.camera.project(
            geom.camera.backproject(sample.depth[t], sample.camera_positions[t], sample.camera_quaternions[t]),
            sample.camera_positions[t], sample.camera_quaternions[t])
        roundtrip.append(np.linalg.norm(projected.reshape(-1,2) - uv, axis=-1))
        radial.append(np.abs(radial_depth.reshape(-1) - sample.depth[t].reshape(-1)))
    roundtrip = np.concatenate(roundtrip); radial = np.concatenate(radial)

    # X[s,s,p] = P[s,p] and track source-frame pixels without materializing O(T^2HW).
    diagonal_errors = []
    visibility_counts = {"visible": 0, "occluded": 0, "valid": 0}
    source = int(args.source)
    track_x = track_m = track_v = track_uv = None
    for s in range(T):
        for start in range(0, H*W, args.pixel_block):
            x, m, v, block_uv = geom.trajectory_block(s, start, min(start + args.pixel_block, H*W))
            diagonal_errors.append(np.linalg.norm(x[:, s] - P[s, block_uv[:,1], block_uv[:,0]], axis=-1))
            visibility_counts["valid"] += int(v.sum())
            visibility_counts["visible"] += int((m & v).sum())
            visibility_counts["occluded"] += int((~m & v).sum())
            if s == source and start == 0:
                track_x, track_m, track_v, track_uv = x[:16], m[:16], v[:16], block_uv[:16]
    diagonal_errors = np.concatenate(diagonal_errors)
    # Explicitly check projection/instance/depth agreement for visible source-0 labels.
    visible_projected = 0; visible_bad_instance = 0; visible_bad_depth = 0
    if track_x is not None:
        for t in range(T):
            uv_t, _, radial_t = geom.project_trajectory(track_x[:,t], t)
            u = np.floor(uv_t[:,0] + .5).astype(int); v = np.floor(uv_t[:,1] + .5).astype(int)
            inside = (u>=0)&(u<W)&(v>=0)&(v<H)&(track_v[:,t])
            if not inside.any(): continue
            # source instance id is obtained once from source pixels.
            ids = sample.segmentation[source, track_uv[:,1], track_uv[:,0]]
            ii = np.flatnonzero(inside)
            seg_ok = sample.segmentation[t, v[ii], u[ii]] == ids[ii]
            observed = sample.depth[t, v[ii], u[ii]]
            tol = geom.depth_tolerance + geom.depth_relative_tolerance*np.maximum(observed, 1.)
            depth_ok = np.abs(observed-radial_t[ii]) <= tol
            visible_projected += int(track_m[ii,t].sum())
            visible_bad_instance += int((track_m[ii,t] & ~seg_ok).sum())
            visible_bad_depth += int((track_m[ii,t] & ~depth_ok).sum())

    result = {
        "video_name": sample.video_name, "clip_start": sample.clip_start,
        "shape": [T,H,W], "camera_quaternion_order": geom.camera.quaternion_order,
        "camera_forward_axis": geom.camera.forward_axis, "depth_is_euclidean": geom.camera.depth_is_euclidean,
        "depth_tolerance": geom.depth_tolerance, "depth_relative_tolerance": geom.depth_relative_tolerance,
        "pixel_to_3d_to_pixel_max_px": float(roundtrip.max()),
        "pixel_to_3d_to_pixel_p95_px": float(np.percentile(roundtrip,95)),
        "pixel_depth_roundtrip_max": float(radial.max()),
        "diagonal_X_equals_P_max": float(diagonal_errors.max()),
        "diagonal_X_equals_P_p95": float(np.percentile(diagonal_errors,95)),
        "trajectory_valid_visible_occluded": visibility_counts,
        "visible_projection_count": visible_projected,
        "visible_projection_bad_instance": visible_bad_instance,
        "visible_projection_bad_depth": visible_bad_depth,
    }
    if result["pixel_to_3d_to_pixel_max_px"] > 1e-4 or result["diagonal_X_equals_P_max"] > 1e-4:
        raise AssertionError(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    out = pathlib.Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(P[0][...,2], cmap="viridis"); axes[0].set_title("P[0] anchor-camera z")
        rgb = sample.rgb[0].astype(np.float32)/255.
        axes[1].imshow(rgb)
        if track_x is not None:
            for n in range(min(16, len(track_x))):
                uv_track = []
                for t in range(T):
                    uv_t, _, _ = geom.project_trajectory(track_x[n,t], t); uv_track.append(uv_t)
                uv_track = np.asarray(uv_track)
                axes[1].plot(uv_track[:,0], uv_track[:,1], "-", linewidth=.7)
                axes[1].plot(uv_track[:,0], uv_track[:,1], ".", markersize=2)
        axes[1].set_title("source-pixel projected tracks")
        axes[2].imshow(sample.segmentation[0], cmap="tab20"); axes[2].set_title("segmentation / visibility source")
        for ax in axes: ax.set_xlim(0,W); ax.set_ylim(H,0)
        fig.tight_layout(); fig.savefig(out/"geometry_diagnostic.png", dpi=140); plt.close(fig)
    except Exception as exc:
        (out/"figure_error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
    (out/"geometry_diagnostic.json").write_text(json.dumps(result, indent=2))
    print(f"DIAGNOSTIC_DIR: {out}")
    print("GEOMETRY_VALIDATION_OK")


if __name__ == "__main__":
    main()
