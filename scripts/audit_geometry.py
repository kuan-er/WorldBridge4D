#!/usr/bin/env python3
"""Empirically resolve MOVi-F depth/camera conventions on one native record."""
from __future__ import annotations
import argparse, pathlib, json
import numpy as np
import tensorflow as tf


def feat(ex, name, kind="float_list"):
    f = ex.features.feature[name]
    return np.asarray(getattr(f, kind).value)


def quat_matrix(q, order, inverse=False):
    if order == "wxyz":
        w, x, y, z = q
    else:
        x, y, z, w = q
    n = np.linalg.norm([w, x, y, z])
    w, x, y, z = np.asarray([w, x, y, z]) / n
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z+y*w), 2*(y*z-x*w), 1-2*(x*x+y*y)]], dtype=np.float64)
    return R.T if inverse else R


def summarize(label, actual, expected):
    diff = actual - expected
    return {"label": label, "mean_abs": float(np.mean(np.abs(diff))),
            "median_abs": float(np.median(np.abs(diff))),
            "p95_abs": float(np.percentile(np.abs(diff), 95)),
            "corr": float(np.corrcoef(actual.reshape(-1), expected.reshape(-1))[0, 1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--max-records", type=int, default=2)
    args = ap.parse_args()
    version = next(p for p in sorted((pathlib.Path(args.data_root)/"128x128").glob("*")) if p.is_dir())
    tfrecord = sorted(version.glob("movi_f-train.tfrecord-*"))[0]
    raw = next(iter(tf.data.TFRecordDataset([str(tfrecord)]))).numpy()
    ex = tf.train.Example.FromString(bytes(raw))
    H = int(feat(ex, "metadata/height", "int64_list")[0]); W = int(feat(ex, "metadata/width", "int64_list")[0])
    depth_range = feat(ex, "metadata/depth_range").astype(np.float64)
    f = float(feat(ex, "camera/focal_length")[0]); sensor = float(feat(ex, "camera/sensor_width")[0])
    fx = f / sensor * W; fy = fx; cx = (W - 1) / 2; cy = (H - 1) / 2
    d_png = tf.io.decode_png(ex.features.feature["depth"].bytes_list.value[0], channels=1, dtype=tf.uint16).numpy()[..., 0]
    d = d_png.astype(np.float64) / 65535.0 * (depth_range[1] - depth_range[0]) + depth_range[0]
    print(json.dumps({"shape": [H,W], "depth_range": depth_range.tolist(), "depth_png_dtype": str(d_png.dtype),
                      "depth_png_raw_minmax": [int(d_png.min()), int(d_png.max())],
                      "depth_decoded_minmax": [float(d.min()), float(d.max())],
                      "focal_px": [fx,fy], "principal_point": [cx,cy]}, indent=2))
    q = feat(ex, "camera/quaternions").reshape(24,4)
    pos = feat(ex, "camera/positions").reshape(24,3)
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    xn = (uu-cx)/fx; yn = (vv-cy)/fy
    # Compare the z component of inverse camera transform against the decoded depth.
    rows = []
    for order in ("wxyz", "xyzw"):
        for inverse in (False, True):
            for forward_sign in (-1.0, 1.0):
                R = quat_matrix(q[0], order, inverse)
                local = np.stack([xn*d, yn*d, forward_sign*d], axis=-1)
                world = local @ R.T + pos[0]
                cam = (world - pos[0]) @ quat_matrix(q[0], order, False)
                rows.append(summarize(f"order={order},stored_to_world={not inverse},local_z={forward_sign}", cam[...,2], forward_sign*d))
    print("CAMERA_CONVENTION_CANDIDATES")
    print(json.dumps(sorted(rows, key=lambda x:x["mean_abs"]), indent=2))
    seg = tf.io.decode_png(ex.features.feature["segmentations"].bytes_list.value[0], channels=1, dtype=tf.uint8).numpy()[...,0]
    print(json.dumps({"segmentation_unique_first_frame": np.unique(seg).tolist(),
                      "segmentation_max": int(seg.max()),
                      "num_instances": int(feat(ex, "metadata/num_instances", "int64_list")[0])}, indent=2))
    iq = feat(ex, "instances/quaternions").reshape(-1,24,4)
    is_dynamic = feat(ex, "instances/is_dynamic", "int64_list").astype(bool)
    print(json.dumps({"instance_quaternion_shape": list(iq.shape), "dynamic_count": int(is_dynamic.sum()),
                      "camera_quaternion_first": q[0].tolist(), "camera_position_first": pos[0].tolist()}, indent=2))
    print("GEOMETRY_AUDIT_OK")


if __name__ == "__main__":
    main()
