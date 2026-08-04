#!/usr/bin/env python3
"""Audit the native MOVi-F 128x128 TFRecord schema and one deterministic example.

This is deliberately an audit rather than an adapter: it reports the exact keys,
encoded dtypes/shapes, and finite ranges before geometry code interprets them.
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np


def decode_feature(feature):
    kind = feature.WhichOneof("kind")
    if kind == "bytes_list":
        return {"kind": kind, "count": len(feature.bytes_list.value),
                "byte_lengths": [len(x) for x in feature.bytes_list.value[:4]]}
    if kind == "float_list":
        vals = list(feature.float_list.value)
        return {"kind": kind, "count": len(vals),
                "min": float(np.min(vals)) if vals else None,
                "max": float(np.max(vals)) if vals else None}
    if kind == "int64_list":
        vals = list(feature.int64_list.value)
        return {"kind": kind, "count": len(vals),
                "min": int(np.min(vals)) if vals else None,
                "max": int(np.max(vals)) if vals else None}
    return {"kind": kind}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-records", type=int, default=2)
    args = ap.parse_args()
    root = pathlib.Path(args.data_root)
    version_dirs = sorted((root / "128x128").glob("*") if (root / "128x128").exists() else root.glob("*"))
    version = next((p for p in version_dirs if p.is_dir()), root)
    features_path = version / "features.json"
    print(json.dumps({"data_root": str(root), "version_dir": str(version),
                      "features_json": str(features_path)}, indent=2))
    if features_path.exists():
        print("FEATURES_JSON_BEGIN")
        print(features_path.read_text())
        print("FEATURES_JSON_END")
    try:
        import tensorflow as tf
    except Exception as exc:
        print(f"TENSORFLOW_IMPORT_ERROR: {type(exc).__name__}: {exc}")
        sys.exit(2)
    files = sorted(version.glob(f"movi_f-{args.split}.tfrecord-*"))
    split_names = sorted({p.name.split("movi_f-", 1)[1].split(".tfrecord-", 1)[0] for p in version.glob("movi_f-*.tfrecord-*")})
    print(f"AVAILABLE_SPLITS: {json.dumps(split_names)}")
    print(f"TFRECORD_FILES: {len(files)}")
    if not files:
        raise FileNotFoundError(f"no TFRecords for split={args.split} under {version}")
    ds = tf.data.TFRecordDataset([str(files[0])])
    for index, raw in enumerate(ds.take(args.max_records)):
        ex = tf.train.Example.FromString(bytes(raw.numpy()))
        print(f"EXAMPLE_{index}_KEYS: {json.dumps(sorted(ex.features.feature))}")
        decoded = {k: decode_feature(v) for k, v in sorted(ex.features.feature.items())}
        print(f"EXAMPLE_{index}_SUMMARY: {json.dumps(decoded, sort_keys=True)}")
        for key, feature in sorted(ex.features.feature.items()):
            if feature.WhichOneof("kind") == "bytes_list" and feature.bytes_list.value:
                vals = feature.bytes_list.value
                print(f"BYTES_FIELD {key}: count={len(vals)} first16={vals[0][:16].hex()}")
    print("AUDIT_OK")


if __name__ == "__main__":
    main()
