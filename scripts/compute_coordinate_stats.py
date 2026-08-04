#!/usr/bin/env python3
"""Compute reusable train-only coordinate normalization statistics."""
from __future__ import annotations
import argparse
import pathlib
import sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from worldbridge.data import MOViFDataset
from worldbridge.pipeline import train_coordinate_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/dataset/MOVi-F")
    ap.add_argument("--output", required=True)
    ap.add_argument("--clip-length", type=int, default=21)
    ap.add_argument("--clip-start", type=int, default=0)
    ap.add_argument("--depth-tolerance", type=float, default=0.05)
    ap.add_argument("--depth-relative-tolerance", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    ds = MOViFDataset(args.data_root, "train", args.clip_length, args.clip_start, None, args.seed)
    mean, scale = train_coordinate_stats(ds, args.depth_tolerance, args.depth_relative_tolerance)
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez(
            handle, mean=mean, scale=scale, examples=len(ds),
            clip_length=args.clip_length, clip_start=args.clip_start,
            depth_tolerance=args.depth_tolerance,
            depth_relative_tolerance=args.depth_relative_tolerance,
        )
    tmp.replace(output)
    print({"output": str(output), "examples": len(ds), "mean": mean.tolist(), "scale": scale.tolist()}, flush=True)
    print(f"COORDINATE_STATS_CREATED: {output}", flush=True)


if __name__ == "__main__":
    main()
