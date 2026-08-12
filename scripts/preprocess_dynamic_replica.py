#!/usr/bin/env python3
"""Prepare Dynamic Replica clip indexes and deterministic Wan latent shards.

The raw Dynamic Replica release is treated as immutable. Geometry is referenced
by the original trajectory/depth/camera files; this stage does not duplicate
terabytes of raw frames.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]

def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)

def load_annotations(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return value

def stream_name(sequence_name: str, camera: str) -> str:
    return f"{sequence_name}_source_{camera}"

def read_rgb(path: Path) -> np.ndarray:
    # The protocol's crop is a shared 720x720 centre crop followed by resize.
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (1280, 720):
            raise ValueError(f"Unexpected RGB size for {path}: {im.size}")
        im = im.crop((280, 0, 1000, 720)).resize((128, 128), Image.Resampling.LANCZOS)
        return np.asarray(im, dtype=np.uint8)

def read_video(raw_train_root: Path, row: dict[str, Any]) -> torch.Tensor:
    frames = np.stack([read_rgb(raw_train_root / x["rgb"]) for x in row["frames"]])
    return torch.from_numpy(frames).permute(0, 3, 1, 2)


def git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "0" * 40

def build_rows(raw_root: Path, ann_path: Path, seed: int, camera_filter: str) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    annotations = load_annotations(ann_path)
    by_stream: dict[str, list[dict[str, Any]]] = {}
    for e in annotations:
        camera = e.get("camera_name")
        seq = e.get("sequence_name")
        if camera not in {"left", "right"} or not isinstance(seq, str):
            continue
        if camera_filter != "both" and camera != camera_filter:
            continue
        by_stream.setdefault(stream_name(seq, camera), []).append(e)
    for key in by_stream:
        by_stream[key].sort(key=lambda e: int(e["frame_number"]))
        if len(by_stream[key]) < 21:
            del by_stream[key]
    parents = sorted({e["sequence_name"] for values in by_stream.values() for e in values})
    # Deterministic parent-level 90/10 holdout. Both stereo streams stay in the
    # same split because their parent id is the sequence name.
    rng = np.random.default_rng(seed)
    shuffled = parents.copy()
    rng.shuffle(shuffled)
    n_valid = max(1, round(len(shuffled) * 0.1))
    valid_parents = set(shuffled[:n_valid])
    rows: list[dict[str, Any]] = []
    split_streams: dict[str, list[str]] = {"train": [], "validation": []}
    for sname in sorted(by_stream):
        values = by_stream[sname]
        parent = values[0]["sequence_name"]
        split = "validation" if parent in valid_parents else "train"
        split_streams[split].append(sname)
        # Non-overlapping, ordered 21-frame clips; no padding/interpolation.
        for start in range(0, len(values) - 20, 21):
            window = values[start:start + 21]
            clip_id = f"{sname}_{start:06d}_x1"
            rows.append({
                "clip_id": clip_id,
                "parent_id": parent,
                "stream": sname,
                "start": int(window[0]["frame_number"]),
                "stride": 1,
                "timestamps": [float(x["frame_timestamp"]) for x in window],
                "frames": [
                    {
                        "rgb": x["image"]["path"],
                        "depth": x["depth"]["path"],
                        "trajectory": x["trajectories"]["path"],
                        "instance_id_map": x.get("instance_id_map_path"),
                        "viewpoint": x["viewpoint"],
                        "size": x["image"]["size"],
                    } for x in window
                ],
                "source_release": "Dynamic Replica dynamic_stereo train release",
            })
    rows.sort(key=lambda r: r["clip_id"])
    for i, r in enumerate(rows):
        r["index"] = i
    return rows, split_streams

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, default=Path("/dataset/Dynamic_dataset/dynamic_stereo"))
    ap.add_argument("--output-root", type=Path, default=Path("/data/WorldBridge4D-persistent/datasets/dynamic_stereo_worldbridge4d_v1"))
    ap.add_argument("--vae", type=Path, default=Path("/data/WorldBridge4D/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"))
    ap.add_argument("--seed", type=int, default=20260811)
    ap.add_argument("--camera", choices=("left", "right", "both"), default="left")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--shard-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--io-workers", type=int, default=16)
    ap.add_argument("--metadata-only", action="store_true")
    args = ap.parse_args()
    raw_root = args.raw_root.resolve()
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "splits").mkdir(exist_ok=True)
    (out / "samples").mkdir(exist_ok=True)
    (out / "latents" / "wan2.1_1.3b_fp32").mkdir(parents=True, exist_ok=True)
    ann = raw_root / "train" / "frame_annotations_train.jgz"
    rows, split_streams = build_rows(raw_root, ann, args.seed, args.camera)
    # Parent membership is recovered from the deterministic split by stream set.
    train_streams, valid_streams = set(split_streams["train"]), set(split_streams["validation"])
    train = [r for r in rows if r["stream"] in train_streams]
    valid = [r for r in rows if r["stream"] in valid_streams]
    write_jsonl(out / "splits" / "train.jsonl", train)
    write_jsonl(out / "splits" / "validation.jsonl", valid)
    write_jsonl(out / "samples" / "index.jsonl", rows)
    (out / "PREPROCESSING_STATE.json").write_text(json.dumps({
        "status": "metadata_ready_latents_pending",
        "rows": len(rows), "train": len(train), "validation": len(valid),
        "seed": args.seed, "camera": args.camera, "raw_root": str(raw_root), "vae": str(args.vae),
        "git_commit": git_commit(), "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    if args.metadata_only:
        print(json.dumps({"rows": len(rows), "train": len(train), "validation": len(valid)}))
        return
    if not args.vae.is_file():
        raise FileNotFoundError(args.vae)
    from worldbridge.wan import WanVAEEncoder
    device = torch.device(args.device)
    encoder = WanVAEEncoder(args.vae, device=device, dtype=torch.float32)
    pool = ThreadPoolExecutor(max_workers=args.io_workers)
    # Resume safely at shard boundaries. Each shard contains only tensors and
    # is directly mmap-loadable by safetensors.
    for first in range(0, len(rows), args.shard_size):
        batch_rows = rows[first:first + args.shard_size]
        shard = out / "latents" / "wan2.1_1.3b_fp32" / f"shard_{first:08d}_{len(batch_rows):05d}.safetensors"
        if shard.exists():
            continue
        latents: list[torch.Tensor] = []
        for batch_start in range(0, len(batch_rows), args.batch_size):
            mini_rows = batch_rows[batch_start:batch_start + args.batch_size]
            videos = list(pool.map(lambda row: read_video(raw_root / "train", row), mini_rows))
            rgb = torch.stack(videos)
            with torch.inference_mode():
                latents.extend(encoder(rgb).cpu().contiguous())
        tensor = torch.stack(latents).to(torch.float32)
        save_file({"latents": tensor}, str(shard), metadata={"clip_start": str(first), "clip_count": str(len(batch_rows))})
        print(f"wrote {shard} {tuple(tensor.shape)}", flush=True)
    pool.shutdown(wait=True)
    (out / "PREPROCESSING_STATE.json").write_text(json.dumps({
        "status": "latents_ready_geometry_adapter_pending", "rows": len(rows),
        "train": len(train), "validation": len(valid), "seed": args.seed, "camera": args.camera,
        "raw_root": str(raw_root), "vae": str(args.vae), "vae_sha256": sha256_file(args.vae),
        "git_commit": git_commit(), "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
