#!/usr/bin/env python3
"""Create/re-encode the immutable 256px clean-latent tiers.

Metadata/geometry preparation remains dataset-specific. This command consumes
those indexes, always reads the original RGB, and never upsamples old latents.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from worldbridge.data import MOViFDataset
from worldbridge.dynamic_replica import DynamicReplicaDataset
from worldbridge.pointodyssey import PointOdysseyDataset
from worldbridge.training256 import MOViF256Dataset
from worldbridge.wan import WAN_LATENT_SHAPE_256, WanVAEEncoder


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while value := stream.read(chunk):
            digest.update(value)
    return digest.hexdigest()


def rows(cache_root: Path, split: str) -> list[dict]:
    path = cache_root / "splits" / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def build_movi_index(raw_root: Path, cache_root: Path) -> None:
    split_dir = cache_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        path = split_dir / f"{split}.jsonl"
        if path.exists():
            continue
        dataset = MOViFDataset(raw_root, split=split, clip_length=21, clip_start=0)
        with path.open("w", encoding="utf-8") as stream:
            for index in range(len(dataset)):
                stream.write(json.dumps({
                    "index": index, "raw_index": index,
                    "clip_id": f"movi-f/{split}/{index:06d}",
                    "parent_id": f"movi-f-{split}-{index:06d}",
                    "start": 0, "stride": 1,
                    "timestamps": [frame / 12.0 for frame in range(21)],
                }, separators=(",", ":")) + "\n")


def dataset_reader(name: str, raw_root: Path, cache_root: Path, split: str):
    if name == "kubric":
        build_movi_index(raw_root, cache_root)
        return MOViF256Dataset(raw_root, cache_root, split)
    if name == "pointodyssey":
        return PointOdysseyDataset(cache_root, split, image_size=256, raw_root=raw_root)
    if name == "dynamic_replica":
        return DynamicReplicaDataset(cache_root, split, image_size=256, raw_root=raw_root)
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("kubric", "pointodyssey", "dynamic_replica"), required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True,
                        help="new immutable 256 latent/index root")
    parser.add_argument("--geometry-cache-root", type=Path,
                        help="existing PO/DR v1 index root; defaults to --cache-root")
    parser.add_argument("--vae", type=Path, default=Path("/data/WorldBridge4D/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--io-workers", type=int, default=8)
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--max-clips", type=int)
    args = parser.parse_args()
    if not args.vae.is_file():
        raise FileNotFoundError(args.vae)
    output = args.cache_root / "latents" / "wan2.1_1.3b_fp32_256"
    output.mkdir(parents=True, exist_ok=True)
    if args.dataset != "kubric":
        geometry_root = (args.geometry_cache_root or args.cache_root).resolve()
        for split in args.splits:
            source_index = geometry_root / "splits" / f"{split}.jsonl"
            target_index = args.cache_root / "splits" / f"{split}.jsonl"
            if not source_index.is_file():
                raise FileNotFoundError(source_index)
            target_index.parent.mkdir(parents=True, exist_ok=True)
            if target_index.exists() and target_index.read_bytes() != source_index.read_bytes():
                raise RuntimeError(f"refusing to overwrite different 256 index: {target_index}")
            if not target_index.exists():
                target_index.write_bytes(source_index.read_bytes())
    encoder = WanVAEEncoder(args.vae, device=args.device, dtype=torch.float32,
                            expected_shape=WAN_LATENT_SHAPE_256)
    pool = ThreadPoolExecutor(max_workers=max(1, args.io_workers))
    artifacts = []
    for split in args.splits:
        # Consumers index each split locally. Train keeps the canonical path;
        # validation receives a separate tier with its own zero-based shards.
        split_output = output if split == "train" else args.cache_root / "latents" / f"wan2.1_1.3b_fp32_256_{split}"
        split_output.mkdir(parents=True, exist_ok=True)
        split_index = 0
        geometry_root = (args.geometry_cache_root or args.cache_root).resolve()
        dataset = dataset_reader(args.dataset, args.raw_root.resolve(), geometry_root, split)
        count = len(dataset) if args.max_clips is None else min(len(dataset), args.max_clips)
        for offset in range(0, count, args.shard_size):
            end = min(count, offset + args.shard_size)
            shard = split_output / f"shard_{split_index:08d}_{end-offset:05d}.safetensors"
            if shard.exists():
                split_index += end - offset
                continue
            latent_batches = []
            for batch_start in range(offset, end, args.batch_size):
                indices = range(batch_start, min(end, batch_start + args.batch_size))
                videos = list(pool.map(dataset.rgb if hasattr(dataset, "rgb") else lambda i: dataset.sample(i).rgb, indices))
                rgb = torch.from_numpy(np.stack(videos)).permute(0, 1, 4, 2, 3).contiguous()
                with torch.inference_mode():
                    latent_batches.append(encoder(rgb).float().cpu())
            values = torch.cat(latent_batches)
            if tuple(values.shape[1:]) != WAN_LATENT_SHAPE_256:
                raise RuntimeError(f"unexpected 256 latent shape: {tuple(values.shape)}")
            temporary = shard.with_suffix(".tmp.safetensors")
            save_file({"latents": values.contiguous()}, str(temporary), metadata={
                "dataset": args.dataset, "split": split, "first_split_index": str(offset),
            })
            temporary.replace(shard)
            artifacts.append({"path": str(shard), "sha256": sha256(shard), "count": len(values)})
            print(json.dumps({"dataset": args.dataset, "split": split, "encoded": end, "total": count}), flush=True)
            split_index += end - offset
    pool.shutdown(wait=True)
    metadata = {
        "dataset": args.dataset, "raw_root": str(args.raw_root.resolve()),
        "cache_root": str(args.cache_root.resolve()), "vae": str(args.vae.resolve()),
        "vae_sha256": sha256(args.vae), "latent_shape": list(WAN_LATENT_SHAPE_256),
        "posterior": "mean", "source_resolution": "original", "output_rgb_grid": [256, 256],
        "artifacts": artifacts,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("THREE_DATASET_256_LATENTS_READY", flush=True)


if __name__ == "__main__":
    main()
