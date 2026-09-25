"""DR 512x512 square native cache producer: manifest + VAE latent + RGB.

Threaded pipeline: N reader threads decode 21 raw 720x1280 PNGs, centre-crop
720x720, LANCZOS-resize to 512x512, and feed a single GPU VAE encoder. Latent
and RGB publications are write-once, per-clip atomic, and idempotent across a
SIGTERM restart (the next run skips already-published indices).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from worldbridge.data.cache.native import CONTRACT_DR512, NativeLatentCache, file_sha256
from worldbridge.data.cache.native_rgb import NativeRGBCache
from worldbridge.data.native_inputs import (
    build_dr512_manifest, validate_manifest, _check_source, rgb_identity,
)
from worldbridge.models.wan import WanVAEEncoder
from worldbridge.utils.io import atomic_json

RAW_W, RAW_H = 1280, 720
CROP_X, CROP_Y, CROP_SIZE = 280, 0, 720
TARGET = 512
SMOKE_COUNT = 3


def read_clip(manifest: dict, i: int) -> tuple[int, str, np.ndarray, dict]:
    row = manifest["records"][i]
    frames = []
    for path in row["paths"]:
        _check_source(manifest, path)
        with Image.open(path) as im:
            if im.size != (RAW_W, RAW_H):
                raise ValueError(f"unexpected Dynamic Replica raw RGB size {im.size}: {path}")
            im = im.convert("RGB").crop((CROP_X, CROP_Y, CROP_X + CROP_SIZE, CROP_Y + CROP_SIZE))
            frames.append(np.asarray(
                im.resize((TARGET, TARGET), Image.Resampling.LANCZOS), dtype=np.uint8,
            ))
    rgb = np.stack(frames)
    return i, row["clip_id"], rgb, rgb_identity(rgb)


def encode(encoder: WanVAEEncoder, rgb: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(rgb.transpose(0, 3, 1, 2)[None]).float().div_(255.0).cuda()
    with torch.no_grad():
        latent = encoder(x)
    return latent.float().cpu().numpy()[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--stage", choices=["smoke", "bulk"], required=True)
    p.add_argument("--readers", type=int, default=8)
    p.add_argument("--progress-every", type=int, default=25)
    args = p.parse_args()
    assert args.readers >= 1 and args.progress_every >= 1
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == config_cache_gpu(args.config)
    assert not any(os.environ.get(k) for k in
                   ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF", "PYTORCH_NO_CUDA_MEMORY_CACHING"))
    assert torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    torch.set_num_threads(1)

    config = yaml.safe_load(Path(args.config).read_text())
    vae_sha = file_sha256(config["vae_checkpoint"])
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "dynamic_replica.json"
    manifest = build_dr512_manifest(config, vae_sha)
    validate_manifest(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    if not output.joinpath("manifest.json").exists():
        output.joinpath("manifest.json").write_text(json.dumps(manifest, sort_keys=True))

    latent_cache = NativeLatentCache(
        config["cache_root"], "dynamic_replica", manifest["sha256"],
        tuple(manifest["latent_shape"]), vae_sha, contract=CONTRACT_DR512,
    )
    rgb_cache = NativeRGBCache(config["rgb_root"], manifest)

    stopping = False
    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    encoder = WanVAEEncoder(config["vae_checkpoint"], device="cuda:0", dtype=torch.float32,
                            expected_shape=tuple(manifest["latent_shape"]))
    assert not encoder.vae.use_tiling and not encoder.vae.use_slicing
    print(json.dumps({
        "event": "DR512_VAE_LOADED", "manifest_sha256": manifest["sha256"],
        "native_hw": manifest["native_hw"], "latent_shape": manifest["latent_shape"],
        "transform": manifest["transform"], "stage": args.stage,
        "readers": args.readers, "vae_sha256": vae_sha,
    }), flush=True)

    if args.stage == "smoke":
        indices = list(range(SMOKE_COUNT))
        first = None
        for i in indices:
            idx, clip_id, rgb, identity = read_clip(manifest, i)
            latent = encode(encoder, rgb)
            if first is None:
                first = latent.copy()
            latent_cache.write(idx, clip_id, latent, identity)
            rgb_cache.write(idx, rgb, identity)
            print(json.dumps({"event": "DR512_SMOKE_CLIP", "index": idx,
                              "clip_id": clip_id, "rgb_shape": list(rgb.shape),
                              "latent_shape": list(latent.shape)}), flush=True)
        repeat = encode(encoder, read_clip(manifest, 0)[2])
        repeat_error = float(np.max(np.abs(repeat - first)))
        assert repeat_error == 0, f"non-deterministic VAE mean: {repeat_error}"
        smoke = {"manifest_sha256": manifest["sha256"], "repeat_max_abs_error": repeat_error,
                 "contract": CONTRACT_DR512, "transform": manifest["transform"],
                 "vae_sha256": vae_sha, "clips": indices}
        atomic_json(output / "smoke_complete.json", smoke)
        print(json.dumps({"event": "DR512_SMOKE_OK", **smoke}), flush=True)
        return

    smoke = json.loads((output / "smoke_complete.json").read_text())
    assert smoke["manifest_sha256"] == manifest["sha256"]
    assert smoke["repeat_max_abs_error"] == 0 and smoke["contract"] == CONTRACT_DR512

    indices = list(range(len(manifest["records"])))
    total = len(indices)
    processed = written = reused = 0
    begin = time.monotonic()

    def read_only(i: int):
        return read_clip(manifest, i)

    with ThreadPoolExecutor(max_workers=args.readers) as pool:
        futures = {pool.submit(read_only, i): i for i in indices}
        for future in as_completed(futures):
            if stopping:
                future.cancel()
                continue
            idx, clip_id, rgb, identity = future.result()
            latent = encode(encoder, rgb)  # GPU: single-threaded consumer
            written += int(latent_cache.write(idx, clip_id, latent, identity))
            reused += int(not rgb_cache.write(idx, rgb, identity))
            processed += 1
            if processed % args.progress_every == 0 or processed == total:
                rate = processed / max(time.monotonic() - begin, 1e-6)
                print(json.dumps({
                    "event": "DR512_BULK_PROGRESS", "processed": processed, "total": total,
                    "written": written, "reused": reused,
                    "clips_per_second": round(rate, 3),
                    "eta_seconds": round((total - processed) / max(rate, 1e-9)),
                    "peak_cuda_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                }), flush=True)
        if stopping:
            print(json.dumps({"event": "DR512_BULK_STOPPED_EARLY", "processed": processed,
                              "total": total}), flush=True)
            sys.exit(1)

    report = {"event": "DR512_CACHE_COMPLETE", "manifest_sha256": manifest["sha256"],
              "contract": CONTRACT_DR512, "transform": manifest["transform"],
              "clips": total, "written": written, "reused": reused,
              "elapsed_seconds": round(time.monotonic() - begin, 1),
              "latent_root": str(latent_cache.root), "rgb_root": str(rgb_cache.root),
              "peak_cuda_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3)}
    atomic_json(output / "complete.json", report)
    atomic_json(output / "ready.json", {
        "manifest_sha256": manifest["sha256"], "config_sha256": file_sha256(args.config),
        "vae_sha256": vae_sha, "training_ready": False, "complete": True,
    })
    atomic_json(latent_cache.root / "bulk_complete.json", {
        "manifest_sha256": manifest["sha256"], "processed": total,
        "all_existing_training_index_entries_verified": True,
        "contract": CONTRACT_DR512, "rgb_root": str(rgb_cache.root),
        "elapsed_seconds": round(time.monotonic() - begin, 1),
    })
    atomic_json(rgb_cache.root / "bulk_complete.json", {
        "manifest_sha256": manifest["sha256"], "processed": total,
        "requested": total, "contract": "native_rgb_uint8_snapshot_v1",
        "elapsed_seconds": round(time.monotonic() - begin, 1),
    })
    print(json.dumps(report), flush=True)


def config_cache_gpu(config_path: str) -> str:
    cfg = yaml.safe_load(Path(config_path).read_text())
    return str(cfg["cache_gpu"])


if __name__ == "__main__":
    main()
