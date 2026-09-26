#!/usr/bin/env python3
"""H031 crossed 256/512 matched fixed-clip EPE evaluation (Kubric, train split).

Runs the old256 checkpoint (h030 rgb1x step155000) and the current native512
checkpoint (h031 step183000) at both 256 and 512 on the SAME fixed train clips,
producing a 2x2 point-weighted raw EPE (meters) matrix.

Key facts that make this a clean cross:
- Both checkpoints share the identical architecture (1056 model keys,
  decoder_only, frozen Wan+geometry) and the identical coordinate mean/scale, so
  the metric is in the same meters frame and only the decoder weights differ.
- ``native_512`` only switches the decoder query grid / geometry dense spatial
  size at runtime; it introduces no new weights, so each checkpoint loads strict
  into a single native_512-built model that serves both resolutions.
- Kubric train clips exist in both the 256 cache (MOViF256Dataset) and the
  native512 cache (NativeKubricDataset) with the same index/clip_id. The 256
  train latent shards were pruned, so the 256 latent is recomputed on the fly
  with the frozen deterministic VAE from the 256-resized RGB (identical to the
  original cached values); the 256 GT/RGB come from the audited 256 caches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from worldbridge.data.factory import load_dataset  # noqa: E402
from worldbridge.data.text_conditions import load_inference_text_condition  # noqa: E402
from worldbridge.models.factory import build_real_model, precision_dtype  # noqa: E402
from worldbridge.models.wan import WAN_LATENT_SHAPE_256, WanVAEEncoder  # noqa: E402

FRAME_COUNT = 21
DEFAULT_SOURCES = (0, 5, 10, 15, 20)
ALL_TARGETS = tuple(range(FRAME_COUNT))
PERSISTENT_ROOT = Path("/data/WorldBridge4D-runs")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def emit(event: str, **values):
    print(json.dumps({"event": event, **values}, allow_nan=False), flush=True)


def load_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if int(checkpoint.get("training_state", {}).get("global_step", -1)) < 0:
        raise ValueError(f"checkpoint missing training_state.global_step: {path}")
    return checkpoint


def make_native_512_config(base_config: dict) -> dict:
    """Build a single model config that serves both 256 and 512 latent inputs."""
    config = dict(base_config)
    # The native512 route flag enables the 64x64 latent / adaptive query grid.
    config["native_kubric512_b1_a4_k9"] = True
    config["gradient_checkpointing"] = False
    config["precision"] = "bf16"
    # Remove runtime-only Wan weight loading; strict checkpoint load supplies
    # the frozen backbone exactly as trained.
    for key in ("wan_checkpoint", "wan_dit_root", "empty_text_condition"):
        config.pop(key, None)
    return config


def clip_tensor_from_numpy(rgb: np.ndarray, device: torch.device,
                           dtype: torch.dtype) -> torch.Tensor:
    """uint8 [T,H,W,3] -> [T,3,H,W] in [-1,1]."""
    if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"RGB clip must be uint8 [T,H,W,3], got {rgb.shape}/{rgb.dtype}")
    tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2).to(device, dtype=dtype)
    return tensor / 127.5 - 1.0


def decode_targets(model, z4d, source: int, source_pyramid, targets: list[int],
                   target_chunk: int, device: torch.device) -> torch.Tensor:
    outputs = []
    for start in range(0, len(targets), target_chunk):
        chunk = targets[start:start + target_chunk]
        outputs.append(model.decoder(
            z4d,
            torch.full((1, len(chunk)), source, device=device, dtype=torch.long),
            torch.tensor(chunk, device=device, dtype=torch.long)[None],
            source_pyramid=source_pyramid,
        ).normalized_xyz)
    return torch.cat(outputs, dim=1)[0].float().cpu()


def encode_256_latent(vae: WanVAEEncoder, rgb256: np.ndarray, device: torch.device,
                      dtype: torch.dtype) -> torch.Tensor:
    """Recompute the 256 clean latent from the 256-resized RGB (frozen VAE)."""
    tensor = torch.from_numpy(rgb256).permute(0, 3, 1, 2)[None].to(device)
    with torch.inference_mode():
        latent = vae(tensor).float().to(device, dtype=dtype)
    del tensor
    return latent


def evaluate_cell(model, checkpoint, dataset, vae, resolution, condition,
                  device, dtype, clip_indices, sources, targets, target_chunk) -> dict:
    mean = torch.as_tensor(np.asarray(checkpoint["coordinate_mean"], np.float32)).reshape(1, 3, 1, 1)
    scale = torch.as_tensor(np.asarray(checkpoint["coordinate_scale"], np.float32)).reshape(1, 3, 1, 1)
    point_error_sum = 0.0
    point_count = 0
    per_clip = []
    started = time.monotonic()
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=dtype, enabled=device.type == "cuda",
    ):
        for index in clip_indices:
            clip_started = time.monotonic()
            rgb = dataset.rgb(index)
            if resolution == 256:
                latent = encode_256_latent(vae, rgb, device, dtype)
            else:
                latent = torch.from_numpy(dataset.clean_latent(index))[None].to(device, dtype=dtype)
            z4d = model.backbone(latent, condition)
            clip_err_sum = 0.0
            clip_count = 0
            for source in sources:
                xyz_np, valid_np = dataset.source_all_targets(index, source)
                target = torch.from_numpy(np.asarray(xyz_np)[list(targets)]).float()
                valid = torch.from_numpy(np.asarray(valid_np)[list(targets)]).bool()
                src_rgb = clip_tensor_from_numpy(rgb[source:source + 1], device, dtype)
                pyramid = model.decoder.encode_source_rgb(src_rgb)
                normalized = decode_targets(model, z4d, source, pyramid, list(targets),
                                            target_chunk, device)
                raw = normalized * scale + mean
                error = torch.linalg.vector_norm(raw - target, dim=1)
                mask = valid & torch.isfinite(error)
                clip_err_sum += float((error * mask).sum())
                clip_count += int(mask.sum())
                del target, valid, src_rgb, pyramid, normalized, raw, error, mask
            del latent, rgb, z4d
            if clip_count:
                per_clip.append({"index": int(index), "points": clip_count,
                                 "epe_m": clip_err_sum / clip_count,
                                 "seconds": time.monotonic() - clip_started})
                point_error_sum += clip_err_sum
                point_count += clip_count
            else:
                per_clip.append({"index": int(index), "points": 0, "epe_m": None,
                                 "seconds": time.monotonic() - clip_started})
    return {
        "point_weighted_mean_epe_m": point_error_sum / point_count if point_count else None,
        "valid_points": int(point_count),
        "clips": len(per_clip),
        "per_clip": per_clip,
        "elapsed_seconds": time.monotonic() - started,
    }


def select_clips(count: int, seed: int, dataset_len: int) -> list[int]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(dataset_len)[:count]
    return sorted(int(i) for i in indices)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-checkpoint", required=True)
    parser.add_argument("--current-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--clips", type=int, default=40, help="fixed train clip count")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--sources", type=int, nargs="+", default=list(DEFAULT_SOURCES))
    parser.add_argument("--target-chunk", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-clips", type=int, help="cap clips for a gate/smoke run")
    parser.add_argument("--list-clips", action="store_true",
                        help="print the plan and exit without GPU work")
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    if not output_root.is_relative_to(PERSISTENT_ROOT):
        raise ValueError(f"output must be under {PERSISTENT_ROOT}, got {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    old_path = Path(args.old_checkpoint).resolve()
    cur_path = Path(args.current_checkpoint).resolve()
    old_sha = digest(old_path)
    cur_sha = digest(cur_path)

    old_ck = load_checkpoint(old_path)
    cur_ck = load_checkpoint(cur_path)
    old_cfg = dict(old_ck["config"])
    cur_cfg = dict(cur_ck["config"])

    # 256 route from the old checkpoint's own config (allow missing train
    # latent shards; the 256 latent is recomputed on the fly). native512 route
    # from the current checkpoint's own config. Both index the same train clips.
    dataset_256 = load_dataset(old_cfg, "kubric", split="train", allow_missing_latents=True)
    dataset_512 = load_dataset(cur_cfg, "kubric", split="train")
    if len(dataset_256) != len(dataset_512):
        raise ValueError(f"256/native512 train lengths differ: {len(dataset_256)} vs {len(dataset_512)}")

    clip_indices = select_clips(args.clips, args.seed, len(dataset_256))
    if args.limit_clips:
        clip_indices = clip_indices[:args.limit_clips]
    sources = tuple(sorted(set(args.sources)))
    targets = ALL_TARGETS

    plan = {
        "old_checkpoint": str(old_path), "old_checkpoint_sha256": old_sha,
        "old_checkpoint_step": int(old_ck["training_state"]["global_step"]),
        "current_checkpoint": str(cur_path), "current_checkpoint_sha256": cur_sha,
        "current_checkpoint_step": int(cur_ck["training_state"]["global_step"]),
        "clip_indices": clip_indices, "sources": list(sources), "targets": list(targets),
        "target_chunk": args.target_chunk, "seed": args.seed,
        "metric": "point_weighted_mean_raw_epe_m_over_valid_points",
        "caveat": "train-split fixed clips, not held-out generalization; 256 latent recomputed on the fly with the frozen VAE",
    }
    (output_root / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if args.list_clips:
        emit("CROSS_EVAL_PLAN", plan=plan)
        return

    device = torch.device(args.device)
    dtype = precision_dtype("bf16")
    condition, _prompt_meta = load_inference_text_condition(cur_cfg, "kubric")
    condition = condition.to(device, dtype=dtype)

    vae = WanVAEEncoder(
        Path(cur_cfg["wan_root"]) / "Wan2.1_VAE.pth", device=device, dtype=torch.float32,
        expected_shape=tuple(WAN_LATENT_SHAPE_256),
    )

    labels = ("old256", "current512")
    datasets = {"256": dataset_256, "512": dataset_512}
    summaries: dict[str, dict] = {}

    for label, checkpoint in zip(labels, (old_ck, cur_ck)):
        config = make_native_512_config(cur_cfg)
        model = build_real_model(config, device, load_wan_pretrained=False)
        missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
        if missing or unexpected:
            raise ValueError(
                f"strict-load mismatch for {label}: missing={missing[:5]} unexpected={unexpected[:5]}"
            )
        model.requires_grad_(False).eval()
        emit("model_loaded", label=label,
             parameters=sum(p.numel() for p in model.parameters()),
             trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        for res_label, dataset in datasets.items():
            resolution = int(res_label)
            cell_id = f"{label}@{res_label}"
            cell_path = output_root / f"{cell_id}.json"
            if cell_path.is_file():
                summaries[cell_id] = json.loads(cell_path.read_text())
                emit("cell_reused", cell=cell_id,
                     point_weighted_mean_epe_m=summaries[cell_id]["point_weighted_mean_epe_m"])
                continue
            summary = evaluate_cell(model, checkpoint, dataset, vae, resolution, condition,
                                    device, dtype, clip_indices, sources, targets, args.target_chunk)
            summary.update({
                "cell": cell_id, "checkpoint": label, "resolution": res_label,
                "checkpoint_sha256": plan["old_checkpoint_sha256"] if label == "old256"
                else plan["current_checkpoint_sha256"],
                "checkpoint_step": plan["old_checkpoint_step"] if label == "old256"
                else plan["current_checkpoint_step"],
            })
            (output_root / f"{cell_id}.json").write_text(json.dumps(summary, indent=2) + "\n")
            summaries[cell_id] = summary
            emit("cell_complete", cell=cell_id,
                 point_weighted_mean_epe_m=summary["point_weighted_mean_epe_m"],
                 valid_points=summary["valid_points"],
                 elapsed_seconds=summary["elapsed_seconds"])
        del model

    table = {}
    for label in labels:
        for res in ("256", "512"):
            cell_id = f"{label}@{res}"
            table[cell_id] = summaries[cell_id]["point_weighted_mean_epe_m"]
    result = {"plan": plan, "cells": summaries,
              "matrix_epe_m": {
                  "old256": {"256": table["old256@256"], "512": table["old256@512"]},
                  "current512": {"256": table["current512@256"], "512": table["current512@512"]},
              }}
    (output_root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    emit("CROSS_EVAL_OK", matrix_epe_m=result["matrix_epe_m"])


if __name__ == "__main__":
    main()
