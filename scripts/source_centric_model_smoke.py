#!/usr/bin/env python3
"""Bounded real-Wan smoke for the source-centric training path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldbridge.data import MOViFDataset
from worldbridge.dense4d import masked_visibility_bce
from worldbridge.dense4d_data import CoordinateStats
from worldbridge.dense4d_prefetch import (
    build_source_centric_batch,
    make_source_centric_plan,
    source_centric_loss_weights,
    weighted_masked_pair_smooth_l1,
)
from worldbridge.dense4d_runtime import build_real_model, encode_clean_video_latents, parameter_groups, precision_dtype


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("real-Wan smoke requires CUDA")
    torch.cuda.set_device(0 if device.index is None else device.index)
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    dtype = precision_dtype(config["precision"])
    dataset = MOViFDataset(
        config["data_root"], split="train", clip_length=21,
        clip_start=int(config.get("clip_start", 0)), max_examples=2, seed=int(config["seed"]),
    )
    samples = [dataset[index] for index in range(len(dataset))]
    stats = CoordinateStats.from_npz(config["coordinate_stats"])
    latent_start = time.perf_counter()
    latents = encode_clean_video_latents(samples, config["wan_root"], device)
    latent_seconds = time.perf_counter() - latent_start
    plan = make_source_centric_plan(len(samples), len(samples), 21, 0)
    batch = build_source_centric_batch(samples, stats, plan)
    model = build_real_model(config, device)
    groups = parameter_groups(model, config)
    optimizer = torch.optim.AdamW(groups, weight_decay=float(config.get("weight_decay", 0.0)))
    clean = torch.cat(latents).to(device=device, dtype=dtype)
    source = batch.source_cpu.to(device)
    target = batch.target_cpu.to(device)
    target_xyz = batch.normalized_xyz_cpu.to(device)
    valid = batch.valid_cpu.to(device)
    visible = batch.visible_cpu.to(device)
    pair_weights = torch.from_numpy(source_centric_loss_weights(plan.source, plan.target)).to(device)
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype == torch.bfloat16):
        prediction, z4d, output = model(clean, source, target)
        xyz_loss = weighted_masked_pair_smooth_l1(prediction.float(), target_xyz.float(), valid, pair_weights)
        visibility_loss = prediction.sum() * 0.0
        if output.visibility_logits is not None:
            visibility_loss = masked_visibility_bce(
                output.visibility_logits.float(), visible, valid, source, target,
                float(config.get("visibility_pos_weight", 1.0)),
            )
        loss = xyz_loss + float(config.get("lambda_visibility", 0.0)) * visibility_loss
    loss.backward()
    optimizer.step()
    backbone_gradient = any(parameter.grad is not None for parameter in model.backbone.parameters() if parameter.requires_grad)
    decoder_gradient = any(parameter.grad is not None for parameter in model.decoder.parameters() if parameter.requires_grad)
    result = {
        "samples": len(samples), "source_shape": list(source.shape), "target_shape": list(target.shape),
        "xyz_shape": list(target_xyz.shape), "valid_shape": list(valid.shape),
        "visible_shape": list(visible.shape), "clean_latent_shape": list(clean.shape),
        "z4d_shape": list(z4d.shape), "prediction_shape": list(prediction.shape),
        "loss": float(loss.detach()), "xyz_loss": float(xyz_loss.detach()),
        "visibility_loss": float(visibility_loss.detach()), "backbone_gradient": backbone_gradient,
        "decoder_gradient": decoder_gradient, "latent_seconds": latent_seconds,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
    }
    print(json.dumps(result, indent=2), flush=True)
    if not backbone_gradient or not decoder_gradient:
        raise RuntimeError("source-centric real-Wan smoke gradient gate failed")
    print("SOURCE_CENTRIC_MODEL_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
