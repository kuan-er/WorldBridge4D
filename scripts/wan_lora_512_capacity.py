#!/usr/bin/env python3
"""One-step 512x512 Wan1.3B-LoRA Dense4D capacity gate on MOVi-F.

This deliberately uses one source/target pair so the test isolates whether the
high-resolution Wan path plus a chunkable decoder can complete
forward/backward/Adam on one A100. Formal all-target training would call the
same decoder repeatedly with target chunks and accumulate the loss.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from worldbridge.data import MOViFDataset
from worldbridge.dense4d import masked_pair_smooth_l1
from worldbridge.dense4d_runtime import build_real_model, parameter_groups, precision_dtype
from worldbridge.geometry import GeometryBuilder
from worldbridge.wan import WanVAEEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--target", type=int, default=20)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    device = torch.device(args.device)
    torch.manual_seed(int(config.get("seed", 2026)))
    np.random.seed(int(config.get("seed", 2026)))

    dataset = MOViFDataset(config["data_root"], split="train", clip_length=21,
                           clip_start=0, max_examples=1, seed=int(config.get("seed", 2026)))
    sample = dataset[0]
    if (sample.height, sample.width) != (512, 512):
        raise RuntimeError(f"expected native 512x512 sample, got {sample.height}x{sample.width}")
    latent_shape = tuple(int(value) for value in config["wan_latent_shape"])
    encoder = WanVAEEncoder(Path(config["wan_root"]) / "Wan2.1_VAE.pth", device=device,
                            dtype=torch.float32, expected_shape=latent_shape)
    rgb = torch.from_numpy(sample.rgb).permute(0, 3, 1, 2)[None].to(device)
    with torch.inference_mode():
        clean = encoder(rgb).to(dtype=precision_dtype(config["precision"]))
    del encoder, rgb
    torch.cuda.empty_cache()

    source, target = int(args.source), int(args.target)
    builder = GeometryBuilder(sample)
    xyz, _, valid, _ = builder.trajectory_block(source, coordinate_frame="source",
                                                 compute_visibility=False)
    target_xyz = torch.from_numpy(xyz[:, target].reshape(512, 512, 3).transpose(2, 0, 1))[None, None].to(device)
    valid_t = torch.from_numpy(valid[:, target].reshape(512, 512))[None, None].to(device)
    mean = torch.as_tensor(config["coordinate_mean"], device=device).view(1, 1, 3, 1, 1)
    scale = torch.as_tensor(config["coordinate_scale"], device=device).view(1, 1, 3, 1, 1)
    target_norm = (target_xyz - mean) / scale
    source_t = torch.tensor([[source]], device=device)
    target_t = torch.tensor([[target]], device=device)

    model = build_real_model(config, device)
    groups = parameter_groups(model, config)
    optimizer = torch.optim.AdamW(groups, weight_decay=float(config.get("weight_decay", 0.0)))
    dtype = precision_dtype(config["precision"])
    trainable = {group["name"]: sum(parameter.numel() for parameter in group["params"]) for group in groups}
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.autocast("cuda", dtype=dtype):
        prediction, z4d, _ = model(clean, source_t, target_t)
        loss = masked_pair_smooth_l1(prediction.float(), target_norm.float(), valid_t,
                                     beta=float(config.get("smooth_l1_beta", 0.05)))
    after_forward = torch.cuda.max_memory_allocated(device) / 2**30
    loss.backward()
    after_backward = torch.cuda.max_memory_allocated(device) / 2**30
    lora_grad = any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for name, parameter in model.named_parameters() if "lora_" in name
    )
    adapter_ids = {id(parameter) for parameter in model.backbone.adapter_parameters}
    adapter_grad = any(
        id(parameter) in adapter_ids and parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    decoder_grad = any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.decoder.parameters()
    )
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    result = {
        "status": "pass", "loss": float(loss.detach()), "elapsed_seconds": time.perf_counter() - started,
        "sample": sample.video_name, "input_shape": list(sample.rgb.shape), "latent_shape": list(clean.shape),
        "wan_token_grid": [latent_shape[1], latent_shape[2] // 2, latent_shape[3] // 2],
        "geometry_shape": list(z4d.dense.shape), "prediction_shape": list(prediction.shape),
        "valid_points": int(valid_t.sum()), "trainable_parameters": trainable,
        "forward_peak_gib": after_forward, "backward_peak_gib": after_backward,
        "optimizer_peak_gib": peak, "lora_gradient": lora_grad,
        "adapter_gradient": adapter_grad, "decoder_gradient": decoder_grad,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print("WAN_LORA_512_CAPACITY_OK", flush=True)


if __name__ == "__main__":
    main()
