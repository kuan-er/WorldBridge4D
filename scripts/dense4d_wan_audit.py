#!/usr/bin/env python3
"""Real Wan clean-latent, timestep, sign, topology, and gradient audit."""
from __future__ import annotations

import argparse
import gc
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from worldbridge.data import MOViFDataset
from worldbridge.dense4d import FeedForwardWanBackbone, verify_flow_velocity_algebra
from worldbridge.dense4d_runtime import load_empty_condition
from worldbridge.wan import WAN_LATENT_SHAPE, WanDiTMapping, WanVAEEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/dataset/MOVi-F")
    parser.add_argument("--wan-root", default="/dataset/Wan2.1-T2V-1.3B")
    parser.add_argument("--empty-condition", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    dataset = MOViFDataset(args.data_root, split="train", clip_length=21, clip_start=0, max_examples=1)
    sample = dataset[0]
    rgb = torch.from_numpy(sample.rgb).permute(0, 3, 1, 2)[None].to(device)

    encoder = WanVAEEncoder(pathlib.Path(args.wan_root) / "Wan2.1_VAE.pth", device=device, dtype=torch.float32)
    with torch.inference_mode():
        first = encoder(rgb)
        second = encoder(rgb)
    deterministic_max = float((first - second).abs().max())
    clean_latent = first.cpu()
    del encoder, first, second, rgb
    gc.collect(); torch.cuda.empty_cache()

    condition = load_empty_condition(args.empty_condition)
    mapping = WanDiTMapping(
        pathlib.Path(args.wan_root) / "diffusion_pytorch_model.safetensors",
        condition=condition, device=device, dtype=torch.bfloat16,
    )
    mapping.dit.enable_gradient_checkpointing()
    mapping.train()
    recorded_timesteps = []

    def record_timestep(module, positional, keyword):
        timestep = positional[0] if positional else keyword["timestep"]
        recorded_timesteps.append(timestep.detach().float().cpu())

    handle = mapping.dit.condition_embedder.register_forward_pre_hook(record_timestep, with_kwargs=True)
    latent = clean_latent.to(device=device, dtype=torch.bfloat16)
    tau = torch.zeros(1, device=device, dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        raw_velocity = mapping(latent, tau)
        z4d = -raw_velocity
        loss = z4d.float().square().mean()
    loss.backward()
    handle.remove()
    gradient = next((parameter.grad for parameter in mapping.dit.parameters() if parameter.grad is not None), None)
    algebra = verify_flow_velocity_algebra(device)
    result = {
        "rgb_layout": ["batch", "physical_time", "channel", "height", "width"],
        "wan_input_layout": ["batch", "channel", "physical_time", "height", "width"],
        "rgb_shape": [1, 21, 3, 128, 128],
        "clean_latent_shape": list(clean_latent.shape),
        "expected_native_shape": [1, *WAN_LATENT_SHAPE],
        "vae_temporal_compression": "1+(T-1)//4",
        "vae_deterministic_mean_max_difference": deterministic_max,
        "external_flow_tau": tau.float().cpu().tolist(),
        "actual_wan_timestep": recorded_timesteps[0].tolist(),
        "wan_timestep_scale": mapping.timestep_scale,
        "raw_velocity_shape": list(raw_velocity.shape),
        "z4d_shape": list(z4d.shape),
        "z4d_equals_negative_raw_max_error": float((z4d + raw_velocity).abs().max()),
        "dit_gradient": gradient is not None and bool(torch.isfinite(gradient).all()),
        "empty_condition_shape": list(condition.shape),
        "flow_convention": algebra,
    }
    assert tuple(clean_latent.shape[1:]) == WAN_LATENT_SHAPE
    assert raw_velocity.shape == latent.shape == z4d.shape
    assert result["actual_wan_timestep"] == [0.0]
    assert result["z4d_equals_negative_raw_max_error"] == 0.0
    assert result["dit_gradient"]
    print(json.dumps(result, indent=2), flush=True)
    print("DENSE4D_WAN_AUDIT_OK", flush=True)


if __name__ == "__main__":
    main()
