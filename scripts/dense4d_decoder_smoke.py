#!/usr/bin/env python3
"""CUDA forward/backward smoke for the dense query decoder without Wan weights."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile

import torch
from torch import nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from worldbridge.dense4d import (
    DenseQueryDecoder, DenseQueryWanModel, RotaryEmbedding2D, flatten_z4d,
    masked_pair_smooth_l1, unflatten_z4d,
)


class TinyWanFinalOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.dit = nn.Conv3d(16, 16, 1)

    def forward(self, clean):
        return -self.dit(clean)  # same negative-final-output gradient topology


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    decoder = DenseQueryDecoder(
        query_dim=256, embedding_dim=128, num_layers=2, num_heads=8,
        upsample_channels=(256, 128, 64, 32), coarse_diagnostic=True,
    ).to(device)
    model = DenseQueryWanModel(TinyWanFinalOutput().to(device), decoder)
    clean = torch.randn(1, 16, 6, 16, 16, device=device)
    source = torch.tensor([[0, 7, 20, 14]], device=device)
    target = torch.tensor([[0, 20, 0, 7]], device=device)
    prediction, z4d, output = model(clean, source, target)
    target_xyz = torch.randn_like(prediction)
    valid = torch.ones(1, 4, 128, 128, dtype=torch.bool, device=device)
    loss = masked_pair_smooth_l1(prediction, target_xyz, valid)
    loss.backward()
    memory, coordinates = flatten_z4d(z4d.detach())
    restored = unflatten_z4d(memory, 6, 16, 16)

    rope = RotaryEmbedding2D(32).to(device)
    broadcast = torch.randn(1, 1, 1, 256, 32, device=device)
    v, u = torch.meshgrid(torch.arange(16, device=device), torch.arange(16, device=device), indexing="ij")
    query_coordinates = torch.stack((u.reshape(-1), v.reshape(-1)), dim=-1)
    rotated = rope(broadcast, query_coordinates)
    spatial_rope_difference = float((rotated[..., 0, :] - rotated[..., -1, :]).abs().mean())
    query_forward = decoder.query_content(torch.tensor([[3]], device=device), torch.tensor([[18]], device=device))
    query_reverse = decoder.query_content(torch.tensor([[18]], device=device), torch.tensor([[3]], device=device))

    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "checkpoint.pt"
        torch.save(model.state_dict(), path)
        clone = DenseQueryWanModel(
            TinyWanFinalOutput().to(device),
            DenseQueryDecoder(query_dim=256, embedding_dim=128, num_layers=2, num_heads=8,
                              upsample_channels=(256, 128, 64, 32), coarse_diagnostic=True).to(device),
        )
        clone.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        checkpoint_ok = True

    result = {
        "z4d_shape": list(z4d.shape), "memory_shape": list(memory.shape),
        "memory_coordinates_shape": list(coordinates.shape),
        "memory_roundtrip_max": float((restored - z4d.detach()).abs().max()),
        "query_content_shape": list(query_forward.shape),
        "source_target_query_mean_abs_difference": float((query_forward - query_reverse).abs().mean()),
        "source_target_embeddings_independent": decoder.source_embedding.weight.data_ptr() != decoder.target_embedding.weight.data_ptr(),
        "spatial_rope_mean_abs_difference": spatial_rope_difference,
        "low_resolution_feature_shape": list(output.low_resolution_feature.shape),
        "coarse_shape": list(output.coarse_normalized_xyz.shape),
        "output_shape": list(prediction.shape), "loss": float(loss.detach()),
        "backbone_gradient": model.backbone.dit.weight.grad is not None,
        "decoder_gradient": decoder.upsampler.xyz.weight.grad is not None,
        "checkpoint_load_ok": checkpoint_ok,
    }
    assert result["z4d_shape"] == [1, 16, 6, 16, 16]
    assert result["memory_shape"] == [1, 1536, 16]
    assert result["memory_roundtrip_max"] == 0.0
    assert result["source_target_query_mean_abs_difference"] > 0
    assert result["spatial_rope_mean_abs_difference"] > 0
    assert result["output_shape"] == [1, 4, 3, 128, 128]
    assert result["backbone_gradient"] and result["decoder_gradient"] and checkpoint_ok
    print(json.dumps(result, indent=2), flush=True)
    print("DENSE4D_DECODER_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
