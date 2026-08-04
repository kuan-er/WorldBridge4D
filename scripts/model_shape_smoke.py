#!/usr/bin/env python3
"""Exact-shape and reverse-mode smoke test for both latent variants."""
from __future__ import annotations
import json, pathlib, sys
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from worldbridge.models import WorldLatentModel


def main():
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B,T,H,W = 1,21,128,128
    values = torch.randn(B,T,H,W,3,device=device)
    visible = torch.ones(B,T,H,W,dtype=torch.bool,device=device)
    valid = torch.ones_like(visible)
    p = torch.randn(B,T,3,H,W,device=device)
    for mode in ("compact", "full"):
        model = WorldLatentModel(mode).to(device)
        if mode == "compact":
            z = model(pointmaps=p, point_valid=valid, anchor_values=values,
                      anchor_visible=visible, anchor_valid=valid, source_length=T)
        else:
            h = torch.randn(B,T,H,W,32,device=device,requires_grad=True)
            z = model(trajectory_features=h, source_length=T)
        assert tuple(z.shape) == (B,16,6,16,16), tuple(z.shape)
        source = torch.tensor([[0., 11., 20.]],device=device)
        target = torch.tensor([[3., 10., 18.]],device=device)
        uv = torch.tensor([[[.2,.3],[.5,.6],[.8,.1]]],device=device,requires_grad=True)
        z2, pred = model(pointmaps=p, point_valid=valid, anchor_values=values,
                         anchor_visible=visible, anchor_valid=valid,
                         trajectory_features=(h if mode == "full" else None),
                         source_time=source, uv01=uv, target_time=target, source_length=T)
        loss = pred.square().mean() + z2.square().mean()
        loss.backward()
        assert torch.isfinite(loss).item()
        assert uv.grad is not None and torch.isfinite(uv.grad).all().item()
        print(json.dumps({"mode": mode, "device": str(device), "latent_shape": list(z.shape),
                          "parameters": sum(x.numel() for x in model.parameters()),
                          "loss": float(loss.detach()), "decoder_uv_grad_norm": float(uv.grad.norm())}))
    print("MODEL_SHAPE_SMOKE_OK")


if __name__ == "__main__":
    main()
