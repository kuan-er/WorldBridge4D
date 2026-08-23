"""Heavyweight compatibility checks for immutable production checkpoints."""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import torch
import yaml

from ..models.factory import build_real_model


def validate_checkpoint_model(
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Construct the registered architecture and strict-load a full checkpoint."""
    config_path = Path(config_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    config = yaml.safe_load(config_path.read_text())
    payload = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=True,
    )
    if "model" not in payload or "training_state" not in payload:
        raise ValueError("checkpoint lacks model or training_state")
    model = build_real_model(config, torch.device(device), load_wan_pretrained=False)
    incompatible = model.load_state_dict(payload["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict load returned incompatible keys: {incompatible}")
    result = {
        "global_step": int(payload["training_state"]["global_step"]),
        "model_state_entries": len(payload["model"]),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    del model, payload
    gc.collect()
    return result
