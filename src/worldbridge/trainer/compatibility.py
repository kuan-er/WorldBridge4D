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


def compare_training_checkpoints(
    reference_path: str | Path,
    candidate_path: str | Path,
) -> dict[str, Any]:
    """Require exact recursive equality of two full training checkpoints."""
    reference = torch.load(
        Path(reference_path).resolve(), map_location="cpu", mmap=True, weights_only=True,
    )
    candidate = torch.load(
        Path(candidate_path).resolve(), map_location="cpu", mmap=True, weights_only=True,
    )
    tensor_count = 0
    tensor_elements = 0

    def compare(left: Any, right: Any, path: str) -> None:
        nonlocal tensor_count, tensor_elements
        if type(left) is not type(right):
            raise AssertionError(
                f"{path}: type differs: {type(left).__name__} != {type(right).__name__}"
            )
        if isinstance(left, torch.Tensor):
            tensor_count += 1
            tensor_elements += left.numel()
            if left.shape != right.shape or left.dtype != right.dtype:
                raise AssertionError(
                    f"{path}: tensor metadata differs: "
                    f"{left.shape}/{left.dtype} != {right.shape}/{right.dtype}"
                )
            if not torch.equal(left, right):
                detail = ""
                if left.is_floating_point() and left.numel():
                    maximum = (left.float() - right.float()).abs().max().item()
                    detail = f", max_abs={maximum}"
                raise AssertionError(f"{path}: tensor values differ{detail}")
            return
        if isinstance(left, dict):
            if list(left) != list(right):
                raise AssertionError(f"{path}: mapping keys/order differ")
            for key in left:
                compare(left[key], right[key], f"{path}.{key}")
            return
        if isinstance(left, (list, tuple)):
            if len(left) != len(right):
                raise AssertionError(f"{path}: sequence lengths differ")
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                compare(left_item, right_item, f"{path}[{index}]")
            return
        if left != right:
            raise AssertionError(f"{path}: values differ: {left!r} != {right!r}")

    compare(reference, candidate, "checkpoint")
    result = {
        "global_step": int(reference["training_state"]["global_step"]),
        "model_state_entries": len(reference["model"]),
        "compared_tensors": tensor_count,
        "compared_tensor_elements": tensor_elements,
    }
    del reference, candidate
    gc.collect()
    return result
