"""Optimizer groups, warmup, and parameter accounting."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..models.worldbridge import DenseQueryWanModel

def parameter_groups(model: DenseQueryWanModel, config: dict[str, Any]) -> list[dict[str, Any]]:
    named = list(model.named_parameters())
    names_by_id = {id(parameter): name for name, parameter in named}
    rgb_prefixes = (
        "decoder.upsampler.source_rgb_encoder.",
        "decoder.upsampler.source_fusions.",
        "decoder.query_rgb_projection.",
    )
    separate_rgb = bool(config.get("source_rgb_separate_optimizer_group", False))
    rgb = [
        parameter for name, parameter in named
        if separate_rgb and parameter.requires_grad and name.startswith(rgb_prefixes)
    ]
    rgb_ids = {id(parameter) for parameter in rgb}
    backbone = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    adapter = [parameter for parameter in getattr(model.backbone, "adapter_parameters", [])
               if parameter.requires_grad]
    adapter_ids = {id(parameter) for parameter in adapter}
    wan = [parameter for parameter in backbone if id(parameter) not in adapter_ids]
    camera_ids = {id(parameter) for parameter in model.camera_head_parameters()}
    decoder_ids = {id(parameter) for parameter in model.decoder.parameters()}
    lost_camera = camera_ids - decoder_ids
    if lost_camera:
        raise RuntimeError(f"camera parameters outside the decoder: {len(lost_camera)}")
    decoder = [
        parameter for parameter in model.decoder.parameters()
        if parameter.requires_grad and id(parameter) not in rgb_ids and id(parameter) not in camera_ids
    ]
    groups = []
    if wan:
        groups.append({"params": wan, "lr": float(config.get("backbone_learning_rate", config["learning_rate"])),
                       "name": "wan_backbone"})
    if adapter:
        groups.append({"params": adapter, "lr": float(config.get("geometry_learning_rate", config["learning_rate"])),
                       "name": "geometry_adapter"})
    if decoder:
        groups.append({"params": decoder, "lr": float(config["learning_rate"]), "name": "dense_decoder"})
    if rgb:
        multiplier = float(config.get("source_rgb_learning_rate_multiplier", 1.0))
        if not np.isfinite(multiplier) or multiplier <= 0:
            raise ValueError("source RGB learning-rate multiplier must be finite and positive")
        rgb_lr = float(config["learning_rate"]) * multiplier
        decay = [parameter for parameter in rgb if parameter.ndim > 1]
        no_decay = [parameter for parameter in rgb if parameter.ndim <= 1]
        if decay:
            groups.append({
                "params": decay, "lr": rgb_lr,
                "weight_decay": float(config["weight_decay"]),
                "name": "source_rgb_decay",
            })
        if no_decay:
            groups.append({
                "params": no_decay, "lr": rgb_lr, "weight_decay": 0.0,
                "name": "source_rgb_no_decay",
            })
        grouped_rgb = {id(parameter) for group in groups[-2:] for parameter in group["params"]} \
            if decay and no_decay else {id(parameter) for parameter in (decay + no_decay)}
        if grouped_rgb != rgb_ids:
            missing = [names_by_id[value] for value in rgb_ids - grouped_rgb]
            raise RuntimeError(f"source RGB optimizer grouping lost parameters: {missing[:8]}")
    camera = [parameter for parameter in model.camera_head_parameters() if parameter.requires_grad]
    if camera:
        grouped = {id(parameter) for group in groups for parameter in group['params']}
        if grouped & camera_ids:
            raise RuntimeError("camera parameters leaked into another optimizer group")
        groups.append({'params': camera, 'lr': float(config['camera_supervision']['learning_rate']),
                       'name': 'camera_head'})
    if not groups:
        raise ValueError("model has no trainable parameters")
    return groups


def apply_fresh_group_warmup(
    optimizer: torch.optim.Optimizer,
    group_names: set[str],
    update_number: int,
    start_step: int,
    warmup_steps: int,
    max_scale: float = 1.0,
) -> float:
    """Ramp newly unfrozen groups without reducing the mature RGB groups."""
    if not 0.0 < max_scale <= 1.0:
        raise ValueError("fresh-group maximum LR scale must be in (0,1]")
    if warmup_steps <= 0:
        factor = max_scale
    else:
        progress = int(update_number) - int(start_step)
        if progress <= 0:
            raise ValueError("fresh-group warm-up requires an update after the phase start")
        factor = max_scale * min(1.0, progress / int(warmup_steps))
    found: set[str] = set()
    for group in optimizer.param_groups:
        name = str(group.get("name"))
        if name in group_names:
            group["lr"] = float(group["lr"]) * factor
            found.add(name)
    if found != group_names:
        raise RuntimeError(f"fresh optimizer groups missing: {sorted(group_names - found)}")
    return factor
