"""Fail-closed validation of the registered production protocol."""
from __future__ import annotations

from typing import Any

import numpy as np

def validate_config(config: dict[str, Any], world: int) -> None:
    expected = {
        "image_size": 256, "clip_length": 21, "latent_spatial_size": 32,
        "query_dim": 1536, "embedding_dim": 768, "num_cross_attn_layers": 5,
        "num_heads": 12, "geometry_dim": 512, "geometry_spatial_size": 32,
        "motion_slots": 8,
    }
    mismatches = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if mismatches:
        raise ValueError(f"256/200M frozen configuration mismatch: {mismatches}")
    if bool(config.get("source_rgb_pyramid", False)):
        if list(config.get("source_rgb_channels", [])) != [32, 64, 128]:
            raise ValueError("source RGB pyramid fixes channels to [32,64,128]")
        expected_rgb_scales = (
            [32, 64, 128, 256]
            if bool(config.get("source_rgb_fusion_32", False))
            else [64, 128, 256]
        )
        if list(config.get("source_rgb_fusion_scales", [])) != expected_rgb_scales:
            raise ValueError(
                f"source RGB pyramid fixes fusion scales to {expected_rgb_scales}"
            )
        if not bool(config.get("source_rgb_zero_init", False)):
            raise ValueError("source RGB pyramid requires zero-initialized residual gates")
        if not config.get("source_rgb_cache_root"):
            raise ValueError("source RGB pyramid requires source_rgb_cache_root")
        if int(config.get("source_rgb_cache_max_open_shards", 0)) < 1:
            raise ValueError("source RGB mmap cache size must be positive")
    wan_num_layers = int(config.get("wan_num_layers", 30))
    if wan_num_layers < 16:
        raise ValueError("wan_num_layers must include readout blocks 13,14,15")
    expected_hidden_layers = [13, 14, 15, wan_num_layers - 1]
    if list(config.get("wan_hidden_layers", [])) != expected_hidden_layers:
        raise ValueError(
            f"wan_hidden_layers must include the model's final block: {expected_hidden_layers}"
        )
    mode = str(config.get("trainable_mode", "full"))
    allowed_modes = {"full", "source_rgb_plus_wan_decoder", "decoder_only"}
    if mode not in allowed_modes:
        raise ValueError(
            "three-dataset training supports full and audited decoder phases"
        )
    if mode in {"source_rgb_plus_wan_decoder", "decoder_only"}:
        if not bool(config.get("source_rgb_pyramid", False)):
            raise ValueError("source-RGB trainable modes require the RGB pyramid")
        if not bool(config.get("source_rgb_separate_optimizer_group", False)):
            raise ValueError("source-RGB trainable modes require separate optimizer groups")
        multiplier = float(config.get("source_rgb_learning_rate_multiplier", 0.0))
        if multiplier != 10.0:
            raise ValueError("production source-RGB LR multiplier must be exactly 10")
    if bool(config.get("pre_attention_rgb_query", False)):
        if mode != "decoder_only":
            raise ValueError("pre-attention RGB query requires decoder_only training")
        if not bool(config.get("source_rgb_pyramid", False)):
            raise ValueError("pre-attention RGB query requires source RGB")
    if mode in {"source_rgb_plus_wan_decoder", "decoder_only"}:
        if int(config.get("joint_fresh_group_warmup_steps", 0)) < 0:
            raise ValueError("joint fresh-group warm-up steps must be non-negative")
        max_scale = float(config.get("joint_fresh_group_max_lr_scale", 1.0))
        if not 0.0 < max_scale <= 1.0:
            raise ValueError("joint fresh-group maximum LR scale must be in (0,1]")
    logits = np.asarray(config.get("layer_gate_initial_logits"), dtype=np.float64)
    expected_logits = np.array([0.0, 0.0, 0.0, -1.0986122887])
    if logits.shape != (4,) or not np.allclose(logits, expected_logits, atol=1e-10):
        raise ValueError(f"incorrect readout initialization: {logits}")
    weights = np.exp(logits - logits.max()); weights /= weights.sum()
    if not np.allclose(weights, [0.3, 0.3, 0.3, 0.1], atol=1e-8):
        raise ValueError(f"incorrect initial layer weights: {weights}")
    if world != 2:
        raise ValueError(f"production training requires exactly 2 ranks; got {world}")
    cycle_enabled = bool(config.get("cycle_reprojection_enabled", False))
    cycle_names = tuple(str(name) for name in config.get(
        "cycle_reprojection_datasets", ["kubric"],
    ))
    supported_cycle_names = {"kubric", "pointodyssey", "dynamic_replica"}
    if cycle_enabled and (
        not cycle_names or set(cycle_names) - supported_cycle_names
        or len(set(cycle_names)) != len(cycle_names)
    ):
        raise ValueError(
            "cycle_reprojection_datasets must be a non-empty subset of "
            "[kubric, pointodyssey, dynamic_replica]"
        )
    if int(config.get("cycle_reprojection_pixel_stride", 1)) < 1:
        raise ValueError("cycle_reprojection_pixel_stride must be positive")
    if float(config.get("cycle_reprojection_weight", 0.0)) < 0.0:
        raise ValueError("cycle_reprojection_weight must be non-negative")
    if float(config.get("cycle_reprojection_huber_delta", 0.01)) <= 0.0:
        raise ValueError("cycle_reprojection_huber_delta must be positive")
    accumulation = int(config.get("gradient_accumulation", 0))
    microbatch = int(config.get("microbatch_per_gpu", 0))
    allowed_batching = {(4, 1), (4, 2)} if cycle_enabled else {(2, 2)}
    if (accumulation, microbatch) not in allowed_batching:
        expected = " or ".join(
            f"gradient_accumulation={accum} and microbatch_per_gpu={micro}"
            for accum, micro in sorted(allowed_batching)
        )
        raise ValueError(f"training requires {expected}")
    required_targets = 13 if cycle_enabled else 19
    if int(config["targets_per_source"]) != required_targets:
        raise ValueError(f"training requires targets_per_source={required_targets}")
    prefetch_depth = int(config.get("geometry_prefetch_depth", 2))
    prefetch_workers = int(config.get(
        "geometry_prefetch_workers", min(4, accumulation * microbatch * 2),
    ))
    if not 1 <= prefetch_depth <= 16:
        raise ValueError("geometry_prefetch_depth must be in [1,16]")
    if not 1 <= prefetch_workers <= 32:
        raise ValueError("geometry_prefetch_workers must be in [1,32]")
    kubric = config.get("datasets", {}).get("kubric", {})
    sample_cache_size = int(kubric.get("geometry_sample_cache_size", 16))
    if not 1 <= sample_cache_size <= 256:
        raise ValueError("geometry_sample_cache_size must be in [1,256]")
    max_open_shards = kubric.get("geometry_mmap_max_open_shards")
    if max_open_shards is not None and not 1 <= int(max_open_shards) <= 4096:
        raise ValueError("geometry_mmap_max_open_shards must be in [1,4096]")
    diagnostic_every = int(config.get("diagnostic_every_steps", 20))
    if diagnostic_every < 1:
        raise ValueError("diagnostic_every_steps must be positive")
    extension_start = config.get("schedule_extension_start_step")
    extension_horizon = config.get("schedule_extension_horizon_steps")
    if (extension_start is None) != (extension_horizon is None):
        raise ValueError("both cosine schedule extension fields must be configured")
    if extension_start is not None:
        warmup = int(config["warmup_steps"])
        original_horizon = int(config["schedule_horizon_steps"])
        extension_start = int(extension_start)
        extension_horizon = int(extension_horizon)
        if not warmup < extension_start < original_horizon < extension_horizon:
            raise ValueError(
                "schedule extension must satisfy warmup < start < original horizon < extended horizon"
            )
        max_steps = int(config.get("max_steps", extension_horizon))
        if not extension_start < max_steps <= extension_horizon:
            raise ValueError(
                "max_steps must be after the schedule extension start and no later "
                "than its horizon"
            )
