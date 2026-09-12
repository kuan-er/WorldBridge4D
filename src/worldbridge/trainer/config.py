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
    master_precision = str(config.get("fsdp_master_precision", "model"))
    if master_precision not in {"model", "fp32"}:
        raise ValueError("fsdp_master_precision must be model or fp32")
    if master_precision == "fp32" and (config.get("precision") != "bf16" or mode != "decoder_only"):
        raise ValueError("FP32 master trial requires BF16 compute and decoder_only")
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
    if bool(config.get("cycle_reprojection_normalize_to_xyz", False)):
        if float(config.get("cycle_reprojection_normalization_epsilon", 1e-6)) <= 0.0:
            raise ValueError("cycle_reprojection_normalization_epsilon must be positive")
        if float(config.get("cycle_reprojection_normalization_max_scale", 1000.0)) <= 0.0:
            raise ValueError("cycle_reprojection_normalization_max_scale must be positive")
    boundary = config.get('boundary_supervision')
    if boundary is not None:
        if (not isinstance(boundary, dict)
                or set(boundary) != {'multiplier', 'radius_px', 'depth_relative_jump'}
                or float(boundary['multiplier']) not in (1.0, 2.0)
                or boundary['radius_px'] != 2
                or float(boundary['depth_relative_jump']) != 0.05):
            raise ValueError('boundary control requires multiplier1or2 radius2 depth_jump0.05')
        if (mode != 'decoder_only' or master_precision != 'fp32'
                or config.get('precision') != 'bf16' or not cycle_enabled
                or set(cycle_names) != supported_cycle_names
                or float(config.get('cycle_reprojection_weight', 0.0)) != 0.0):
            raise ValueError('boundary control requires FP32/BF16 decoder and all-dataset cycle0 paths')
    contrast_weight = float(config.get('source_edge_contrast_weight', 0.0))
    if contrast_weight not in (0.0, 0.01, 0.1):
        raise ValueError('audited source edge contrast weight must be 0, 0.01 or 0.1')
    if contrast_weight > 0 and (boundary is None or float(boundary['multiplier']) != 2.0):
        raise ValueError('source edge contrast control requires boundary2x')
    native_k15 = bool(config.get('native_kubric512_b1_a4_k15', False))
    native_k9 = bool(config.get('native_kubric512_b1_a4_k9', False))
    native_k5 = bool(config.get('native_kubric512_b1_a4_k5', False))
    if sum((native_k15, native_k9, native_k5)) > 1:
        raise ValueError('select only one native512 target profile')
    native_512 = native_k15 or native_k9 or native_k5
    native_long = bool(config.get('native_kubric512_k5_10k', False))
    native_mix_trial = bool(config.get('native_kubric512_k9_mix_trial', False))
    native_mix_170k = bool(config.get('native_kubric512_k9_mix_170k', False))
    # K11 is an explicit target-count override on the same native512 K9 route.
    native_k11 = bool(config.get('native_kubric512_k11_trial', False))
    if native_k11 and not (native_k9 and native_mix_trial and native_mix_170k):
        raise ValueError('K11 capacity trial requires the native K9 mixture170k base route')
    native_k5_mix = bool(config.get('native_kubric512_k5_mix_resume', False))
    if native_k5_mix and (native_k11 or not (native_k9 and native_mix_trial and native_mix_170k)):
        raise ValueError('K5 mixture resume requires the native K9 mixture170k base route without K11')
    native_k3_mix = bool(config.get('native_kubric512_k3_mix_resume', False))
    if native_k3_mix and (native_k11 or native_k5_mix or not (native_k9 and native_mix_trial and native_mix_170k)):
        raise ValueError('K3 mixture resume requires the native K9 mixture170k base route without K11/K5 override')
    native_k3_200k = bool(config.get('native_kubric512_k3_mix_200k', False))
    if native_k3_200k and not native_k3_mix:
        raise ValueError('native200k extension requires the audited K3 mixture resume route')
    if native_mix_170k and not native_mix_trial:
        raise ValueError('native K9 170k extension requires the explicit mixture profile')
    if native_long and not native_k5:
        raise ValueError('native10k continuation requires the audited K5 profile')
    if native_mix_trial and (not native_k9 or native_long):
        raise ValueError('native mixture trial requires K9, not the K5-10k profile')
    from .schedulers import validated_mix_counts
    mix = validated_mix_counts(config.get('dataset_mix_counts'))
    if native_mix_trial:
        if mix != {'kubric': 10, 'pointodyssey': 5, 'dynamic_replica': 5}:
            raise ValueError('native K9 mixture trial requires exact50/25/25')
    elif mix != validated_mix_counts():
        raise ValueError('non-legacy mixture requires an explicit native K9 mixture trial')
    if native_512:
        if (mode != 'decoder_only' or master_precision != 'fp32' or not cycle_enabled
                or set(cycle_names) != supported_cycle_names or boundary is not None
                or contrast_weight != 0 or float(config.get('cycle_reprojection_weight', -1)) != 0
                or not config.get('pre_attention_rgb_query') or config.get('query_grid_size') != 32
                or any(config.get(k, False) for k in ('cycle_b2_a2_k19', 'cycle_b2_a2_k15', 'xyz_b2_a2_k15'))):
            raise ValueError('native512 B1/A4 requires unchanged FP32 decoder XYZ/cycle0/RGB1x control')
        gt_values = config.get('datasets', {}).get('kubric', {})
        demand_gt = gt_values.get('native_geometry_mode', 'staged') == 'verified_cache_or_raw'
        if demand_gt and not (native_long or native_mix_trial):
            raise ValueError('native on-demand GT requires explicit continuation')
        if native_mix_trial:
            selected = int(config.get('selected_checkpoint_step', -1))
            endpoint = 200000 if native_k3_200k else (170000 if native_mix_170k else selected + 2000)
            if (config.get('native_capacity_test_only') or selected < 150500
                    or selected >= endpoint or config.get('max_steps') != endpoint
                    or not (bool(gt_values.get('native_geometry_full_corpus')) ^ demand_gt)):
                raise ValueError('native K9 mixture requires its declared full-resume budget and verified GT reader')
            restart = config.get('lr_restart', {})
            if (restart.get('start_step') != 150000 or restart.get('warmup_steps') != 500
                    or restart.get('end_step') != (endpoint if native_mix_170k else 160010) or restart.get('schedule') != 'warmup_hold'
                    or config['max_steps'] > restart['end_step']):
                raise ValueError('native K9 mix trial preserves existing150k warmup and3e-6 hold horizon')
        elif native_long:
            if (config.get('native_capacity_test_only')
                    or int(config.get('max_steps', 0)) != 160010
                    or int(config.get('selected_checkpoint_step', -1)) != 150010
                    or not (bool(gt_values.get('native_geometry_full_corpus')) ^ demand_gt)):
                raise ValueError('native10k requires full or explicit native on-demand GT and150010->160010 continuation')
            restart = config.get('lr_restart', {})
            if (restart.get('start_step') != 150000 or restart.get('warmup_steps') != 500
                    or restart.get('end_step') != 160010 or restart.get('schedule') != 'warmup_hold'):
                raise ValueError('native10k preserves150k LR warmup/hold without another restart')
        elif (not config.get('native_capacity_test_only')
                or not 150000 < int(config.get('max_steps', 0)) <= 150010
                or int(config.get('selected_checkpoint_step', -1)) != 150000):
            raise ValueError('native512 requires bounded capacity or explicit audited10k continuation')
        rates = config.get('lr_restart', {}).get('group_learning_rates', {})
        if set(rates) != {'dense_decoder', 'source_rgb_decay', 'source_rgb_no_decay'} or any(float(v) != 3e-6 for v in rates.values()):
            raise ValueError('native512 comparison retains absolute RGB1x rates3e-6')
    accumulation = int(config.get("gradient_accumulation", 0))
    microbatch = int(config.get("microbatch_per_gpu", 0))
    cycle_b2_k19 = bool(config.get("cycle_b2_a2_k19", False))
    cycle_b2_k15 = bool(config.get("cycle_b2_a2_k15", False))
    xyz_b2_k15 = bool(config.get("xyz_b2_a2_k15", False))
    if xyz_b2_k15 and (cycle_enabled or cycle_b2_k19 or cycle_b2_k15):
        raise ValueError("XYZ-only B2/A2/K15 requires cycle disabled and no cycle profile")
    if cycle_b2_k19 and cycle_b2_k15:
        raise ValueError("select only one B2/A2 cycle target profile")
    if (cycle_b2_k19 or cycle_b2_k15) and not cycle_enabled:
        raise ValueError("B2/A2 cycle profiles require the cycle objective")
    allowed_batching = {(4, 1)} if native_512 else ({(2, 2)} if cycle_b2_k19 or cycle_b2_k15 else ({(4, 1), (4, 2)} if cycle_enabled else {(2, 2)}))
    if (accumulation, microbatch) not in allowed_batching:
        expected = " or ".join(
            f"gradient_accumulation={accum} and microbatch_per_gpu={micro}"
            for accum, micro in sorted(allowed_batching)
        )
        raise ValueError(f"training requires {expected}")
    required_targets = 3 if native_k3_mix else (11 if native_k11 else (5 if native_k5 or native_k5_mix else (9 if native_k9 else (15 if native_k15 or cycle_b2_k15 or xyz_b2_k15 else (13 if cycle_enabled and not cycle_b2_k19 else 19)))))
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
    for key in ("trace_first_updates", "cuda_empty_cache_every_steps"):
        if int(config.get(key, 0)) < 0:
            raise ValueError(f"{key} must be non-negative")
    stall = float(config.get("runtime_stall_traceback_seconds", 0))
    timeout = float(config.get("distributed_timeout_seconds", 86400))
    if not np.isfinite(stall) or stall < 0 or not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("invalid runtime timeout/traceback interval")
    restart = config.get("lr_restart")
    if restart is not None:
        if mode != "decoder_only":
            raise ValueError("LR restart currently requires decoder_only")
        start = int(restart["start_step"])
        end = int(restart["end_step"])
        warmup = int(restart["warmup_steps"])
        if not 0 <= start < start + warmup < end or int(config["max_steps"]) > end:
            raise ValueError("invalid LR restart phase interval")
        rates = restart["group_learning_rates"]
        if set(rates) != {"dense_decoder", "source_rgb_decay", "source_rgb_no_decay"}:
            raise ValueError("LR restart requires explicit rates for all decoder/RGB groups")
        if any(not np.isfinite(float(rate)) or float(rate) <= 0 for rate in rates.values()):
            raise ValueError("LR restart rates must be finite and positive")
        if config.get("schedule_extension_start_step") is not None:
            raise ValueError("LR restart must not also enable legacy cosine extension")
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
