"""Fail-closed validation of the single supported H033 training profile.

One route is supported and validated positively: full-model native512 training
(Kubric512 + DR512 + PO256), source-frame coordinates, BF16 compute with FP32
master weights, K10 camera supervision, and the B1/A4 batching that goes with it.
Everything else is rejected, including the historical decoder-only, cycle,
boundary, schedule-extension and staged-input profiles that H033 deleted.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..data.constants import DATASET_NAMES
from .schedulers import validated_mix_counts

ARCHITECTURE = {
    "image_size": 256, "clip_length": 21, "latent_spatial_size": 32,
    "query_dim": 1536, "embedding_dim": 768, "num_cross_attn_layers": 5,
    "num_heads": 12, "geometry_dim": 512, "geometry_spatial_size": 32,
    "motion_slots": 8,
}
REMOVED_FEATURES = (
    "cycle_reprojection_", "boundary_supervision", "source_edge_contrast_weight",
    "schedule_extension_start_step", "schedule_extension_horizon_steps",
    "staged_inputs", "pipeline_wait_timeout_seconds", "geometry_replay",
    "lazy_vae_cache", "lazy_vae_pipeline",
)
LEGACY_PROFILES = (
    "native_capacity_test_only", "native_kubric512_b1_a4_k15",
    "native_kubric512_b1_a4_k9", "native_kubric512_b1_a4_k5",
    "native_kubric512_k5_10k", "native_kubric512_k9_mix_trial",
    "native_kubric512_k9_mix_170k", "native_kubric512_k11_trial",
    "native_kubric512_k5_mix_resume", "native_kubric512_k3_mix_resume",
    "native_kubric512_k3_mix_200k", "native_kubric512_b2_a2_k9_mix_200k",
    "native_kubric512_b2_a2_k5_mix_200k", "native_kubric512_b1_a4_k9_mix_200k",
    "native_kubric512_b1_a4_k11_mix_200k", "cycle_b2_a2_k19", "cycle_b2_a2_k15",
    "cycle_b2_a2_k9", "xyz_b2_a2_k15",
)
EXPECTED_MIXTURE = {"kubric": 10, "pointodyssey": 5, "dynamic_replica": 5}
CAMERA_LOSS_WEIGHTS = {'diagonal_xyz', 'offdiagonal_xyz', 'diagonal_ray', 'ray_field',
                       'front', 'pose_rotation', 'pose_translation'}
CAMERA_KEYS = {'pose_hidden', 'ray_hidden', 'seed', 'learning_rate', 'warmup_steps',
               'phase_start_step', 'loss_weights', 'intrinsics_mode', 'pose_translation_scale'}


def validate_config(config: dict[str, Any], world: int) -> None:
    _reject_deleted_features(config)
    mismatches = {key: (config.get(key), value) for key, value in ARCHITECTURE.items()
                  if config.get(key) != value}
    if mismatches:
        raise ValueError(f"256/200M frozen configuration mismatch: {mismatches}")
    _validate_source_rgb(config)
    _validate_readout(config)
    _validate_precision(config, world)
    _validate_profile(config)
    camera = _validate_camera(config)
    _validate_datasets(config)
    _validate_lr_restart(config, camera)
    _validate_runtime_controls(config)


def _reject_deleted_features(config: dict[str, Any]) -> None:
    removed = sorted(key for key in config
                     if any(key.startswith(prefix) for prefix in REMOVED_FEATURES)
                     or key in REMOVED_FEATURES)
    if removed:
        raise ValueError(f"removed training features are no longer configurable: {removed}")
    legacy = sorted(key for key in LEGACY_PROFILES if config.get(key))
    if legacy:
        raise ValueError(f"legacy training profiles are no longer supported: {legacy}")


def _validate_source_rgb(config: dict[str, Any]) -> None:
    if not bool(config.get("source_rgb_pyramid", False)):
        raise ValueError("the supported profile requires the source RGB pyramid")
    if list(config.get("source_rgb_channels", [])) != [32, 64, 128]:
        raise ValueError("source RGB pyramid fixes channels to [32,64,128]")
    expected_scales = ([32, 64, 128, 256] if bool(config.get("source_rgb_fusion_32", False))
                       else [64, 128, 256])
    if list(config.get("source_rgb_fusion_scales", [])) != expected_scales:
        raise ValueError(f"source RGB pyramid fixes fusion scales to {expected_scales}")
    if not bool(config.get("source_rgb_zero_init", False)):
        raise ValueError("source RGB pyramid requires zero-initialized residual gates")
    if not config.get("source_rgb_cache_root"):
        raise ValueError("source RGB pyramid requires source_rgb_cache_root")
    if int(config.get("source_rgb_cache_max_open_shards", 0)) < 1:
        raise ValueError("source RGB mmap cache size must be positive")
    if not bool(config.get("pre_attention_rgb_query", False)):
        raise ValueError("the supported profile requires the pre-attention RGB query")


def _validate_readout(config: dict[str, Any]) -> None:
    wan_num_layers = int(config.get("wan_num_layers", 30))
    expected_layers = [13, 14, 15, wan_num_layers - 1]
    if wan_num_layers < 16 or list(config.get("wan_hidden_layers", [])) != expected_layers:
        raise ValueError(f"wan_hidden_layers must be exactly {expected_layers}")
    logits = np.asarray(config.get("layer_gate_initial_logits"), dtype=np.float64)
    expected_logits = np.array([0.0, 0.0, 0.0, -1.0986122887])
    if logits.shape != (4,) or not np.allclose(logits, expected_logits, atol=1e-10):
        raise ValueError(f"incorrect readout initialization: {logits}")
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    if not np.allclose(weights, [0.3, 0.3, 0.3, 0.1], atol=1e-8):
        raise ValueError(f"incorrect initial layer weights: {weights}")


def _validate_precision(config: dict[str, Any], world: int) -> None:
    if world != 2:
        raise ValueError(f"production training requires exactly 2 ranks; got {world}")
    master_precision = str(config.get("fsdp_master_precision", "model"))
    if master_precision not in {"model", "fp32"}:
        raise ValueError("fsdp_master_precision must be model or fp32")
    if master_precision == "fp32" and config.get("precision") != "bf16":
        raise ValueError("FP32 master requires BF16 compute")


def _validate_profile(config: dict[str, Any]) -> None:
    if not bool(config.get("native_kubric512_full", False)):
        raise ValueError("the supported profile is native_kubric512_full")
    if not bool(config.get("native_dr512", False)):
        raise ValueError("the supported profile requires native_dr512")
    if config.get("backbone_readout") != "wan_hidden_structured":
        raise ValueError("the supported profile requires the structured Wan readout")
    if config.get("coordinate_frame") != "source":
        raise ValueError("the supported profile uses source-frame coordinates")
    if str(config.get("trainable_mode", "full")) != "full":
        raise ValueError("the supported profile trains the full model")
    if config.get("full_mode_unfreeze_resume"):
        raise ValueError("full-mode unfreeze resume is not part of the supported profile")
    if config.get("precision") != "bf16":
        raise ValueError("the supported profile requires bf16 compute")
    if config.get("fsdp_master_precision") != "fp32":
        raise ValueError("the supported profile requires fp32 master precision")
    if (int(config.get("gradient_accumulation", 0)), int(config.get("microbatch_per_gpu", 0))) != (4, 1):
        raise ValueError("the supported profile requires gradient_accumulation=4 and microbatch_per_gpu=1")
    if validated_mix_counts(config.get("dataset_mix_counts")) != EXPECTED_MIXTURE:
        raise ValueError("the supported profile requires the exact 50/25/25 mixture")


def _validate_camera(config: dict[str, Any]) -> dict[str, Any] | None:
    camera = config.get('camera_supervision')
    camera_k10 = bool(config.get('camera_k10', False))
    if camera is None or not camera_k10:
        raise ValueError('the supported profile requires camera supervision and camera_k10')
    if int(config.get('targets_per_source', 0)) != 10:
        raise ValueError('camera K10 requires exactly 10 targets per source')
    if not isinstance(camera, dict) or set(camera) != CAMERA_KEYS:
        raise ValueError('camera supervision requires the explicit decoder-native contract')
    if (camera['intrinsics_mode'] != 'per_frame_source_independent'
            or int(camera['pose_hidden']) < 8 or int(camera['ray_hidden']) < 8
            or int(camera['warmup_steps']) < 1 or int(camera['phase_start_step']) < 0
            or not np.isfinite(camera['learning_rate']) or camera['learning_rate'] <= 0):
        raise ValueError('invalid camera readout architecture or learning rate')
    scales = camera['pose_translation_scale']
    values = list(scales.values()) if isinstance(scales, dict) else [scales]
    if ((isinstance(scales, dict) and set(scales) != set(DATASET_NAMES))
            or any(not np.isfinite(float(value)) or float(value) <= 0 for value in values)):
        raise ValueError('camera pose translation scale must be positive for every dataset')
    weights = camera['loss_weights']
    if (set(weights) != CAMERA_LOSS_WEIGHTS
            or any(not np.isfinite(value) or value <= 0 for value in weights.values())):
        raise ValueError('camera loss weights must be explicit finite positive values')
    return camera


def _validate_datasets(config: dict[str, Any]) -> None:
    kubric = config.get('datasets', {}).get('kubric', {})
    if kubric.get('native_geometry_mode', 'staged') != 'verified_cache_or_raw':
        raise ValueError('the supported profile requires the verified native GT demand reader')
    dynamic_replica = config.get('datasets', {}).get('dynamic_replica', {})
    for key in ('native_manifest', 'native_rgb_root', 'native_latent_root'):
        if not dynamic_replica.get(key):
            raise ValueError(f'native DR512 requires dynamic_replica {key}')
    prefetch_depth = int(config.get("geometry_prefetch_depth", 2))
    prefetch_workers = int(config.get("geometry_prefetch_workers", 2))
    if not 1 <= prefetch_depth <= 16:
        raise ValueError("geometry_prefetch_depth must be in [1,16]")
    if not 1 <= prefetch_workers <= 32:
        raise ValueError("geometry_prefetch_workers must be in [1,32]")
    sample_cache_size = int(kubric.get("geometry_sample_cache_size", 16))
    if not 1 <= sample_cache_size <= 256:
        raise ValueError("geometry_sample_cache_size must be in [1,256]")
    max_open_shards = kubric.get("geometry_mmap_max_open_shards")
    if max_open_shards is not None and not 1 <= int(max_open_shards) <= 4096:
        raise ValueError("geometry_mmap_max_open_shards must be in [1,4096]")


def _validate_lr_restart(config: dict[str, Any], camera: dict[str, Any] | None) -> None:
    restart = config.get("lr_restart")
    if restart is None:
        raise ValueError("the supported profile requires the explicit lr_restart phase")
    start = int(restart["start_step"])
    end = int(restart["end_step"])
    warmup = int(restart["warmup_steps"])
    if not 0 <= start < start + warmup < end or int(config["max_steps"]) > end:
        raise ValueError("invalid LR restart phase interval")
    expected_groups = {"wan_backbone", "geometry_adapter", "dense_decoder",
                       "source_rgb_decay", "source_rgb_no_decay", "camera_head"}
    if camera is not None and restart["group_learning_rates"].get('camera_head') != camera['learning_rate']:
        raise ValueError('camera group schedule must match its learning rate')
    if set(restart["group_learning_rates"]) != expected_groups:
        raise ValueError("LR restart requires explicit rates for all six groups")
    if any(not np.isfinite(float(rate)) or float(rate) <= 0
           for rate in restart["group_learning_rates"].values()):
        raise ValueError("LR restart rates must be finite and positive")


def _validate_runtime_controls(config: dict[str, Any]) -> None:
    if int(config.get("diagnostic_every_steps", 20)) < 1:
        raise ValueError("diagnostic_every_steps must be positive")
    for key in ("trace_first_updates", "cuda_empty_cache_every_steps"):
        if int(config.get(key, 0)) < 0:
            raise ValueError(f"{key} must be non-negative")
    stall = float(config.get("runtime_stall_traceback_seconds", 0))
    timeout = float(config.get("distributed_timeout_seconds", 86400))
    if not np.isfinite(stall) or stall < 0 or not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("invalid runtime timeout/traceback interval")
