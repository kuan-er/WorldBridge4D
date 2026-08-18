from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import json
import re

import numpy as np
import pytest
import torch
from torch import nn

from worldbridge.dense4d import DenseQueryDecoder, WanHiddenGeometryBackbone, masked_pair_smooth_l1
from worldbridge.training256 import (
    CachedExternalDataset, KubricGeometryMmapStore, LazyLatentCache, MOViF256Dataset,
    apply_cosine_schedule, dataset_for_step, deterministic_dataset_schedule,
    deterministic_sample_plan, extended_cosine_learning_rate_factor,
    sample_eligible_targets, source_with_eligible_targets, training_diagnostic_due,
)
from worldbridge.wan import WAN_LATENT_SHAPE_256, WanDiTMapping
from worldbridge.text_conditions import load_inference_text_condition
from scripts.create_wan_text_conditions import (
    PROMPTS, TASK_INSTRUCTION, completed_cache, native_encoder,
)


class TinyExplicitConditionMapping(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(16, 32)
        self.conditions = []
        self.dit = nn.Module()
        self.dit.config = SimpleNamespace(
            num_attention_heads=2, attention_head_dim=16, patch_size=(1, 2, 2),
        )
        self.dit.norm_out = nn.LayerNorm(32)
        self.dit.proj_out = nn.Linear(32, 16)
        self.dit.scale_shift_table = nn.Parameter(torch.randn(1, 2, 32))

    def forward_hidden_layers(self, latent, tau, layers, condition=None):
        self.conditions.append(condition)
        pooled = torch.nn.functional.avg_pool3d(latent, (1, 2, 2), (1, 2, 2))
        tokens = pooled.permute(0, 2, 3, 4, 1).reshape(latent.shape[0], 6 * 16 * 16, 16)
        value = self.projection(tokens)
        return tuple(value * (layer + 1) for layer in layers), (6, 16, 16)


def test_native_wan_text_encoder_skips_official_package_init(tmp_path):
    package = tmp_path / "wan"
    modules = package / "modules"
    modules.mkdir(parents=True)
    (package / "__init__.py").write_text("raise RuntimeError('must not import wan.__init__')\n")
    (modules / "__init__.py").write_text("raise RuntimeError('must not import modules.__init__')\n")
    (modules / "tokenizers.py").write_text("class HuggingfaceTokenizer: pass\n")
    (modules / "t5.py").write_text(
        "from .tokenizers import HuggingfaceTokenizer\n"
        "class T5EncoderModel: pass\n"
    )
    encoder, layout = native_encoder(tmp_path)
    assert encoder.__name__ == "T5EncoderModel"
    assert layout == "official_wan_package"


def test_completed_text_condition_cache_rejects_partial_and_accepts_verified(tmp_path):
    import hashlib

    assert completed_cache(tmp_path) is None
    summary = {}
    for name, prompt in PROMPTS.items():
        path = tmp_path / f"{name}.pt"
        path.write_bytes(name.encode())
        summary[name] = {
            "dataset": name, "prompt": prompt, "shape": [1, 512, 4096],
            "condition_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (tmp_path / "metadata.json").write_text(json.dumps(summary))
    assert completed_cache(tmp_path) == summary
    (tmp_path / "kubric.pt").write_bytes(b"partial")
    assert completed_cache(tmp_path) is None


def test_three_dataset_prompts_match_training_plan_and_yaml_exactly():
    import yaml

    assert set(PROMPTS) == {"kubric", "pointodyssey", "dynamic_replica"}
    assert all(prompt.startswith(TASK_INSTRUCTION) for prompt in PROMPTS.values())
    assert all(prompt.count(TASK_INSTRUCTION) == 1 for prompt in PROMPTS.values())
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/worldbridge4d_256_three_dataset_200m_fsdp.yaml").read_text())
    assert config["prompts"] == PROMPTS
    plan = (root / "docs/WORLDBRIDGE4D_256_THREE_DATASET_TRAINING_PLAN.md").read_text()
    documented = re.findall(r"```text\n(Estimate dense three-dimensional point trajectories[^\n]+)\n```", plan)
    assert documented == [PROMPTS[name] for name in ("kubric", "pointodyssey", "dynamic_replica")]


def test_inference_condition_is_dataset_specific_and_checksum_verified(tmp_path):
    import hashlib

    prompts = {name: PROMPTS[name] for name in PROMPTS}
    paths = {"metadata": str(tmp_path / "metadata.json")}
    metadata = {}
    for index, name in enumerate(("kubric", "pointodyssey", "dynamic_replica")):
        path = tmp_path / f"{name}.pt"
        value = torch.full((1, 512, 4096), float(index), dtype=torch.bfloat16)
        torch.save({"encoder_hidden_states": value}, path)
        paths[name] = str(path)
        metadata[name] = {
            "dataset": name, "prompt": prompts[name], "shape": [1, 512, 4096],
            "condition_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    config = {"prompts": prompts, "text_conditions": paths}
    pointodyssey, record = load_inference_text_condition(config, "pointodyssey")
    assert torch.equal(pointodyssey, torch.ones_like(pointodyssey))
    assert record["prompt"] == PROMPTS["pointodyssey"]
    with pytest.raises(ValueError, match="dataset must be"):
        load_inference_text_condition(config, "unknown")
    with Path(paths["kubric"]).open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_inference_text_condition(config, "kubric")


def test_resume_planning_uses_status_sidecar_not_full_checkpoint_load():
    source = (Path(__file__).resolve().parents[1] / "scripts/train_three_dataset_256_fsdp.py").read_text()
    planning = source[source.index("# Cache planning needs only"):source.index("planned_start =")]
    assert '"resume_status_path", resume.parent / "train_status.json"' in planning
    assert "torch.load" not in planning


def test_full_resume_skips_redundant_native_wan_weights(monkeypatch, tmp_path):
    tiny = nn.Linear(2, 2)
    monkeypatch.setattr(
        WanDiTMapping, "_architecture", staticmethod(lambda _path: {"num_layers": 1}),
    )
    monkeypatch.setattr(WanDiTMapping, "_new_model", staticmethod(lambda _config: tiny))
    missing = tmp_path / "native_wan_weights_are_not_needed.safetensors"
    loaded = WanDiTMapping._load(
        str(missing), torch.device("cpu"), torch.float32,
        load_pretrained_weights=False,
    )
    assert loaded is tiny
    with pytest.raises(FileNotFoundError, match="WAN DiT checkpoint not found"):
        WanDiTMapping._load(
            str(missing), torch.device("cpu"), torch.float32,
            load_pretrained_weights=True,
        )

    source = (Path(__file__).resolve().parents[1] / "scripts/train_three_dataset_256_fsdp.py").read_text()
    assert "load_wan_pretrained = not resume.is_file()" in source
    assert "load_wan_pretrained=load_wan_pretrained" in source


def test_full_resume_loads_rank0_model_before_fsdp_sync():
    source = (Path(__file__).resolve().parents[1] / "scripts/train_three_dataset_256_fsdp.py").read_text()
    helper = source[
        source.index("def load_unwrapped_model_checkpoint"):
        source.index("def load_optimizer_checkpoint")
    ]
    assert 'model.load_state_dict(payload["model"], strict=True)' in helper
    assert "fsdp_state_context" not in helper

    main = source[source.index("def main()") :]
    model_load = main.index("load_unwrapped_model_checkpoint(")
    fsdp_wrap = main.index("fsdp = FSDP(")
    optimizer_load = main.index("load_optimizer_checkpoint(")
    assert model_load < fsdp_wrap < optimizer_load
    assert "sync_module_states=True" in main[fsdp_wrap:optimizer_load]


def test_wandb_resume_can_skip_already_published_steps():
    source = (Path(__file__).resolve().parents[1] / "scripts/train_three_dataset_256_fsdp.py").read_text()
    assert '"--wandb-log-after-step", type=int, default=-1' in source
    assert "completed > args.wandb_log_after_step" in source


def test_checkpoint_publishes_matching_planning_sidecar():
    source = (Path(__file__).resolve().parents[1] / "scripts/train_three_dataset_256_fsdp.py").read_text()
    assert 'atomic_json(checkpoint_dir / "train_status.json", checkpoint_status)' in source
    assert 'atomic_json(output / "train_status.json", checkpoint_status)' in source


def test_staging_is_checksum_verified_reusable_and_wires_runtime_paths(tmp_path):
    import yaml
    from scripts.stage_three_dataset_256_inputs import stage_training_inputs

    wan = tmp_path / "wan"; wan.mkdir()
    dit = wan / "diffusion_pytorch_model.safetensors"; dit.write_bytes(b"dit-weights")
    vae = wan / "Wan2.1_VAE.pth"; vae.write_bytes(b"vae-weights")
    output = tmp_path / "output"; output.mkdir()
    resume = output / "latest.pt"; resume.write_bytes(b"resume-state")
    (output / "train_status.json").write_text('{"completed_steps":2}\n')
    source_config = tmp_path / "config.yaml"
    source_config.write_text(yaml.safe_dump({"wan_root": str(wan)}))
    staged_config, staged_resume = stage_training_inputs(
        source_config, tmp_path / "ssd", resume
    )
    values = yaml.safe_load(staged_config.read_text())
    assert Path(values["wan_checkpoint"]).read_bytes() == b"dit-weights"
    assert Path(values["vae_checkpoint"]).read_bytes() == b"vae-weights"
    assert Path(staged_resume).read_bytes() == b"resume-state"
    assert values["resume_status_path"] == str(output / "train_status.json")
    second_config, second_resume = stage_training_inputs(
        source_config, tmp_path / "ssd", resume
    )
    assert second_config == staged_config
    assert second_resume == staged_resume
    # Resume checkpoints already live on /data and are intentionally not copied
    # into a second object on the same filesystem.
    assert len(list((tmp_path / "ssd" / "objects").iterdir())) == 2


def test_inference_script_encodes_backbone_once_before_target_chunks():
    source = (Path(__file__).resolve().parents[1] / "scripts/infer_three_dataset_256.py").read_text()
    assert "z4d = model.backbone(latent, condition)" in source
    assert "model.decoder(z4d, source_tensor, target_tensor)" in source
    assert "model(latent, source_tensor, target_tensor, condition)" not in source


def test_bf16_layer_gate_audit_uses_quantized_config_logits():
    logits = torch.tensor([0.0, 0.0, 0.0, -1.0986122887], dtype=torch.bfloat16)
    weights = logits.float().softmax(0)
    assert not torch.allclose(weights, torch.tensor([0.3, 0.3, 0.3, 0.1]), atol=1e-7, rtol=0)
    torch.testing.assert_close(weights, logits.float().softmax(0), atol=1e-7, rtol=0)


def test_256_latent_contract_and_explicit_prompt_reaches_mapping():
    assert WAN_LATENT_SHAPE_256 == (16, 6, 32, 32)
    mapping = TinyExplicitConditionMapping()
    backbone = WanHiddenGeometryBackbone(
        mapping, hidden_layers=(0, 1, 2, 3), geometry_dim=32,
        spatial_size=32, motion_slots=4, num_heads=4,
        layer_gate_initial_logits=(0.0, 0.0, 0.0, -1.0986122887),
    )
    condition = torch.randn(1, 512, 4096)
    output = backbone(torch.randn(1, *WAN_LATENT_SHAPE_256), condition)
    assert mapping.conditions == [condition]
    assert output.dense.shape == (1, 32, 21, 32, 32)
    torch.testing.assert_close(backbone.layer_weights(), torch.tensor([0.3, 0.3, 0.3, 0.1]))


def test_200m_decoder_exact_parameter_count_component():
    decoder = DenseQueryDecoder(
        num_frames=21, latent_shape=(512, 21, 32, 32), query_dim=1536,
        embedding_dim=768, num_layers=5, num_heads=12,
        upsample_channels=(1536, 768, 384, 192), output_size=(256, 256),
        query_grid_size=32, structured_motion_slots=8,
        structured_local_queries=True,
    )
    assert sum(parameter.numel() for parameter in decoder.parameters()) == 175_178_627


def test_required_latents_respect_true_microbatch_slots():
    from scripts.train_three_dataset_256_fsdp import required_latent_indices

    class Dataset:
        rows = []
        def __len__(self): return 100

    datasets = {name: Dataset() for name in ("kubric", "pointodyssey", "dynamic_replica")}
    accumulated = required_latent_indices(
        datasets, 20260812, 0, 1, rank=0, accumulation=2, microbatch_per_gpu=1
    )
    batched = required_latent_indices(
        datasets, 20260812, 0, 1, rank=0, accumulation=1, microbatch_per_gpu=2
    )
    assert accumulated == batched
    assert sum(map(len, batched.values())) == 2


def test_pipeline_work_is_global_unique_balanced_and_needed_step_ordered():
    from scripts.train_three_dataset_256_fsdp import pipeline_work_for_rank

    gathered = [
        [(9, "kubric", 1), (3, "pointodyssey", 4), (7, "kubric", 1)],
        [(2, "kubric", 1), (5, "dynamic_replica", 8), (6, "pointodyssey", 4)],
    ]
    rank0 = pipeline_work_for_rank(gathered, 0, 2)
    rank1 = pipeline_work_for_rank(gathered, 1, 2)
    combined = rank0 + rank1
    keys = [(name, index) for _step, name, index in combined]
    assert len(keys) == len(set(keys)) == 3
    earliest = {(name, index): step for step, name, index in combined}
    assert earliest[("kubric", 1)] == 2
    assert earliest[("pointodyssey", 4)] == 3
    assert earliest[("dynamic_replica", 8)] == 5
    assert rank0 == sorted(rank0, key=lambda value: (value[0], ("kubric", "pointodyssey", "dynamic_replica").index(value[1]), value[2]))
    assert rank1 == sorted(rank1, key=lambda value: (value[0], ("kubric", "pointodyssey", "dynamic_replica").index(value[1]), value[2]))


def test_offline_lazy_cache_hash_owner_balances_fully_overlapping_plans():
    from scripts.train_three_dataset_256_fsdp import lazy_latent_owner

    for name in ("kubric", "pointodyssey", "dynamic_replica"):
        counts = [0, 0]
        for index in range(10_000):
            counts[lazy_latent_owner(name, index, world=2)] += 1
        assert counts == [5_000, 5_000]


def test_kubric_mmap_conversion_is_exact_atomic_and_resumable(tmp_path):
    from scripts.convert_kubric_geometry_to_mmap import convert

    source = tmp_path / "compact"
    destination = tmp_path / "mmap"
    source.mkdir()
    originals = []
    for index in range(3):
        depth = np.arange(24, dtype=np.float32).reshape(2, 3, 4) + index
        depth_valid = (depth % 2 == 0).astype(np.uint8)
        segmentation = np.arange(24, dtype=np.int32).reshape(2, 3, 4) + 10 * index
        originals.append((depth, depth_valid, segmentation))
        np.savez_compressed(
            source / f"geom_{index:06d}.npz",
            depth=depth, depth_valid=depth_valid, segmentation=segmentation,
            camera_positions=np.zeros((2, 3), np.float32),
            camera_quaternions=np.zeros((2, 4), np.float32),
            focal_length=np.float32(1), sensor_width=np.float32(1),
            field_of_view=np.float32(1),
            instance_positions=np.zeros((1, 2, 3), np.float32),
            instance_quaternions=np.zeros((1, 2, 4), np.float32),
            instance_dynamic=np.zeros(1, np.uint8),
            instance_visibility=np.zeros((1, 2), np.uint16),
            depth_range=np.array([0, 1], np.float32), clip_start=np.int64(0),
        )
    manifest = convert(source, destination, range(3), shard_size=2, min_free_gib=0)
    assert manifest["complete"] and manifest["count"] == 3 and manifest["shards"] == 2
    assert not list(destination.glob("*.tmp"))
    store = KubricGeometryMmapStore(destination, max_open_shards=1)
    for index, expected in enumerate(originals):
        for actual, value in zip(store.read(index), expected):
            assert np.array_equal(actual, value)
    assert len(store._shards) == 1
    all_shards = KubricGeometryMmapStore(destination)
    all_shards.read(0); all_shards.read(2)
    assert all_shards.max_open_shards == 2
    assert len(all_shards._shards) == 2
    old = MOViF256Dataset._load_compact_sample(source / "geom_000001.npz")
    new = MOViF256Dataset._load_compact_sample(
        source / "geom_000001.npz", store.read(1),
    )
    for field in ("depth", "depth_valid", "segmentation"):
        assert np.array_equal(getattr(old, field), getattr(new, field))
    mtimes = {path: path.stat().st_mtime_ns for path in destination.glob("*.npy")}
    convert(source, destination, range(3), shard_size=2, min_free_gib=0)
    assert mtimes == {path: path.stat().st_mtime_ns for path in destination.glob("*.npy")}


def test_experimental_k_modes_require_explicit_world_size_flags():
    import scripts.train_three_dataset_256_fsdp as train

    base = {
        "image_size": 256, "clip_length": 21, "latent_spatial_size": 32,
        "query_dim": 1536, "embedding_dim": 768, "num_cross_attn_layers": 5,
        "num_heads": 12, "geometry_dim": 512, "geometry_spatial_size": 32,
        "motion_slots": 8, "gradient_accumulation": 2, "microbatch_per_gpu": 1,
        "wan_hidden_layers": [13, 14, 15, 29],
        "layer_gate_initial_logits": [0.0, 0.0, 0.0, -1.0986122887],
        "targets_per_source": 21,
    }
    train.validate_config(base, world=2, allow_two_gpu=True)
    batched = {**base, "gradient_accumulation": 1, "microbatch_per_gpu": 2}
    train.validate_config(batched, world=2, allow_two_gpu=True)
    matched_k10 = {
        **base, "targets_per_source": 10,
        "gradient_accumulation": 2, "microbatch_per_gpu": 2,
    }
    train.validate_config(matched_k10, world=2, allow_two_gpu=True)
    k16_b2_a2 = {**matched_k10, "targets_per_source": 16}
    train.validate_config(k16_b2_a2, world=2, allow_two_gpu=True)
    train.validate_config(
        {**k16_b2_a2, "targets_per_source": 19},
        world=2, allow_two_gpu=True,
    )
    train.validate_config(
        {
            **k16_b2_a2,
            "geometry_prefetch_depth": 4,
            "geometry_prefetch_workers": 8,
            "datasets": {"kubric": {
                "geometry_sample_cache_size": 32,
                "geometry_mmap_max_open_shards": 90,
            }},
        },
        world=2, allow_two_gpu=True,
    )
    with pytest.raises(ValueError, match="geometry_prefetch_depth"):
        train.validate_config(
            {**k16_b2_a2, "geometry_prefetch_depth": 0},
            world=2, allow_two_gpu=True,
        )
    with pytest.raises(ValueError, match="geometry_prefetch_workers"):
        train.validate_config(
            {**k16_b2_a2, "geometry_prefetch_workers": 33},
            world=2, allow_two_gpu=True,
        )
    with pytest.raises(ValueError, match="geometry_sample_cache_size"):
        train.validate_config(
            {**k16_b2_a2, "datasets": {"kubric": {"geometry_sample_cache_size": 0}}},
            world=2, allow_two_gpu=True,
        )
    with pytest.raises(ValueError, match="geometry_mmap_max_open_shards"):
        train.validate_config(
            {**k16_b2_a2, "datasets": {"kubric": {"geometry_mmap_max_open_shards": 0}}},
            world=2, allow_two_gpu=True,
        )
    k16_four_gpu = {
        **k16_b2_a2, "gradient_accumulation": 1,
    }
    train.validate_config(
        k16_four_gpu, world=4, allow_two_gpu=False,
        allow_four_gpu_experiment=True,
    )
    with pytest.raises(ValueError, match="explicit compatible"):
        train.validate_config(matched_k10, world=4, allow_two_gpu=False)
    with pytest.raises(ValueError, match="explicit compatible"):
        train.validate_config(batched, world=4, allow_two_gpu=False)
    with pytest.raises(ValueError, match="one of"):
        train.validate_config(base, world=4, allow_two_gpu=False)
    with pytest.raises(ValueError, match="one of"):
        train.validate_config(
            {**k16_four_gpu, "targets_per_source": 21}, world=4,
            allow_two_gpu=False, allow_four_gpu_experiment=True,
        )
    with pytest.raises(ValueError, match="requires exactly 4 ranks"):
        train.validate_config(
            k16_four_gpu, world=2, allow_two_gpu=True,
            allow_four_gpu_experiment=True,
        )
    extended = {
        **k16_b2_a2,
        "targets_per_source": 19,
        "warmup_steps": 1000,
        "schedule_horizon_steps": 100000,
        "schedule_extension_start_step": 56000,
        "schedule_extension_horizon_steps": 150000,
        "max_steps": 150000,
        "diagnostic_ensure_dataset_coverage": True,
    }
    train.validate_config(extended, world=2, allow_two_gpu=True)
    with pytest.raises(ValueError, match="extension"):
        train.validate_config(
            {**extended, "schedule_extension_horizon_steps": 90000},
            world=2, allow_two_gpu=True,
        )


def _handoff_original_config():
    return {
        "targets_per_source": 19,
        "microbatch_per_gpu": 2,
        "gradient_accumulation": 2,
        "warmup_steps": 1000,
        "schedule_horizon_steps": 100000,
        "max_steps": 100000,
        "checkpoint_steps": [55000, 60000, 100000],
        "datasets": {"kubric": {}, "pointodyssey": {}, "dynamic_replica": {}},
        "tracking": {"tags": ["k19"]},
    }


def test_gpu14_handoff_config_preserves_protocol_and_changes_only_registered_controls(tmp_path):
    from scripts.prepare_three_dataset_256_gpu14_handoff import build_extended_config

    config = build_extended_config(
        _handoff_original_config(), 56435, 150000, tmp_path / "train_status.json",
    )
    assert config["targets_per_source"] == 19
    assert config["microbatch_per_gpu"] == config["gradient_accumulation"] == 2
    assert config["schedule_horizon_steps"] == 100000
    assert config["schedule_extension_start_step"] == 56435
    assert config["schedule_extension_horizon_steps"] == config["max_steps"] == 150000
    assert config["geometry_prefetch_workers"] == 4
    assert config["geometry_prefetch_depth"] == 2
    assert config["diagnostic_ensure_dataset_coverage"] is True
    assert config["datasets"]["kubric"]["geometry_sample_cache_size"] == 16
    assert config["datasets"]["kubric"]["geometry_mmap_max_open_shards"] == 90
    assert config["checkpoint_steps"] == [60000, 100000, 150000]


def test_gpu14_handoff_preparer_freezes_and_verifies_complete_checkpoint(tmp_path):
    from scripts.prepare_three_dataset_256_gpu14_handoff import (
        finalize_checksum_marker, prepare, verify_marker,
    )

    source = tmp_path / "latest.pt"
    status = tmp_path / "train_status.json"
    clips = {"kubric": 10, "pointodyssey": 20, "dynamic_replica": 30}
    torch.save({
        "format": 3,
        "model": {"weight": torch.ones(1)},
        "optimizer": {"state": {0: {"step": torch.tensor(1)}}},
        "config": _handoff_original_config(),
        "training_state": {
            "global_step": 56435, "world_size": 2,
            "rng_states": [{"rank": 0}, {"rank": 1}], "clips_seen": clips,
        },
    }, source)
    status.write_text(json.dumps({
        "completed_steps": 56435, "world_size": 2, "clips_seen": clips,
    }))
    value = prepare(
        source, status, tmp_path / "handoff", tmp_path / "fast-checkpoints", 150000,
    )
    marker = verify_marker(Path(value["marker"]))
    assert marker["completed_step"] == 56435
    assert Path(marker["checkpoint"]).stat().st_ino == source.stat().st_ino
    assert Path(marker["config"]).is_file()
    assert json.loads(Path(marker["resume_status"]).read_text())["completed_steps"] == 56435

    deferred = prepare(
        source, status, tmp_path / "deferred-handoff",
        tmp_path / "deferred-fast-checkpoints", 150000, defer_checksum=True,
    )
    deferred_marker = Path(deferred["marker"])
    assert verify_marker(deferred_marker, verify_checksum=False)["checksum_state"] == "deferred"
    with pytest.raises(ValueError, match="not complete"):
        verify_marker(deferred_marker)
    completed = finalize_checksum_marker(deferred_marker)
    assert completed["checksum_state"] == "complete"
    assert len(completed["checkpoint_sha256"]) == 64
    assert verify_marker(deferred_marker)["checkpoint_sha256"] == completed["checkpoint_sha256"]


def test_periodic_checkpoint_pruning_bounds_disk_usage(tmp_path):
    from scripts.train_three_dataset_256_fsdp import prune_periodic_checkpoints

    for step in (1000, 2000, 5000):
        (tmp_path / f"checkpoint-{step:07d}.pt").write_bytes(b"checkpoint")
    (tmp_path / "latest.pt").write_bytes(b"latest")
    removed = prune_periodic_checkpoints(tmp_path, keep_last=2)
    assert removed == ["checkpoint-0001000.pt"]
    assert (tmp_path / "latest.pt").read_bytes() == b"latest"
    assert [path.name for path in sorted(tmp_path.glob("checkpoint-*.pt"))] == [
        "checkpoint-0002000.pt", "checkpoint-0005000.pt",
    ]


def test_latest_checkpoint_is_atomic_same_inode_link(tmp_path):
    from scripts.train_three_dataset_256_fsdp import update_latest_checkpoint

    first = tmp_path / "checkpoint-0000050.pt"; first.write_bytes(b"first")
    second = tmp_path / "checkpoint-0000100.pt"; second.write_bytes(b"second")
    latest = tmp_path / "latest.pt"
    update_latest_checkpoint(first, latest)
    assert latest.read_bytes() == b"first"
    assert latest.stat().st_ino == first.stat().st_ino
    update_latest_checkpoint(second, latest)
    assert latest.read_bytes() == b"second"
    assert latest.stat().st_ino == second.stat().st_ino


def test_checkpoint_replica_is_ordered_atomic_and_supersedes_old_jobs(tmp_path):
    from scripts.replicate_checkpoint import replicate_checkpoint

    source = tmp_path / "ssd" / "checkpoint-0000100.pt"
    source.parent.mkdir(); source.write_bytes(b"new-checkpoint")
    destination = tmp_path / "hdd" / "latest.pt"
    destination.parent.mkdir(); destination.write_bytes(b"old-checkpoint")
    assert replicate_checkpoint(
        source, destination, 100, chunk_bytes=4, throttle_seconds=0,
    ) == "complete"
    assert destination.read_bytes() == b"new-checkpoint"
    older = tmp_path / "ssd" / "checkpoint-0000050.pt"
    older.write_bytes(b"must-not-downgrade")
    assert replicate_checkpoint(
        older, destination, 50, chunk_bytes=4, throttle_seconds=0,
    ) == "superseded"
    assert destination.read_bytes() == b"new-checkpoint"
    status = json.loads((destination.parent / "latest.pt.replica.json").read_text())
    assert status["requested_step"] == status["completed_step"] == 100


def test_train_status_recovery_requires_complete_checkpoint_and_conflict_free_sidecar(
    tmp_path, monkeypatch,
):
    from scripts.recover_three_dataset_256_train_status import recover_status

    checkpoint = tmp_path / "latest.pt"
    output = tmp_path / "train_status.json"
    payload = {
        "format": 3,
        "config": {"max_steps": 100},
        "training_state": {
            "global_step": 50, "world_size": 2,
            "rng_states": [{"rank": 0}, {"rank": 1}],
            "clips_seen": {
                "kubric": 136, "pointodyssey": 120, "dynamic_replica": 144,
            },
        },
    }
    monkeypatch.setattr(
        "scripts.recover_three_dataset_256_train_status.torch.load",
        lambda *_args, **_kwargs: payload,
    )
    checkpoint.write_bytes(b"checkpoint")
    status = recover_status(checkpoint, output)
    assert status["completed_steps"] == 50
    assert status["world_size"] == 2
    assert status["recovered_from_checkpoint"] is True
    assert json.loads(output.read_text()) == status
    assert recover_status(checkpoint, output) == status

    output.write_text('{"completed_steps":49}\n')
    with pytest.raises(RuntimeError, match="conflicting status sidecar"):
        recover_status(checkpoint, output)


def test_launchers_default_staging_and_latents_to_persistent_storage():
    root = Path(__file__).resolve().parents[1]
    scripts = (
        "run_three_dataset_256_fsdp.sh", "run_precompute_latents_5gpu.sh",
        "run_three_dataset_256_auto_pick.sh", "run_three_dataset_256_2gpu_k16_100k.sh",
        "run_three_dataset_256_2gpu_k16_100k_precache.sh",
        "run_three_dataset_256_4gpu_k10_10k.sh",
    )
    for name in scripts:
        source = (root / "scripts" / name).read_text()
        assert "STAGING_ROOT=\"${STAGING_ROOT:-/data/WorldBridge4D-persistent/" in source
        assert "STAGING_ROOT=\"${STAGING_ROOT:-/tmp/" not in source


def test_two_gpu_k16_100k_launcher_is_fresh_pinned_and_wandb_online():
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "scripts/run_three_dataset_256_2gpu_k16_100k.sh").read_text()
    assert 'GPUS="${GPUS:-4,6}"' in launcher
    assert 'NPROC=2' in launcher
    assert 'STEPS="${STEPS:-100000}"' in launcher
    assert 'FRESH_START=1' in launcher
    assert 'WANDB_MODE=online' in launcher
    assert 'LAZY_VAE_PIPELINE=1' in launcher
    assert 'GPU_FREE_MIN_MIB="${GPU_FREE_MIN_MIB:-76000}"' in launcher


def test_four_gpu_k10_launcher_is_fresh_pinned_and_wandb_online():
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "scripts/run_three_dataset_256_4gpu_k10_10k.sh").read_text()
    generic = (root / "scripts/run_three_dataset_256_fsdp.sh").read_text()
    assert 'GPUS="${GPUS:-2,4,5,6}"' in launcher
    assert 'NPROC=4' in launcher
    assert 'STEPS="${STEPS:-10000}"' in launcher
    assert 'ALLOW_FOUR_GPU_EXPERIMENT=1' in launcher
    assert 'FRESH_START=1' in launcher
    assert 'WANDB_MODE=online' in launcher
    assert 'LAZY_VAE_PIPELINE=1' in launcher
    assert 'GPU_FREE_MIN_MIB="${GPU_FREE_MIN_MIB:-55000}"' in launcher
    assert 'FRESH_START=1 refuses existing trajectory artifact' in generic
    assert 'CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OUTPUT}"' in generic
    assert '--checkpoint-dir "$CHECKPOINT_DIR"' in generic
    assert '--durable-checkpoint "$DURABLE_CHECKPOINT"' in generic


def test_gpu14_150k_watcher_is_pinned_audited_stable_and_resume_only():
    root = Path(__file__).resolve().parents[1]
    watcher = (root / "scripts/wait_resume_three_dataset_256_gpu14_150k.sh").read_text()
    assert "--verify-marker" in watcher
    assert 'GPUS=1,4' in watcher
    assert 'NPROC=2' in watcher
    assert 'STEPS=150000' in watcher
    assert 'FRESH_START=0' in watcher
    assert 'STABLE_SECONDS="${STABLE_SECONDS:-90}"' in watcher
    assert 'MIN_FREE_MIB="${MIN_FREE_MIB:-76000}"' in watcher
    assert 'query-compute-apps=pid' in watcher
    assert 'STAGE_INPUTS=0' in watcher
    assert 'POST_RESUME_CHECKSUM_MARKER="$MARKER"' in watcher
    assert "stage_three_dataset_256_inputs.py" not in watcher
    assert '--verify-marker "$MARKER" --skip-checksum' in watcher
    assert 'WANDB_LOG_AFTER_STEP="$RESUME_STEP"' in watcher
    assert " kill " not in watcher and "pkill" not in watcher


def test_post_resume_checksum_starts_only_after_strict_optimizer_restore():
    import inspect
    from worldbridge.dense4d_runtime import build_real_model

    parameter = inspect.signature(build_real_model).parameters["load_wan_pretrained"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is True
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/train_three_dataset_256_fsdp.py").read_text()
    restore = source.index("load_optimizer_checkpoint(", source.index("def main()"))
    loaded = source.index('"event": "resume_state_loaded"', restore)
    checksum = source.index("launch_post_resume_checksum(", loaded)
    loop = source.index("for step in range(start_step, target_steps):", checksum)
    assert restore < loaded < checksum < loop


def test_three_dataset_cycle_has_exact_ratio_and_is_resume_pure():
    cycle = deterministic_dataset_schedule(20260812)
    assert len(cycle) == 20
    assert cycle.count("kubric") == 7
    assert cycle.count("pointodyssey") == 6
    assert cycle.count("dynamic_replica") == 7
    assert [dataset_for_step(step, 20260812) for step in range(40)] == list(cycle) * 2


def test_small_256_index_maps_back_to_geometry_clip_ids(tmp_path, monkeypatch):
    split = tmp_path / "splits"; split.mkdir()
    (split / "train.jsonl").write_text('{"clip_id":"keep"}\n')
    latent = tmp_path / "latents" / "wan2.1_1.3b_fp32_256"; latent.mkdir(parents=True)
    class Geometry:
        rows = [{"clip_id": "skip"}, {"clip_id": "keep"}]
        def __len__(self): return 2
        def source_all_targets(self, index, source): return index, source
        def source_all_targets_with_visibility(self, index, source): return index, source, True
    class Latents:
        def __getitem__(self, index): return index
    monkeypatch.setattr("worldbridge.training256.LatentShardStore", lambda *_args, **_kwargs: Latents())
    dataset = CachedExternalDataset(Geometry(), tmp_path, "pointodyssey")
    assert len(dataset) == 1
    assert dataset.source_all_targets(0, 7) == (1, 7)
    assert dataset.clean_latent(0) == 0


def test_durable_cache_preflight_requires_backup_and_live_target(tmp_path):
    from scripts.validate_three_dataset_256_cache_roots import validate_cache_roots

    config = {"datasets": {}}
    for name in ("kubric", "pointodyssey", "dynamic_replica"):
        cache_root = tmp_path / "persistent" / name
        config["datasets"][name] = {"cache_root": str(cache_root)}
        (cache_root / "latents" / "wan2.1_1.3b_fp32_256_lazy_backup").mkdir(
            parents=True
        )
    roots = validate_cache_roots(config, create=True, forbidden_roots=())
    assert all(Path(path).is_dir() for path in roots.values())

    kubric = Path(roots["kubric"])
    kubric.rmdir()
    missing = tmp_path / "missing-hot-cache"
    kubric.symlink_to(missing, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink target missing"):
        validate_cache_roots(config, forbidden_roots=())

    kubric.unlink()
    scratch = tmp_path / "scratch"; scratch.mkdir()
    kubric.symlink_to(scratch, target_is_directory=True)
    assert validate_cache_roots(config, forbidden_roots=())["kubric"] == str(kubric)

    backup = (
        tmp_path / "persistent" / "pointodyssey" / "latents" /
        "wan2.1_1.3b_fp32_256_lazy_backup"
    )
    backup.rmdir()
    with pytest.raises(RuntimeError, match="durable latent backup missing"):
        validate_cache_roots(config, forbidden_roots=())


def test_durable_cache_count_audit_is_exact(tmp_path):
    from scripts.validate_three_dataset_256_cache_roots import assert_expected_latent_files

    roots = {}
    for name, count in (("kubric", 2), ("pointodyssey", 1), ("dynamic_replica", 0)):
        root = tmp_path / name; root.mkdir()
        roots[name] = str(root)
        for index in range(count):
            (root / f"latent_{index:08d}.safetensors").touch()
    assert assert_expected_latent_files(roots, 3) == {
        "kubric": 2, "pointodyssey": 1, "dynamic_replica": 0,
    }
    with pytest.raises(RuntimeError, match="count mismatch"):
        assert_expected_latent_files(roots, 4)


def test_kubric_compact_geometry_audit_requires_exact_atomic_outputs(tmp_path):
    from scripts.compact_kubric_geometry import audit_outputs

    wanted = {2, 7}
    for index in wanted:
        (tmp_path / f"geom_{index:06d}.npz").write_bytes(b"npz")
    assert audit_outputs(wanted, tmp_path) == {
        "expected": 2, "files": 2, "bytes": 6,
    }

    (tmp_path / "geom_000007.npz").unlink()
    with pytest.raises(RuntimeError, match="missing=1"):
        audit_outputs(wanted, tmp_path)

    (tmp_path / "geom_000007.npz").write_bytes(b"npz")
    (tmp_path / "geom_000002.2.tmp.npz").touch()
    with pytest.raises(RuntimeError, match="temporaries=1"):
        audit_outputs(wanted, tmp_path)


def test_tmp_migration_tombstone_fails_closed():
    script = Path(__file__).resolve().parents[1] / "scripts/migrate_latents_to_tmp.sh"
    source = script.read_text()
    assert 'exit 2' in source
    assert 'rm -rf "$src"' not in source
    assert 'ln -s "$dst" "$src"' not in source


def test_lazy_latent_cache_roundtrip_identity_checksum_and_no_overwrite(tmp_path):
    cache = LazyLatentCache(tmp_path, "kubric")
    checksum = "a" * 64
    value = np.arange(np.prod(WAN_LATENT_SHAPE_256), dtype=np.float32).reshape(WAN_LATENT_SHAPE_256)
    assert cache.write(7, "clip-7", value, checksum)
    assert not cache.write(7, "clip-7", value + 1, checksum)
    np.testing.assert_array_equal(cache.read(7, "clip-7"), value)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        cache.read(7, "wrong-clip")
    cache.set_vae_sha256("b" * 64)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        cache.read(7, "clip-7")


def test_cached_external_dataset_falls_back_to_lazy_latent(tmp_path):
    split = tmp_path / "splits"; split.mkdir()
    (split / "train.jsonl").write_text('{"clip_id":"keep"}\n')
    class Geometry:
        rows = [{"clip_id": "keep"}]
        def source_all_targets(self, index, source): return index, source
        def source_all_targets_with_visibility(self, index, source): return index, source, True
        def rgb(self, index): return np.zeros((21, 256, 256, 3), np.uint8)
    dataset = CachedExternalDataset(
        Geometry(), tmp_path, "pointodyssey", allow_missing_latents=True
    )
    assert not dataset.latent_cached(0)
    value = np.zeros(WAN_LATENT_SHAPE_256, np.float32)
    assert dataset.cache_latent(0, value, "c" * 64)
    assert dataset.latent_cached(0)
    np.testing.assert_array_equal(dataset.clean_latent(0), value)
    assert dataset.rgb(0).shape == (21, 256, 256, 3)


def test_parent_balanced_plan_keeps_ranks_in_one_scene_block():
    class Dataset:
        rows = [
            {"parent_id": "a"}, {"parent_id": "a"},
            {"parent_id": "b"}, {"parent_id": "b"},
        ]
        def __len__(self):
            return len(self.rows)

    dataset = Dataset()
    first = deterministic_sample_plan(dataset, "pointodyssey", 5, 8, 0, 0)
    second = deterministic_sample_plan(dataset, "pointodyssey", 5, 8, 1, 1)
    assert dataset.rows[first[0]]["parent_id"] == dataset.rows[second[0]]["parent_id"]
    assert first[0] != second[0]


def test_eligible_k_sampling_excludes_empty_pairs_and_is_deterministic():
    valid = np.zeros((21, 4, 4), bool)
    valid[[0, 3, 9, 12, 20], 1, 1] = True
    first = sample_eligible_targets(valid, 4, np.random.default_rng(7))
    second = sample_eligible_targets(valid, 4, np.random.default_rng(7))
    np.testing.assert_array_equal(first, second)
    assert len(set(first.tolist())) == 4
    assert set(first).issubset({0, 3, 9, 12, 20})
    with pytest.raises(ValueError, match="fewer than K=6"):
        sample_eligible_targets(valid, 6, np.random.default_rng(8))


def test_source_selection_requires_exact_target_capacity():
    class Dataset:
        def source_all_targets(self, index, source):
            xyz = np.zeros((21, 3, 1, 1), np.float32)
            valid = np.zeros((21, 1, 1), bool)
            valid[: (3 if source == 0 else 6)] = True
            return xyz, valid

    source, _xyz, valid = source_with_eligible_targets(
        Dataset(), 0, np.array([0, 1]), min_targets=4,
    )
    assert source == 1
    assert valid.reshape(21, -1).any(axis=1).sum() == 6
    with pytest.raises(ValueError, match="no source with 7 eligible targets"):
        source_with_eligible_targets(
            Dataset(), 0, np.array([0, 1]), min_targets=7,
        )


def test_pair_loss_ignores_empty_pair_instead_of_treating_it_as_zero():
    prediction = torch.zeros(1, 2, 3, 1, 1)
    target = torch.ones_like(prediction)
    valid = torch.tensor([[[[True]], [[False]]]])
    loss = masked_pair_smooth_l1(prediction, target, valid, beta=0.05)
    expected = torch.nn.functional.smooth_l1_loss(
        prediction[:, :1], target[:, :1], beta=0.05, reduction="none"
    ).sum(dim=2).mean()
    torch.testing.assert_close(loss, expected)
    with pytest.raises(ValueError, match="no pair"):
        masked_pair_smooth_l1(prediction, target, torch.zeros_like(valid))


def test_cosine_schedule_preserves_group_lr_ratios():
    a, b = nn.Parameter(torch.ones(())), nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([
        {"params": [a], "lr": 5e-5}, {"params": [b], "lr": 3e-4},
    ])
    factor = apply_cosine_schedule(optimizer, 1, 1000, 100000)
    assert factor == 0.001
    np.testing.assert_allclose([group["lr"] for group in optimizer.param_groups], [5e-8, 3e-7])
    factor = apply_cosine_schedule(optimizer, 1000, 1000, 100000)
    assert factor == 1.0
    np.testing.assert_allclose([group["lr"] for group in optimizer.param_groups], [5e-5, 3e-4])


def test_cosine_extension_is_continuous_and_reaches_zero_at_150k():
    start = 56435
    before = extended_cosine_learning_rate_factor(start, 1000, 100000, start, 150000)
    original = extended_cosine_learning_rate_factor(start, 1000, 100000)
    assert before == original
    after = extended_cosine_learning_rate_factor(start + 1, 1000, 100000, start, 150000)
    assert 0 < after <= before
    assert extended_cosine_learning_rate_factor(150000, 1000, 100000, start, 150000) == 0


def test_dataset_coverage_diagnostics_cannot_alias_the_20_step_cycle():
    last_cycle = {}
    logged = {cycle: set() for cycle in range(4)}
    for schedule_step in range(80):
        completed = schedule_step + 1
        name = dataset_for_step(schedule_step, 20260812)
        due = training_diagnostic_due(
            completed, 0, 80, name, schedule_step, 5, True, last_cycle,
        )
        if due:
            cycle = schedule_step // 20
            last_cycle[name] = cycle
            logged[cycle].add(name)
    assert all(values == {"kubric", "pointodyssey", "dynamic_replica"} for values in logged.values())
