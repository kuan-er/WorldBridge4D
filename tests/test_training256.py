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
    CachedExternalDataset, LazyLatentCache, apply_cosine_schedule, dataset_for_step,
    deterministic_dataset_schedule, deterministic_sample_plan, sample_eligible_targets,
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


def test_k21_is_permitted_only_for_two_gpu_gate():
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
    with pytest.raises(ValueError, match="only the two-GPU gate"):
        train.validate_config(batched, world=4, allow_two_gpu=False)
    with pytest.raises(ValueError, match="gate-only"):
        train.validate_config(base, world=4, allow_two_gpu=False)


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
    all_targets = sample_eligible_targets(valid, 6, np.random.default_rng(8))
    assert set(all_targets) == {0, 3, 9, 12, 20}


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
