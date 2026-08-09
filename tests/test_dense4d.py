from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from worldbridge.data import MOViSample
from worldbridge.dense4d import (
    CleanLatentBackbone, DenseQueryDecoder, DenseQueryWanModel, FeedForwardWanBackbone, RotaryEmbedding2D,
    StructuredZ4D, WanHiddenGeometryBackbone, flatten_structured_z4d, flatten_z4d,
    masked_pair_smooth_l1, unflatten_z4d, verify_flow_velocity_algebra,
)
from worldbridge.dense4d_data import CoordinateStats, dense_pair_targets, sample_dense_pairs
from worldbridge.geometry import GeometryBuilder
from worldbridge.pointmap import build_dynamic_pointmap
from worldbridge.wan import WAN_LATENT_SHAPE, WanDiTMapping, rgb_to_wan_input


def synthetic_sample(background=True):
    time, height, width = 3, 2, 2
    segmentation = np.zeros((time, height, width), np.int64) if background else np.ones((time, height, width), np.int64)
    instances = 0 if background else 1
    positions = np.zeros((instances, time, 3), np.float32)
    if instances:
        positions[0, :, 0] = [0, 0.5, 1.0]
    quaternions = np.zeros((instances, time, 4), np.float32)
    if instances:
        quaternions[..., 0] = 1
    return MOViSample(
        "synthetic", np.zeros((time, height, width, 3), np.uint8),
        np.full((time, height, width), 2, np.float32), np.ones((time, height, width), bool), segmentation,
        np.zeros((time, 3), np.float32), np.tile(np.array([1, 0, 0, 0], np.float32), (time, 1)),
        35.0, 32.0, 0.8, positions, quaternions, np.ones(instances, bool),
        np.ones((instances, time), np.uint16), np.array([1, 3], np.float32), 0,
    )


def small_decoder(coarse=False):
    return DenseQueryDecoder(
        query_dim=32, embedding_dim=16, num_layers=1, num_heads=4,
        upsample_channels=(32, 16, 8, 4), coarse_diagnostic=coarse,
    )


def test_geometry_reprojection_and_anchor_world_direction():
    sample = synthetic_sample(True)
    geometry = GeometryBuilder(sample)
    pointmap, valid = geometry.pointmaps()
    uv, _, radial = geometry.camera.project(
        geometry.anchor_to_world(pointmap[0]), sample.camera_positions[0], sample.camera_quaternions[0]
    )
    vv, uu = np.meshgrid(np.arange(sample.height), np.arange(sample.width), indexing="ij")
    np.testing.assert_allclose(uv, np.stack((uu, vv), -1), atol=1e-6)
    np.testing.assert_allclose(radial[valid[0]], sample.depth[0][valid[0]], atol=1e-6)


def test_diagonal_pointmap_and_dynamic_transform():
    sample = synthetic_sample(False)
    dynamic = build_dynamic_pointmap(sample, source=1)
    pointmap, _ = GeometryBuilder(sample).pointmaps()
    np.testing.assert_allclose(dynamic.xyz[1], pointmap[1], atol=1e-6)
    np.testing.assert_allclose(dynamic.xyz[2, ..., 0] - dynamic.xyz[1, ..., 0], 0.5, atol=1e-6)


def test_source_frame_trajectory_is_rigid_change_and_diagonal_is_local_depth():
    sample = synthetic_sample(False)
    sample.camera_positions[:] = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], np.float32)
    angle = np.pi / 3
    sample.camera_quaternions[1] = np.array(
        [np.cos(angle / 2), 0, 0, np.sin(angle / 2)], np.float32
    )
    geometry = GeometryBuilder(sample)
    anchor = build_dynamic_pointmap(sample, source=1, coordinate_frame="anchor")
    local = build_dynamic_pointmap(sample, source=1, coordinate_frame="source")
    expected = geometry.anchor_to_source(anchor.xyz, source=1)
    np.testing.assert_allclose(local.xyz, expected, atol=1e-5)
    np.testing.assert_array_equal(local.visible, anchor.visible)
    np.testing.assert_array_equal(local.valid, anchor.valid)

    world_diagonal = geometry.camera.backproject(
        sample.depth[1], sample.camera_positions[1], sample.camera_quaternions[1]
    )
    expected_diagonal = geometry.world_to_camera_frame(world_diagonal, frame=1)
    np.testing.assert_allclose(local.xyz[1], expected_diagonal, atol=1e-5)


def test_visibility_is_not_validity_and_loss_uses_validity():
    sample = synthetic_sample(False)
    sample.segmentation[2] = 0
    dynamic = build_dynamic_pointmap(sample, source=0)
    assert dynamic.valid[2].all() and not dynamic.visible[2].any()
    prediction = torch.zeros(1, 1, 3, 2, 2)
    target = torch.ones_like(prediction)
    assert masked_pair_smooth_l1(prediction, target, torch.from_numpy(dynamic.valid[2])[None, None]) > 0


def test_wan_input_layout_and_native_clean_latent_contract():
    rgb = torch.zeros(1, 21, 3, 128, 128, dtype=torch.uint8)
    assert rgb_to_wan_input(rgb).shape == (1, 3, 21, 128, 128)
    assert WAN_LATENT_SHAPE == (16, 6, 16, 16)


def test_flow_velocity_sign_from_executable_scheduler():
    result = verify_flow_velocity_algebra()
    assert result["raw_velocity"] == "noise-clean"
    assert result["perception_readout"] == "negative_raw_velocity"
    assert result["maximum_clean_recovery_error"] == 0


class RecordingMapping(nn.Module):
    def __init__(self):
        super().__init__()
        self.dit = nn.Conv3d(16, 16, 1)
        self.last_tau = None

    def forward(self, latent, tau):
        self.last_tau = tau.detach().clone()
        return self.dit(latent)


def test_feedforward_backbone_uses_exact_zero_and_negates():
    mapping = RecordingMapping()
    backbone = FeedForwardWanBackbone(mapping)
    latent = torch.randn(2, *WAN_LATENT_SHAPE)
    expected = -mapping.dit(latent)
    actual = backbone(latent)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(mapping.last_tau, torch.zeros(2))
    assert actual.shape == latent.shape


def test_wan_hidden_extraction_matches_patch_grid_and_never_calls_output_head():
    from diffusers import WanTransformer3DModel

    mapping = WanDiTMapping.__new__(WanDiTMapping)
    nn.Module.__init__(mapping)
    mapping.checkpoint = "synthetic"
    mapping.timestep_scale = 1000.0
    mapping.dit = WanTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=8,
        in_channels=16, out_channels=16, text_dim=32, freq_dim=16,
        ffn_dim=32, num_layers=2, cross_attn_norm=True, rope_max_seq_len=32,
    )
    mapping.register_buffer("empty_condition", torch.zeros(1, 512, 32), persistent=False)
    calls = {"norm": 0, "projection": 0}
    norm_hook = mapping.dit.norm_out.register_forward_hook(
        lambda *_: calls.__setitem__("norm", calls["norm"] + 1)
    )
    projection_hook = mapping.dit.proj_out.register_forward_hook(
        lambda *_: calls.__setitem__("projection", calls["projection"] + 1)
    )
    hidden, grid = mapping.forward_hidden_layers(
        torch.randn(1, *WAN_LATENT_SHAPE), torch.zeros(1), (0, 1)
    )
    norm_hook.remove(); projection_hook.remove()
    assert grid == (6, 8, 8)
    assert len(hidden) == 2
    assert hidden[0].shape == hidden[1].shape == (1, 6 * 8 * 8, 16)
    assert calls == {"norm": 0, "projection": 0}


def test_source_and_target_embeddings_are_independent_and_asymmetric():
    decoder = small_decoder()
    assert decoder.source_embedding is not decoder.target_embedding
    assert decoder.source_embedding.weight.data_ptr() != decoder.target_embedding.weight.data_ptr()
    forward = decoder.query_content(torch.tensor([[3]]), torch.tensor([[18]]))
    reverse = decoder.query_content(torch.tensor([[18]]), torch.tensor([[3]]))
    assert not torch.equal(forward, reverse)


def test_2d_rope_distinguishes_broadcast_spatial_queries():
    rope = RotaryEmbedding2D(8)
    content = torch.randn(1, 1, 1, 1, 8)
    coordinates = torch.tensor([[0, 0], [3, 5]])
    rotated = rope(content.expand(-1, -1, -1, 2, -1), coordinates)
    assert not torch.equal(rotated[..., 0, :], rotated[..., 1, :])


def test_memory_flatten_unflatten_and_temporal_spatial_coordinates():
    z4d = torch.arange(1 * 2 * 3 * 2 * 2).reshape(1, 2, 3, 2, 2).float()
    memory, coordinates = flatten_z4d(z4d)
    restored = unflatten_z4d(memory, 3, 2, 2)
    torch.testing.assert_close(restored, z4d)
    torch.testing.assert_close(coordinates[:4], coordinates[4:8])
    assert memory.shape == (1, 12, 2)


def test_clean_latent_control_and_fullres_coordinate_upsampler():
    latent = torch.randn(1, *WAN_LATENT_SHAPE)
    torch.testing.assert_close(CleanLatentBackbone()(latent), latent)
    decoder = DenseQueryDecoder(
        query_dim=32, embedding_dim=16, num_layers=1, num_heads=4,
        upsample_channels=(32, 16, 8, 4), fullres_coordinates=True,
    )
    output = decoder(latent, torch.tensor([[0]]), torch.tensor([[1]]))
    assert output.normalized_xyz.shape == (1, 1, 3, 128, 128)
    assert decoder.upsampler.xyz.in_channels == 6
    query32 = DenseQueryDecoder(
        query_dim=32, embedding_dim=16, num_layers=1, num_heads=4,
        upsample_channels=(32, 16, 8), query_grid_size=32,
    )
    query32_output = query32(latent, torch.tensor([[0]]), torch.tensor([[1]]))
    assert query32_output.low_resolution_feature.shape == (1, 1, 32, 32, 32)
    assert query32_output.normalized_xyz.shape == (1, 1, 3, 128, 128)


def test_cross_attention_and_upsampler_multiple_pairs_shape():
    decoder = small_decoder(coarse=True)
    z4d = torch.randn(2, *WAN_LATENT_SHAPE)
    output = decoder(z4d, torch.tensor([[0, 7, 20], [3, 5, 9]]), torch.tensor([[0, 20, 0], [8, 5, 1]]))
    assert output.low_resolution_feature.shape == (2, 3, 32, 16, 16)
    assert output.coarse_normalized_xyz.shape == (2, 3, 3, 16, 16)
    assert output.normalized_xyz.shape == (2, 3, 3, 128, 128)


def test_upsampler_has_no_transposed_convolution_or_batchnorm():
    decoder = small_decoder()
    forbidden = (nn.ConvTranspose2d, nn.BatchNorm2d)
    assert not any(isinstance(module, forbidden) for module in decoder.upsampler.modules())


class TinyFinalBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.dit = nn.Conv3d(16, 16, 1)

    def forward(self, latent):
        return -self.dit(latent)


class TinyHiddenMapping(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_projection = nn.Linear(16, 64)
        self.dit = nn.Module()
        self.dit.config = SimpleNamespace(
            num_attention_heads=4, attention_head_dim=16, patch_size=(1, 2, 2),
        )
        self.dit.norm_out = nn.LayerNorm(64)
        self.dit.proj_out = nn.Linear(64, 64)
        self.dit.scale_shift_table = nn.Parameter(torch.randn(1, 2, 64))
        self.final_output_called = False

    def forward(self, latent, tau):
        self.final_output_called = True
        raise AssertionError("structured geometry must bypass the Wan output head")

    def forward_hidden_layers(self, latent, tau, layers):
        pooled = F.avg_pool3d(latent, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        tokens = pooled.permute(0, 2, 3, 4, 1).reshape(latent.shape[0], 6 * 8 * 8, 16)
        hidden = self.hidden_projection(tokens)
        return tuple(hidden * (index + 1) for index in layers), (6, 8, 8)


def test_structured_hidden_readout_bypasses_final_head_and_keeps_st_query_contract():
    mapping = TinyHiddenMapping()
    backbone = WanHiddenGeometryBackbone(
        mapping, hidden_layers=(0, 1), geometry_dim=32, num_frames=21,
        spatial_size=16, motion_slots=4, num_heads=4,
    )
    decoder = DenseQueryDecoder(
        latent_shape=(32, 21, 16, 16), query_dim=32, embedding_dim=16,
        num_layers=1, num_heads=4, upsample_channels=(32, 16, 8, 4),
        structured_motion_slots=4, structured_local_queries=True,
    )
    model = DenseQueryWanModel(backbone, decoder)
    clean = torch.randn(1, *WAN_LATENT_SHAPE)
    prediction, z4d, _ = model(clean, torch.tensor([[0, 20]]), torch.tensor([[20, 0]]))
    assert isinstance(z4d, StructuredZ4D)
    assert z4d.dense.shape == (1, 32, 21, 16, 16)
    assert z4d.motion.shape == (1, 21, 4, 32)
    memory, coordinates = flatten_structured_z4d(z4d)
    assert memory.shape == (1, 21 * 16 * 16 + 21 * 4, 32)
    assert coordinates.shape == (memory.shape[1], 2)
    dense_only_memory, dense_only_coordinates = flatten_structured_z4d(
        StructuredZ4D(z4d.dense, z4d.motion, include_motion=False)
    )
    assert dense_only_memory.shape == (1, 21 * 16 * 16, 32)
    assert dense_only_coordinates.shape == (dense_only_memory.shape[1], 2)
    assert prediction.shape == (1, 2, 3, 128, 128)
    loss = prediction.square().mean()
    loss.backward()
    assert mapping.hidden_projection.weight.grad is not None
    assert backbone.temporal_logits.grad is not None
    assert decoder.upsampler.xyz.weight.grad is not None
    assert mapping.dit.proj_out.weight.grad is None
    assert not mapping.final_output_called


def test_pair_conditioned_motion_query_preserves_shared_initialization_and_gradients():
    kwargs = dict(
        latent_shape=(32, 21, 16, 16), query_dim=32, embedding_dim=16,
        num_layers=1, num_heads=4, upsample_channels=(32, 16, 8, 4),
        structured_motion_slots=4, structured_local_queries=True,
    )
    torch.manual_seed(17)
    baseline = DenseQueryDecoder(**kwargs)
    torch.manual_seed(17)
    conditioned = DenseQueryDecoder(**kwargs, structured_pair_motion_queries=True)
    torch.testing.assert_close(baseline.source_embedding.weight, conditioned.source_embedding.weight)
    torch.testing.assert_close(baseline.upsampler.xyz.weight, conditioned.upsampler.xyz.weight)
    torch.manual_seed(17)
    zero_init = DenseQueryDecoder(**kwargs, structured_pair_motion_queries=True,
                                  structured_pair_motion_zero_init=True)
    assert zero_init.motion_pair_projection[-1].weight.abs().sum() == 0
    assert zero_init.motion_pair_projection[-1].bias.abs().sum() == 0

    z4d = StructuredZ4D(
        torch.randn(1, 32, 21, 16, 16),
        torch.randn(1, 21, 4, 32, requires_grad=True),
    )
    output = conditioned(z4d, torch.tensor([[0, 7]]), torch.tensor([[20, 14]]))
    assert output.normalized_xyz.shape == (1, 2, 3, 128, 128)
    output.normalized_xyz.square().mean().backward()
    assert conditioned.motion_pair_projection[-1].weight.grad is not None
    assert z4d.motion.grad is not None


def test_dense_only_structured_control_keeps_local_query_without_slot_parameters():
    mapping = TinyHiddenMapping()
    backbone = WanHiddenGeometryBackbone(
        mapping, hidden_layers=(0, 1), geometry_dim=32, num_frames=21,
        spatial_size=16, motion_slots=0, num_heads=4,
    )
    decoder = DenseQueryDecoder(
        latent_shape=(32, 21, 16, 16), query_dim=32, embedding_dim=16,
        num_layers=1, num_heads=4, upsample_channels=(32, 16, 8, 4),
        structured_motion_slots=0, structured_local_queries=True,
    )
    model = DenseQueryWanModel(backbone, decoder)
    prediction, z4d, _ = model(
        torch.randn(1, *WAN_LATENT_SHAPE), torch.tensor([[0, 20]]), torch.tensor([[20, 0]])
    )
    assert z4d.motion.shape == (1, 21, 0, 32)
    assert backbone.motion_attention is None
    assert backbone.motion_slot_embedding is None
    memory, coordinates = flatten_structured_z4d(z4d)
    assert memory.shape == (1, 21 * 16 * 16, 32)
    assert coordinates.shape == (memory.shape[1], 2)
    assert prediction.shape == (1, 2, 3, 128, 128)
    prediction.square().mean().backward()
    assert backbone.temporal_logits.grad is not None
    assert decoder.source_local_projection.weight.grad is not None


def test_structured_layer_gates_support_entropy_and_straight_through_topk():
    mapping = TinyHiddenMapping()
    backbone = WanHiddenGeometryBackbone(
        mapping, hidden_layers=(0, 1, 2), geometry_dim=32, motion_slots=4, num_heads=4,
        layer_gate_temperature=0.7, layer_gate_top_k=2,
    )
    backbone.layer_logits.data.copy_(torch.tensor([0.1, 0.8, -0.2]))
    soft = backbone.soft_layer_weights()
    torch.testing.assert_close(soft.sum(), torch.tensor(1.0))
    entropy = backbone.layer_gate_entropy()
    entropy.backward()
    assert backbone.layer_logits.grad is not None
    backbone.eval()
    hard = backbone.layer_weights()
    assert int((hard > 0).sum()) == 2
    torch.testing.assert_close(hard.sum(), torch.tensor(1.0))


def test_layer_gate_initialization_is_seeded_and_nonuniform():
    first = WanHiddenGeometryBackbone(
        TinyHiddenMapping(), hidden_layers=(0, 1, 2), geometry_dim=32,
        motion_slots=4, num_heads=4, layer_gate_init_std=0.01, layer_gate_seed=7,
    )
    second = WanHiddenGeometryBackbone(
        TinyHiddenMapping(), hidden_layers=(0, 1, 2), geometry_dim=32,
        motion_slots=4, num_heads=4, layer_gate_init_std=0.01, layer_gate_seed=7,
    )
    torch.testing.assert_close(first.layer_logits, second.layer_logits)
    assert not torch.equal(first.layer_logits, torch.zeros_like(first.layer_logits))


def test_geometry_adapter_mode_freezes_wan_but_trains_structured_adapter_and_decoder():
    mapping = TinyHiddenMapping()
    backbone = WanHiddenGeometryBackbone(
        mapping, hidden_layers=(0,), geometry_dim=32, motion_slots=4, num_heads=4,
    )
    decoder = DenseQueryDecoder(
        latent_shape=(32, 21, 16, 16), query_dim=32, embedding_dim=16,
        num_layers=1, num_heads=4, upsample_channels=(32, 16, 8, 4),
        structured_motion_slots=4, structured_local_queries=True,
    )
    model = DenseQueryWanModel(backbone, decoder)
    model.configure_trainable("geometry_adapter")
    assert not mapping.hidden_projection.weight.requires_grad
    assert not mapping.dit.proj_out.weight.requires_grad
    assert backbone.temporal_logits.requires_grad
    assert decoder.upsampler.xyz.weight.requires_grad


def test_xyz_gradient_reaches_backbone_and_decoder():
    model = DenseQueryWanModel(TinyFinalBackbone(), small_decoder())
    latent = torch.randn(1, *WAN_LATENT_SHAPE)
    prediction, z4d, _ = model(latent, torch.tensor([[0, 20]]), torch.tensor([[20, 0]]))
    loss = masked_pair_smooth_l1(prediction, torch.randn_like(prediction), torch.ones(1, 2, 128, 128, dtype=torch.bool))
    loss.backward()
    assert model.backbone.dit.weight.grad is not None
    assert model.decoder.upsampler.xyz.weight.grad is not None
    assert z4d.shape == latent.shape


def test_checkpoint_save_load(tmp_path):
    model = DenseQueryWanModel(TinyFinalBackbone(), small_decoder())
    path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), path)
    clone = DenseQueryWanModel(TinyFinalBackbone(), small_decoder())
    clone.load_state_dict(torch.load(path, weights_only=True))
    for first, second in zip(model.parameters(), clone.parameters()):
        torch.testing.assert_close(first, second)


def test_pair_sampler_balances_diagonal_directions_and_gaps():
    source, target = sample_dense_pairs(21, 13, np.random.default_rng(5))
    assert np.any(source == target)
    assert np.any(target > source) and np.any(target < source)
    gap = np.abs(target - source)
    assert np.any((gap > 0) & (gap <= 5)) and np.any(gap >= 10)
    assert np.any(source > 0)


def test_dense_pair_targets_are_source_grid_maps_and_normalized():
    sample = synthetic_sample(False)
    stats = CoordinateStats(np.array([1, 2, 3]), np.array([2, 4, 8]))
    normalized, metric, visible, valid = dense_pair_targets(sample, [1, 1], [1, 2], stats)
    np.testing.assert_allclose(normalized, (metric - stats.mean[None, :, None, None]) / stats.scale[None, :, None, None])
    assert normalized.shape == metric.shape == (2, 3, 2, 2)
    assert visible.shape == valid.shape == (2, 2, 2)
