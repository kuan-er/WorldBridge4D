import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from worldbridge.data import MOViSample
from worldbridge.dense4d import (
    CleanLatentBackbone, DenseQueryDecoder, DenseQueryWanModel, FeedForwardWanBackbone,
    RotaryEmbedding2D, RotaryEmbedding3D, flatten_z4d, masked_pair_smooth_l1,
    masked_visibility_bce, unflatten_z4d, verify_flow_velocity_algebra,
)
from worldbridge.dense4d_data import CoordinateStats, dense_pair_targets, sample_dense_pairs
from worldbridge.dense4d_prefetch import make_source_centric_plan, source_centric_loss_weights
from worldbridge.geometry import GeometryBuilder
from worldbridge.pointmap import build_dynamic_pointmap
from worldbridge.wan import WAN_LATENT_SHAPE, rgb_to_wan_input


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


def test_3d_rope_uses_time_uv_and_preserves_eight_dimensions():
    rope = RotaryEmbedding3D(32)
    values = torch.randn(1, 2, 4, 1, 32)
    coordinates = torch.tensor([[[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [5.0, 4.0, 1.0], [2.5, 1.0, 6.0]],
                                 [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [5.0, 4.0, 1.0], [2.5, 1.0, 6.0]]]])
    rotated = rope(values, coordinates)
    torch.testing.assert_close(rotated[..., 24:], values[..., 24:])
    assert not torch.equal(rotated[..., :24], values[..., :24])


def test_3d_rope_decoder_and_visibility_head_shapes():
    decoder = DenseQueryDecoder(
        query_dim=32, embedding_dim=16, num_layers=1, num_heads=1,
        upsample_channels=(32, 16, 8, 4), rope_mode="3d", visibility_head=True,
    )
    output = decoder(torch.randn(1, *WAN_LATENT_SHAPE), torch.tensor([[0, 7]]), torch.tensor([[1, 20]]))
    assert output.normalized_xyz.shape == (1, 2, 3, 128, 128)
    assert output.visibility_logits is not None
    assert output.visibility_logits.shape == (1, 2, 1, 128, 128)


def test_source_centric_plan_rotates_sources_without_worker_rng():
    first = make_source_centric_plan(5, 5, 21, 0)
    second = make_source_centric_plan(5, 5, 21, 1)
    third = make_source_centric_plan(5, 5, 21, 5)
    np.testing.assert_array_equal(first.sample_indices, second.sample_indices)
    np.testing.assert_array_equal(first.source[:, 0] + 1, second.source[:, 0])
    np.testing.assert_array_equal(first.source[:, 0], np.arange(5))
    np.testing.assert_array_equal(third.source[:, 0], np.arange(5) + 5)
    np.testing.assert_array_equal(first.target[0], np.arange(21))
    weights = source_centric_loss_weights(first.source, first.target)
    np.testing.assert_allclose(weights.sum(1), 1.0, atol=2e-7)
    np.testing.assert_allclose(weights[np.arange(5), first.source[:, 0]], 1.0 / 3.0)
    np.testing.assert_allclose(weights[0, 1:], (2.0 / 3.0) / 20.0)


def test_visibility_loss_masks_only_off_diagonal_validity():
    logits = torch.zeros(1, 2, 1, 2, 2)
    visible = torch.tensor([[[[1, 0], [1, 0]], [[0, 1], [0, 1]]]], dtype=torch.bool)
    valid = torch.tensor([[[[1, 1], [1, 1]], [[1, 1], [0, 1]]]], dtype=torch.bool)
    source = torch.tensor([[2, 3]])
    target = torch.tensor([[2, 1]])
    loss = masked_visibility_bce(logits, visible, valid, source, target, 1.0)
    expected = torch.tensor(float(F.binary_cross_entropy_with_logits(
        logits[0, 1, 0][valid[0, 1]], visible[0, 1][valid[0, 1]].float(), reduction="mean"
    )))
    torch.testing.assert_close(loss, expected)


def test_dense_pair_targets_are_source_grid_maps_and_normalized():
    sample = synthetic_sample(False)
    stats = CoordinateStats(np.array([1, 2, 3]), np.array([2, 4, 8]))
    normalized, metric, visible, valid = dense_pair_targets(sample, [1, 1], [1, 2], stats)
    np.testing.assert_allclose(normalized, (metric - stats.mean[None, :, None, None]) / stats.scale[None, :, None, None])
    assert normalized.shape == metric.shape == (2, 3, 2, 2)
    assert visible.shape == valid.shape == (2, 2, 2)
