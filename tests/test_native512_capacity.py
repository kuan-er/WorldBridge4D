from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
import yaml

from worldbridge.models.decoder import DenseQueryDecoder
from worldbridge.models.backbones.geometry import WanHiddenGeometryBackbone
from worldbridge.models.outputs import StructuredZ4D
from worldbridge.trainer.config import validate_config


def config():
    return yaml.safe_load(Path('configs/h031_k512_po256_dr256_b1_a4_k15_capacity.yaml').read_text())


def test_capacity_config_and_legacy_control():
    validate_config(config(), 2)
    old = yaml.safe_load(Path('configs/h030_152500_to_155k_gpu23_b2_k15_fp32_cycle0_rgb1x.yaml').read_text())
    validate_config(old, 2)


@pytest.mark.parametrize('key,value', [('targets_per_source',7),('microbatch_per_gpu',2),
    ('gradient_accumulation',2),('max_steps',155000),('native_capacity_test_only',False),
    ('source_edge_contrast_weight',0.01),('cycle_b2_a2_k15',True),('image_size',512)])
def test_invalid_capacity_protocol(key, value):
    cfg = config(); cfg[key] = value
    with pytest.raises(ValueError): validate_config(cfg, 2)


def decoder(native):
    return DenseQueryDecoder(num_frames=3, latent_shape=(8,3,32,32), query_dim=16,
        embedding_dim=8, num_layers=0, num_heads=2, upsample_channels=(16,8,8,4),
        output_size=(256,256), query_grid_size=32, structured_motion_slots=2,
        structured_local_queries=True, source_rgb_pyramid=True, source_rgb_channels=(4,4,8),
        source_rgb_fusion_32=True, pre_attention_rgb_query=True, native_512=native)


def test_decoder_same_parameters_exact256_and_native512_backward():
    torch.manual_seed(42)
    old = decoder(False); native = decoder(True)
    native.load_state_dict(old.state_dict(), strict=True)
    assert {k:tuple(v.shape) for k,v in old.state_dict().items()} == {k:tuple(v.shape) for k,v in native.state_dict().items()}
    source = torch.tensor([[0]]); target = torch.tensor([[1]])
    z = StructuredZ4D(torch.randn(1,8,3,32,32), torch.randn(1,3,2,8))
    rgb = torch.randn(1,3,256,256)
    with torch.no_grad():
        a = old(z,source,target,source_rgb=rgb).normalized_xyz
        b = native(z,source,target,source_rgb=rgb).normalized_xyz
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    z512 = StructuredZ4D(torch.randn(1,8,3,64,64), torch.randn(1,3,2,8))
    output = native(z512,source,target,source_rgb=torch.randn(1,3,512,512))
    assert output.normalized_xyz.shape == (1,1,3,512,512)
    assert native.query_coordinates.shape == (1024,2)  # no runtime mutation of checkpoint base grid
    output.normalized_xyz.square().mean().backward()
    assert native.upsampler.xyz.weight.grad is not None
    assert torch.isfinite(native.upsampler.xyz.weight.grad).all()
    with torch.no_grad():
        c = native(z,source,target,source_rgb=rgb).normalized_xyz
    torch.testing.assert_close(a,c,rtol=0,atol=0)
    with pytest.raises(ValueError): old(z512,source,target,source_rgb=torch.randn(1,3,512,512))
    with pytest.raises((ValueError,RuntimeError)): native(z512,source,target,source_rgb=rgb)


class FakeMapping(nn.Module):
    def __init__(self):
        super().__init__()
        self.dit = nn.Module()
        self.dit.config = SimpleNamespace(num_attention_heads=2,attention_head_dim=4,patch_size=(1,2,2))
        self.dit.norm_out = nn.Identity(); self.dit.proj_out = nn.Identity()
        self.dit.scale_shift_table = nn.Parameter(torch.zeros(1))
    def forward_hidden_layers(self, latent, time, layers):
        hw = latent.shape[-1] // 2
        feature = latent[:, :8, :, ::2, ::2].flatten(2).transpose(1,2)
        return [feature for _ in layers], (6,hw,hw)


def test_native_geometry_uses64_grid_and_retains256_function():
    kwargs = dict(hidden_layers=(0,1),geometry_dim=8,num_frames=21,spatial_size=32,motion_slots=2,num_heads=2)
    old = WanHiddenGeometryBackbone(FakeMapping(), **kwargs)
    native = WanHiddenGeometryBackbone(FakeMapping(), native_512=True, **kwargs)
    native.load_state_dict(old.state_dict(),strict=True)
    with torch.no_grad():
        latent = torch.randn(1,16,6,32,32)
        torch.testing.assert_close(old(latent).dense,native(latent).dense,rtol=0,atol=0)
        result = native(torch.randn(1,16,6,64,64))
    assert result.dense.shape == (1,8,21,64,64)
    assert result.motion.shape == (1,21,2,8)


def test_clip_plans_match_b2a2_b1a4():
    from worldbridge.data.sampling import deterministic_sample_plan
    rows = list(range(5737))
    for rank in range(2):
        old = [deterministic_sample_plan(rows,'kubric',20260812,150002,slot,rank,2*2)[:2] for slot in range(4)]
        new = [deterministic_sample_plan(rows,'kubric',20260812,150002,slot,rank,1*4)[:2] for slot in range(4)]
        assert old == new
    assert 2*2*2 == 1*4*2 == 8
    assert 8*15 == 120
