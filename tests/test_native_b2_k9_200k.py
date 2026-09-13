"""B2/A2/K9 profile: no hidden B1 execution, fixed8clip budget, native shapes."""
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from worldbridge.trainer.config import validate_config
from worldbridge.data.sampling import deterministic_sample_plan
from worldbridge.models.decoder import DenseQueryDecoder
from worldbridge.models.outputs import StructuredZ4D

CONFIG = 'configs/h031_k512_b2_a2_k9_mix50_gpu67_to200000.yaml'
PARENT = 'configs/h031_k512_k3_mix50_gpu56_to200000.yaml'

def config():
    return yaml.safe_load(Path(CONFIG).read_text())

def test_minimal_override_and_fixed_clip_budget():
    c = config(); old = yaml.safe_load(Path(PARENT).read_text())
    validate_config(c, 2); validate_config(old, 2)
    assert {k for k in c.keys() | old.keys() if c.get(k) != old.get(k)} == {
        'native_kubric512_k3_mix_200k', 'native_kubric512_k3_mix_resume',
        'native_kubric512_b2_a2_k9_mix_200k', 'microbatch_per_gpu',
        'gradient_accumulation', 'targets_per_source', 'tracking'}
    assert c['microbatch_per_gpu'] == c['gradient_accumulation'] == 2
    assert c['targets_per_source'] == 9 and 2 * 2 * 2 * 9 == 72
    assert c['max_steps'] == c['lr_restart']['end_step'] == 200000
    assert c['lr_restart'] == old['lr_restart']
    assert c['selected_checkpoint_step'] == 152768
    for name in ('kubric', 'pointodyssey', 'dynamic_replica'):
        for rank in range(2):
            plans = [deterministic_sample_plan(list(range(5737)), name, c['seed'],
                      175361, slot, rank, 4) for slot in range(4)]
            rebatch = [p for micro in range(2) for p in plans[micro*2:(micro+1)*2]]
            assert [(p[0],p[1]) for p in rebatch] == [(p[0],p[1]) for p in plans]
            assert len(rebatch) == 4  # same deterministic slots, different micro grouping

@pytest.mark.parametrize('key,value', [
    ('native_kubric512_b2_a2_k9_mix_200k', False), ('microbatch_per_gpu', 1),
    ('gradient_accumulation', 4), ('targets_per_source', 3), ('max_steps', 170000),
    ('native_kubric512_k3_mix_resume', True), ('native_kubric512_k3_mix_200k', True),
    ('native_kubric512_k5_mix_resume', True), ('native_kubric512_k11_trial', True),
    ('native_kubric512_k9_mix_170k', False), ('native_kubric512_b1_a4_k9', False)])
def test_invalid_new_profile_rejected(key, value):
    c = config(); c[key] = value
    with pytest.raises(ValueError): validate_config(c, 2)

@pytest.mark.parametrize('resolution', [256,512])
def test_b2_k9_forward_reverse_backward_native_alignment(resolution, pairs=9):
    torch.manual_seed(424242)
    net = DenseQueryDecoder(num_frames=21, latent_shape=(8,21,32,32), query_dim=16,
        embedding_dim=8,num_layers=0,num_heads=2,upsample_channels=(16,8,8,4),
        output_size=(256,256),query_grid_size=32,structured_motion_slots=2,
        structured_local_queries=True,source_rgb_pyramid=True,source_rgb_channels=(4,4,8),
        source_rgb_fusion_32=True,pre_attention_rgb_query=True,native_512=True)
    # Fresh RGB gates are exactly zero by design. Emulate learned nonzero gates
    # only in this synthetic fixture to test both clips' RGB gradient paths.
    with torch.no_grad():
        for fusion in net.upsampler.source_fusions.values():
            fusion.alpha.fill_(0.1)
    hw=resolution//8
    z=StructuredZ4D(torch.randn(2,8,21,hw,hw),torch.randn(2,21,2,8))
    rgb=torch.randn(2,3,resolution,resolution,requires_grad=True)
    source=torch.tensor([[0]*pairs,[20]*pairs])
    target=torch.tensor([list(range(1,pairs+1)),list(range(19,19-pairs,-1))])
    assert all(len(set(row.tolist())) == pairs for row in target)
    pred=net(z,source,target,source_rgb=rgb).normalized_xyz
    assert pred.shape == (2,pairs,3,resolution,resolution)
    reverse=net(z,target[:,:1],source[:,:1],source_rgb=rgb).normalized_xyz
    assert reverse.shape == (2,1,3,resolution,resolution)
    (pred.square().mean()+reverse.square().mean()).backward()
    assert rgb.grad is not None and all(torch.isfinite(g).all() and g.abs().sum()>0 for g in rgb.grad)
    assert torch.isfinite(net.upsampler.xyz.weight.grad).all()
    assert net.query_coordinates.shape == (1024,2)
