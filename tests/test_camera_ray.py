from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
import yaml
from worldbridge.models.camera import SourceConditionedCameraHead, CameraOutput, quaternion_to_matrix
from worldbridge.models.outputs import StructuredZ4D, DenseQueryOutput
from worldbridge.models.worldbridge import DenseQueryWanModel
from worldbridge.data.sampling import sample_eligible_targets, source_with_eligible_targets
from worldbridge.trainer.camera_objective import (unit_source_rays, diagonal_ray_loss, relative_pose_gt,
    camera_supervision_losses, camera_ray_objective, validate_supervision_camera)
from worldbridge.trainer.config import validate_config
from worldbridge.trainer.optimizer import parameter_groups, apply_fresh_group_warmup
from worldbridge.trainer.schedulers import apply_lr_restart_schedule


def config():
    return yaml.safe_load(Path('configs/h032_camera_ray_196000_to210000.yaml').read_text())


def camera_gt(b=2,h=8,w=8):
    K=torch.eye(3).repeat(b,21,1,1)
    K[:,:,0,0]=w; K[:,:,1,1]=h
    K[:,:,0,2]=(w-1)/2; K[:,:,1,2]=(h-1)/2
    R=torch.eye(3).repeat(b,21,1,1)
    p=torch.zeros(b,21,3)
    p[:,:,0]=torch.arange(21)*0.1
    return K,R,p


def test_config_exact_generator_and_guard():
    import importlib.util
    spec=importlib.util.spec_from_file_location('generator','research/analysis/h032_make_config.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    cfg=config(); assert cfg==module.make_config(); validate_config(cfg,2)
    for key,value in [('targets_per_source',8),('coordinate_frame','anchor'),('full_mode_unfreeze_resume',True)]:
        bad=deepcopy(cfg); bad[key]=value
        with pytest.raises(ValueError): validate_config(bad,2)
    bad=deepcopy(cfg); bad['camera_supervision']['loss_weights']['ray']=float('nan')
    with pytest.raises(ValueError): validate_config(bad,2)
    bad=deepcopy(cfg); del bad['lr_restart']['group_learning_rates']['camera_head']
    with pytest.raises(ValueError): validate_config(bad,2)


def test_sampling_forces_one_diagonal_legacy_unchanged():
    valid=np.ones((21,3,4),bool)
    for source in range(21):
        out=sample_eligible_targets(valid,9,np.random.default_rng(7),diagonal_source=source)
        assert len(set(out))==9 and out[0]==source and (out==source).sum()==1
    expected=np.random.default_rng(7).choice(np.arange(21),9,replace=False)
    assert np.array_equal(expected,sample_eligible_targets(valid,9,np.random.default_rng(7)))
    valid[3]=False
    with pytest.raises(ValueError): sample_eligible_targets(valid,9,np.random.default_rng(7),diagonal_source=3)
    class Dataset:
        def source_all_targets(self,index,source): return np.zeros((21,3,3,4)),valid
    assert source_with_eligible_targets(Dataset(),0,np.array([3,4]),9,True)[0]==4


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_head_shapes_source_conditioning_intrinsics_independence_and_gradients(dtype):
    torch.manual_seed(7)
    head=SourceConditionedCameraHead(12,dim=32,num_heads=4,memory_grid=4).to(dtype=dtype)
    z=StructuredZ4D(torch.randn(2,12,21,8,6,dtype=dtype,requires_grad=True),torch.zeros(2,21,2,12,dtype=dtype))
    s=torch.tensor([0,10])
    out=head(z,s)
    assert out.rotation.shape==(2,21,3,3) and out.translation.shape==(2,21,3) and out.fov.shape==(2,2)
    assert torch.allclose(out.rotation.transpose(-1,-2)@out.rotation,torch.eye(3),atol=1e-5)
    assert torch.allclose(torch.linalg.det(out.rotation),torch.ones(2,21),atol=1e-5)
    assert torch.equal(out.rotation[torch.arange(2),s],torch.eye(3).repeat(2,1,1))
    assert torch.count_nonzero(out.translation[torch.arange(2),s])==0
    other=head(z,torch.tensor([5,3]))
    assert torch.equal(out.fov,other.fov)
    assert not torch.equal(out.translation[:,1],other.translation[:,1])
    K,R,p=camera_gt()
    losses=camera_supervision_losses(out,s,K,R,p,[1,1,1],8,8)
    sum(losses[k] for k in ('pose_rotation','pose_translation','fov')).backward()
    assert torch.isfinite(z.dense.grad).all() and z.dense.grad.abs().sum()>0
    for name,parameter in head.named_parameters():
        assert parameter.grad is not None,name
        assert torch.isfinite(parameter.grad).all(),name
    assert (out.fov>0).all() and (out.fov<torch.pi).all()


def test_production_camera_parameter_count():
    head=SourceConditionedCameraHead(512)
    assert sum(p.numel() for p in head.parameters())==1414409
    assert len(list(head.parameters()))==51


def test_relative_pose_direction_nonidentity_rotation_and_rounding():
    K,R,p=camera_gt(1)
    R[:,3]=quaternion_to_matrix(torch.tensor([0.,0.,2**-0.5,2**-0.5]))
    p[:,3]=torch.tensor([2.,3.,1.])
    source=torch.tensor([3])
    gtR,gtp=relative_pose_gt(R,p,source)
    world_point=torch.tensor([4.,5.,6.])
    target_point=R[0,7].T@(world_point-p[0,7])
    expected=R[0,3].T@(world_point-p[0,3])
    torch.testing.assert_close(gtR[0,7]@target_point+gtp[0,7],expected)
    # Small annotation nonorthogonality: use actual inverse for translation,
    # closest proper rotation for the quaternion target.
    R[:,3,0,0]+=0.0003
    rr,pp=relative_pose_gt(R,p,source)
    torch.testing.assert_close(pp[:,7],torch.linalg.solve(R[:,3],(p[:,7]-p[:,3])[...,None])[...,0])
    torch.testing.assert_close(rr.transpose(-1,-2)@rr,torch.eye(3).repeat(1,21,1,1),atol=1e-6,rtol=1e-6)


def test_ray_physical_denormalization_and_gt_principal():
    K,R,p=camera_gt(1,4,6)
    K[:,:,0,2]+=0.4
    rays=unit_source_rays(K[:,5],4,6)
    assert (rays[:,2]<0).all()
    points=4*rays
    mean=torch.tensor([0.5,0.7,-3.]).view(1,3,1,1)
    scale=torch.tensor([2.,3.,5.]).view(1,3,1,1)
    pred=((points-mean)/scale)[:,None].repeat(1,2,1,1,1).requires_grad_()
    valid=torch.ones(1,2,4,6,dtype=torch.bool)
    source=torch.tensor([[5,5]]); target=torch.tensor([[5,9]])
    args=(valid,source,target,K,mean.flatten(),scale.flatten())
    ray,front,dev=diagonal_ray_loss(pred,*args)
    assert ray<1e-10 and front==0 and dev<1e-6
    changed=pred.detach().clone(); changed[:,1]+=100 # offdiagonal must be unconstrained
    assert diagonal_ray_loss(changed,*args)[0]<1e-10
    changed=pred.detach().clone(); changed[:,0,0]+=0.2; changed.requires_grad_()
    ray,front,dev=diagonal_ray_loss(changed,*args)
    assert ray>0 and dev>0
    ray.backward(); assert changed.grad[:,1].abs().sum()==0
    back=(((-points)-mean)/scale)[:,None].repeat(1,2,1,1,1)
    assert diagonal_ray_loss(back,*args)[1]>0
    no_diag=torch.tensor([[4,9]])
    with pytest.raises(ValueError): diagonal_ray_loss(pred,valid,source,no_diag,K,mean.flatten(),scale.flatten())


def test_invalid_pixels_masked_and_balanced_objective():
    K,R,p=camera_gt(1,4,4)
    source=torch.zeros(1,3,dtype=torch.long); target=torch.tensor([[0,1,2]])
    xyz=(3*unit_source_rays(K[:,0],4,4))[:,None].repeat(1,3,1,1,1)
    valid=torch.ones(1,3,4,4,dtype=torch.bool); valid[:,:,0,0]=False
    prediction=xyz.clone(); prediction[:,:,0,1,1]+=0.2
    prediction[:,:, :,0,0]=float('nan'); prediction.requires_grad_()
    rr,pp=relative_pose_gt(R,p,source[:,0])
    fov=torch.full((1,2),2*np.arctan(.5))
    output=CameraOutput(rr,pp,fov)
    total,losses=camera_ray_objective(prediction,xyz,valid,source,target,output,(K,R,p),[0,0,0],[1,1,1],config()['camera_supervision'])
    assert torch.isfinite(total) and total>0
    total.backward(); assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[:,:,:,0,0].abs().sum()==0
    torch.testing.assert_close(losses['diagonal_xyz'],losses['offdiagonal_xyz'])


def test_calibration_guards():
    K,R,p=camera_gt(1)
    c=dict(intrinsics=K[0].numpy(),rotations=R[0].numpy(),positions=p[0].numpy())
    validate_supervision_camera(c,8,8)
    c2=deepcopy(c); c2['intrinsics'][5,0,0]*=1.01
    with pytest.raises(ValueError,match='clip-shared'): validate_supervision_camera(c2,8,8)
    c2=deepcopy(c); c2['intrinsics'][:,0,2]+=3
    with pytest.raises(ValueError,match='centered'): validate_supervision_camera(c2,8,8)
    c2=deepcopy(c); c2['rotations'][:,0,0]+=0.0003
    validate_supervision_camera(c2,8,8)
    c2['rotations'][:,0,0]+=0.1
    with pytest.raises(ValueError,match='near SO'): validate_supervision_camera(c2,8,8)


def test_fresh_camera_warmup_anchored_to_phase_not_resume():
    cfg=config(); parameters=[nn.Parameter(torch.ones(1)) for _ in range(6)]
    optimizer=torch.optim.AdamW([dict(params=[p],name=name,lr=lr) for p,(name,lr) in zip(parameters,cfg['lr_restart']['group_learning_rates'].items())])
    for update in (196001,196021,196500,200000):
        apply_lr_restart_schedule(optimizer,update,cfg['lr_restart'])
        apply_fresh_group_warmup(optimizer,{'camera_head'},update,196000,500)
        actual={g['name']:g['lr'] for g in optimizer.param_groups}
        assert actual['camera_head']==pytest.approx(1e-4*min((update-196000)/500,1))
        assert actual['wan_backbone']==5e-7 and actual['dense_decoder']==3e-6


def test_model_forward_attaches_camera_and_legacy_state_unchanged():
    class Backbone(nn.Module):
        def __init__(self): super().__init__(); self.proj=nn.Linear(4,4)
        def forward(self,x):
            return StructuredZ4D(self.proj(x.permute(0,2,3,4,1)).permute(0,4,1,2,3),x.new_zeros(x.shape[0],21,1,4))
    class Decoder(nn.Module):
        def __init__(self): super().__init__(); self.proj=nn.Linear(4,3)
        def forward(self,z,source,target,source_rgb=None):
            xyz=self.proj(z.dense.mean((2,3,4)))[:,None,:,None,None].expand(-1,source.shape[1],-1,4,4)
            return DenseQueryOutput(xyz,z.dense)
    old=DenseQueryWanModel(Backbone(),Decoder()); old_state=deepcopy(old.state_dict())
    model=DenseQueryWanModel(Backbone(),Decoder(),SourceConditionedCameraHead(4,32,4,memory_grid=2))
    missing,unexpected=model.load_state_dict(old_state,strict=False)
    assert missing and all(k.startswith('camera_head.') for k in missing) and not unexpected
    for k,v in old_state.items(): assert torch.equal(model.state_dict()[k],v)
    src=torch.tensor([[2,2]]); tgt=torch.tensor([[2,6]]); latent=torch.randn(1,4,21,4,4)
    prediction,z,out=model(latent,src,tgt)
    assert out.camera is not None
    torch.testing.assert_close(prediction,old(latent,src,tgt)[0])
    assert model(latent,src,tgt,z4d_override=z)[2].camera is None
    cfg={'learning_rate':3e-6,'weight_decay':1e-4,'camera_supervision':{'learning_rate':1e-4}}
    groups=parameter_groups(model,cfg)
    names={g['name'] for g in groups}; assert 'camera_head' in names
    assert sum(len(g['params']) for g in groups)==len(list(model.parameters()))
