import torch
from worldbridge.models import QueryDecoder, WorldLatentModel


def test_exact_latent_shapes_for_both_variants():
    torch.manual_seed(0); B,T,H,W=1,5,32,32
    compact=WorldLatentModel("compact",latent_channels=7,latent_time=3,latent_height=4,latent_width=5,
                             trajectory_channels=8,trajectory_hidden=12,reconstruction_channels=8,context_hidden=12,decoder_hidden=16)
    p=torch.randn(B,T,3,H,W); v=torch.ones(B,T,H,W,dtype=torch.bool); x=torch.randn(B,T,H,W,3)
    z=compact(pointmaps=p,point_valid=v,anchor_values=x,anchor_visible=v,anchor_valid=v,source_length=T)
    assert z.shape==(B,7,3,4,5)
    full=WorldLatentModel("full",latent_channels=7,latent_time=3,latent_height=4,latent_width=5,
                         trajectory_channels=8,trajectory_hidden=12,context_hidden=12,decoder_hidden=16)
    zf=full(trajectory_features=torch.randn(B,T,H,W,8),source_length=T)
    assert zf.shape==(B,7,3,4,5)


def test_original_coordinate_to_latent_mapping():
    # Latent value is affine in normalized x/y/source time; trilinear sampling must preserve it.
    D,H,W=3,4,5
    tt=torch.linspace(0,1,D)[:,None,None]; yy=torch.linspace(0,1,H)[None,:,None]; xx=torch.linspace(0,1,W)[None,None,:]
    z=(100*tt+10*yy+xx)[None,None]
    s=torch.tensor([[0.,2.,4.]]) # original source length 5 -> normalized 0,.5,1
    uv=torch.tensor([[[0.,0.],[.5,.5],[1.,1.]]])
    sampled=QueryDecoder.sample_latent(z,s,uv,source_length=5)[0,:,0]
    torch.testing.assert_close(sampled,torch.tensor([0.,55.5,111.]))


def test_decoder_gradients_reach_latent_and_query_coordinates():
    dec=QueryDecoder(latent_channels=4,hidden=16)
    z=torch.randn(1,4,3,4,5,requires_grad=True); uv=torch.tensor([[[.3,.4],[.7,.2]]],requires_grad=True)
    pred=dec(z,torch.tensor([[1.,2.]]),uv,torch.tensor([[3.,0.]]),source_length=5)
    pred.square().mean().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert uv.grad is not None and torch.isfinite(uv.grad).all() and uv.grad.abs().sum()>0
