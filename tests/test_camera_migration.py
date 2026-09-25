"""CPU tests of actual structural-loader whitelist and retained Adam state.

FSDP scatter is replaced only by a name->optimizer-index mapping. A real
2-rank native smoke/save/resume is still mandatory before long training.
"""
from copy import deepcopy
import numpy as np
import pytest
import torch
from torch import nn
from worldbridge.trainer import fsdp_checkpoint as ck
from worldbridge.trainer.checkpoint import capture_rng_state


class Tiny(nn.Module):
    def __init__(self, camera=False):
        super().__init__()
        self.decoder=nn.Linear(3,3)
        if camera: self.camera_head=nn.Linear(3,2)


def payload_for(model):
    opt=torch.optim.AdamW(model.parameters(),lr=3e-6)
    for _ in range(3):
        opt.zero_grad(); model.decoder(torch.ones(1,3)).sum().backward(); opt.step()
    names=[n for n,p in model.named_parameters()]
    sd=opt.state_dict()
    sd['state']={names[i]:value for i,value in sd['state'].items()}
    sd['param_groups'][0]['params']=names; sd['param_groups'][0]['name']='dense_decoder'
    rng=capture_rng_state()
    return dict(model=deepcopy(model.state_dict()),optimizer=sd,
                training_state=dict(global_step=196000,world_size=2,rng_states=[rng,rng],clips_seen={'kubric':8}))


def mock_scatter(full, model, optim):
    names=[n for g in full['param_groups'] for n in g['params']]
    indices={n:i for i,n in enumerate(names)}
    return dict(state={indices[n]:v for n,v in full['state'].items()},
                param_groups=[{**g,'params':[indices[n] for n in g['params']]} for g in full['param_groups']])


def test_camera_structural_loader_retains_all_old_weights_moments_rng_and_step(tmp_path,monkeypatch):
    torch.manual_seed(3)
    old=Tiny(); payload=payload_for(old); path=tmp_path/'old.pt'; torch.save(payload,path)
    new=Tiny(camera=True)
    monkeypatch.setattr(ck.dist,'broadcast_object_list',lambda *a,**kw: None)
    loaded,state,rng=ck.load_unwrapped_model_checkpoint(path,new,0,2,('camera_head.',))
    assert state['global_step']==196000 and len(rng)==2
    for n,v in old.state_dict().items(): assert torch.equal(new.state_dict()[n],v)
    groups=[dict(name='dense_decoder',params=list(new.decoder.parameters()),lr=3e-6),
            dict(name='camera_head',params=list(new.camera_head.parameters()),lr=1e-4)]
    opt=torch.optim.AdamW(groups)
    names={g['name']:[n for n,p in new.named_parameters() if n.startswith('camera_head.' if g['name']=='camera_head' else 'decoder.')] for g in groups}
    monkeypatch.setattr(ck.FSDP,'scatter_full_optim_state_dict',mock_scatter)
    expected_rng=rng[0]['torch_cpu'].clone()
    ck.load_filtered_optimizer_checkpoint(loaded,new,opt,rng,names,0,('camera_head.',),True)
    assert torch.equal(torch.get_rng_state(),expected_rng)
    for n,p in new.named_parameters():
        if n.startswith('camera_head.'):
            assert p not in opt.state
        else:
            for key in ('step','exp_avg','exp_avg_sq'):
                assert torch.equal(opt.state[p][key],payload['optimizer']['state'][n][key])
    opt.zero_grad()
    (new.decoder(torch.ones(1,3)).sum()+new.camera_head(torch.ones(1,3)).sum()).backward(); opt.step()
    assert all(int(opt.state[p]['step'])==(1 if n.startswith('camera_head.') else 4) for n,p in new.named_parameters())
    exact=Tiny(camera=True); exact.load_state_dict(new.state_dict(),strict=True)
    opt2=torch.optim.AdamW([dict(name='dense_decoder',params=list(exact.decoder.parameters()),lr=3e-6),
                            dict(name='camera_head',params=list(exact.camera_head.parameters()),lr=1e-4)])
    opt2.load_state_dict(opt.state_dict())
    for p,q in zip(new.parameters(),exact.parameters()):
        assert torch.equal(p,q)
        for k in ('step','exp_avg','exp_avg_sq'): assert torch.equal(opt.state[p][k],opt2.state[q][k])


def test_structural_model_rejects_unaudited_missing_or_unexpected(tmp_path,monkeypatch):
    monkeypatch.setattr(ck.dist,'broadcast_object_list',lambda *a,**kw: None)
    payload=payload_for(Tiny()); del payload['model']['decoder.bias']
    path=tmp_path/'bad.pt'; torch.save(payload,path)
    with pytest.raises(RuntimeError,match='migration mismatch'):
        ck.load_unwrapped_model_checkpoint(path,Tiny(True),0,2,('camera_head.',))


def test_optimizer_rejects_dropped_old_state(monkeypatch):
    model=Tiny(True); payload=payload_for(Tiny())
    opt=torch.optim.AdamW(model.camera_head.parameters())
    opt.param_groups[0]['name']='camera_head'
    names={'camera_head':['camera_head.weight','camera_head.bias']}
    with pytest.raises(RuntimeError,match='discard existing Adam'):
        ck.load_filtered_optimizer_checkpoint(payload,model,opt,payload['training_state']['rng_states'],names,0,('camera_head.',),True)
