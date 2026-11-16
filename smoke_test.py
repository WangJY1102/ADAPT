"""Synthetic smoke: broadcast, first-order basis gradient, update, variable-way reset."""
import argparse
import json
import torch
from torch import nn
from torch.nn import functional as F
from basis import SharedCosineBasis, CosineClassifier
from model import task_state, call, adapt, build


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cpu')
    p.add_argument('--full',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(222)
    if a.full:
        model,basis,_=build(222,a.device)
        shape=(3,128,128)
    else:
        model=nn.Sequential()  # Named modules match the functional task-head contract.
        model.add_module('encoder',nn.Linear(8,224))
        model.add_module('classifier',CosineClassifier())
        model=model.to(a.device).eval();basis=SharedCosineBasis().to(a.device);shape=(8,)
    before={k:v.detach().clone() for k,v in model.named_parameters()}
    initial=basis.basis_raw.detach().clone()
    opt=torch.optim.Adam([p for p in list(model.parameters())+list(basis.parameters()) if p.requires_grad],lr=1e-4)
    for n in [3,5]:
        sx=torch.randn(n,*shape,device=a.device);sy=torch.arange(n,device=a.device)
        qx=torch.randn(n,*shape,device=a.device)
        pp,bb=task_state(model,basis,n)
        with torch.no_grad():
            probability=call(model,pp,bb,sx).softmax(-1)
            assert torch.allclose(probability,torch.full_like(probability,1/n),atol=1e-6)
        pp=adapt(model,pp,bb,sx,sy,steps=1)
        loss=F.cross_entropy(call(model,pp,bb,qx),sy)
        opt.zero_grad(set_to_none=True);loss.backward()
        assert basis.basis_raw.grad is not None and torch.isfinite(basis.basis_raw.grad).all()
        assert basis.basis_raw.grad.norm()>1e-9
        # Functional adaptation must not mutate the global encoder.
        assert all(torch.equal(before[k],v) for k,v in model.named_parameters())
    opt.step()
    assert not torch.equal(initial,basis.basis_raw)
    assert any(not torch.equal(before[k],v) for k,v in model.named_parameters())
    print(json.dumps(dict(passed=True,device=a.device,full=a.full,ways=[3,5],
        basis_gradient_norm=float(basis.basis_raw.grad.norm()),loss=float(loss.detach()))))


if __name__=='__main__':main()
