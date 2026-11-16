"""First-order selective scan compatibility backend, grouped real B/C only.

Parallel affine-prefix recurrence and analytic reverse recurrence avoid storing
the per-position autograd graph. Not a new model; equations match the serial
selective_scan_torch implementation. Higher-order derivatives are not supported.
"""
import torch
from torch.nn import functional as F
def prefix(a,b):
    offset=1
    while offset<a.shape[-1]:
        bt=b[...,offset:]+a[...,offset:]*b[...,:-offset]
        at=a[...,offset:]*a[...,:-offset]
        b=torch.cat((b[...,:offset],bt),-1)
        a=torch.cat((a[...,:offset],at),-1)
        offset*=2
    return b
def expand(v,d):
    if v.ndim==3:v=v.unsqueeze(1)
    return v.repeat_interleave(d//v.shape[1],1)
def groupback(v,original):
    groups=1 if original.ndim==3 else original.shape[1]
    shape=v.shape
    v=v.reshape(shape[0],groups,shape[1]//groups,shape[2],shape[3]).sum(2)
    return v.squeeze(1) if original.ndim==3 else v
class Scan(torch.autograd.Function):
    @staticmethod
    def forward(ctx,u,raw,A,B,C,D,bias,softplus):
        dt=raw+bias.view(1,-1,1) if bias.numel() else raw
        if softplus:dt=F.softplus(dt)
        a=torch.exp(dt.unsqueeze(2)*A[None,:,:,None])
        drive=dt.unsqueeze(2)*expand(B,u.shape[1])*u.unsqueeze(2)
        states=prefix(a,drive)
        out=(states*expand(C,u.shape[1])).sum(2)
        if D.numel():out=out+u*D.view(1,-1,1)
        ctx.softplus=softplus
        ctx.save_for_backward(u,raw,A,B,C,D,bias,states)
        return out
    @staticmethod
    def backward(ctx,dy):
        if torch.is_grad_enabled():raise RuntimeError('Parallel compatibility scan implements first-order gradients only')
        u,raw,A,B,C,D,bias,h=ctx.saved_tensors
        before=raw+bias.view(1,-1,1) if bias.numel() else raw
        dt=F.softplus(before) if ctx.softplus else before
        a=torch.exp(dt.unsqueeze(2)*A[None,:,:,None])
        adj_a=torch.cat((a[...,1:],torch.zeros_like(a[...,:1])),-1).flip(-1)
        lam=prefix(adj_a,(dy.unsqueeze(2)*expand(C,u.shape[1])).flip(-1)).flip(-1)
        del adj_a
        hp=torch.cat((torch.zeros_like(h[...,:1]),h[...,:-1]),-1)
        ta=lam*hp*a
        gA=(ta*dt.unsqueeze(2)).sum((0,3))
        gd=(ta*A[None,:,:,None]+lam*expand(B,u.shape[1])*u.unsqueeze(2)).sum(2)
        gB=groupback(lam*dt.unsqueeze(2)*u.unsqueeze(2),B)
        gC=groupback(dy.unsqueeze(2)*h,C)
        gu=(lam*dt.unsqueeze(2)*expand(B,u.shape[1])).sum(2)
        gD=(dy*u).sum((0,2)) if D.numel() else None
        if D.numel():gu=gu+dy*D.view(1,-1,1)
        if ctx.softplus:gd=gd*torch.sigmoid(before)
        gb=gd.sum((0,2)) if bias.numel() else None
        return gu,gd,gA,gB,gC,gD,gb,None
def selective_scan_torch(u,delta,A,B,C,D=None,delta_bias=None,delta_softplus=True):
    dtype=u.dtype
    empty=u.new_empty(0,dtype=torch.float32)
    return Scan.apply(u.float(),delta.float(),A.float(),B.float(),C.float(),D.float() if D is not None else empty,delta_bias.float() if delta_bias is not None else empty,delta_softplus).to(dtype)
