"""Small conditional encoder; observed response heads share one bottleneck."""
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self,dilation):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv1d(32,32,3,padding=dilation,dilation=dilation),nn.GELU(),nn.Dropout(.1),
            nn.Conv1d(32,32,3,padding=dilation,dilation=dilation),nn.GELU())
    def forward(self,x):return x+self.layers(x)


class ResponseEncoder(nn.Module):
    def __init__(self,context_only=False):
        super().__init__();self.context_only=context_only
        self.performance=nn.Sequential(nn.Conv1d(21,32,1),nn.GELU())
        self.score=nn.Sequential(nn.Conv1d(13,16,1),nn.GELU())
        self.combine=nn.Conv1d(48,32,1)
        self.blocks=nn.Sequential(Block(1),Block(2),Block(4))
        self.bottleneck=nn.Linear(32*17,32)
        self.projection=nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Linear(64,64))
        self.heads=nn.ModuleDict({name:nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Linear(64,width))
            for name,width in dict(rec=7*17,shape=2*16,direction=2*16*3,peaks=2*17,lag=54).items()})

    def encode(self,x,context,hidden=None):
        assert x.shape[1:]==(7,17) and context.shape[1:]==(13,17)
        hidden=torch.zeros_like(x,dtype=torch.bool) if hidden is None else hidden
        if self.context_only:
            clean=torch.zeros_like(x);valid=torch.zeros_like(x);hidden=torch.zeros_like(hidden)
        else:
            clean=torch.where(hidden,0.,x);valid=torch.ones_like(x)
        p=self.performance(torch.cat((clean,valid,hidden.float()),dim=1))
        c=self.score(context)
        encoded=self.blocks(self.combine(torch.cat((p,c),dim=1)))
        return self.bottleneck(encoded.flatten(1))

    def forward(self,x,context,hidden=None):
        z=self.encode(x,context,hidden)
        return z,self.projection(z),{name:head(z) for name,head in self.heads.items()}


def masks(n,generator):
    # Two complete positions, rather than discarding time or changing values.
    positions=torch.rand(n,17,generator=generator).argsort(dim=1)[:,:2]
    result=torch.zeros(n,17,dtype=torch.bool).scatter_(1,positions,True)
    return result[:,None,:].expand(-1,7,-1)


def masked_mean(values,valid):
    v=valid.to(values.dtype)
    return (values*v).reshape(len(values),-1).sum(1)/v.reshape(len(values),-1).sum(1).clamp_min(1)


def reconstruction(prediction,x,hidden):
    prediction=prediction.reshape(-1,7,17)
    channel=((prediction-x).square()*hidden).sum(2)/hidden.sum(2).clamp_min(1)
    return torch.stack((channel[:,0],channel[:,1],channel[:,2],channel[:,3],channel[:,4:].mean(1)),dim=1).mean(1)


def response_errors(heads,targets):
    shape=masked_mean((heads['shape'].reshape(-1,2,16)-targets['shape']).square(),targets['shape_mask'])
    logits=heads['direction'].reshape(-1,2,16,3)
    direction=F.cross_entropy(logits.reshape(-1,3),targets['direction'].flatten(),reduction='none').reshape(-1,2,16).mean((1,2))
    peaks=F.cross_entropy(heads['peaks'].reshape(-1,2,17).reshape(-1,17),targets['peaks'].flatten(),reduction='none').reshape(-1,2)
    peaks=masked_mean(peaks,targets['peak_mask'])
    lag=masked_mean((heads['lag']-targets['lag']).square(),targets['lag_mask'])
    # Types without lag support contribute zero, not a manufactured zero target.
    return dict(shape=shape,direction=direction,peaks=peaks,lag=lag)


def vicreg(a,b):
    invariance=(a-b).square().mean()
    variance=(F.relu(1-torch.sqrt(a.var(0,unbiased=False)+1e-4)).mean()+
              F.relu(1-torch.sqrt(b.var(0,unbiased=False)+1e-4)).mean())/2
    def covariance(x):
        x=x-x.mean(0);c=x.T@x/max(len(x)-1,1)
        return (c.square().sum()-c.diag().square().sum())/x.shape[1]
    cov=(covariance(a)+covariance(b))/2
    return dict(invariance=invariance,variance=variance,covariance=cov)


def objective(mode,outputs_a,outputs_b,x,hidden_a,hidden_b,targets):
    za,pa,ha=outputs_a;zb,pb,hb=outputs_b
    rec=(reconstruction(ha['rec'],x,hidden_a)+reconstruction(hb['rec'],x,hidden_b))/2
    ra,rb=response_errors(ha,targets),response_errors(hb,targets)
    response={name:(ra[name]+rb[name])/2 for name in ra}
    regular=vicreg(pa,pb)
    per_example=response['shape']+.25*(response['direction']+response['peaks']+response['lag'])
    consistency=regular['invariance']+regular['variance']+.04*regular['covariance']
    loss=rec.mean() if mode=='reconstruction_only' else per_example.mean()+consistency
    if mode in ('response_aux','context_only'):loss=loss+.1*rec.mean()
    parts={**{name:float(value.mean().detach()) for name,value in response.items()},
        **{name:float(value.detach()) for name,value in regular.items()},'rec':float(rec.mean().detach()),'loss':float(loss.detach())}
    return loss,parts
