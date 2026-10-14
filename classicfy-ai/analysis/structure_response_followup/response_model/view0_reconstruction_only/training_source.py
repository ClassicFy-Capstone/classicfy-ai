"""Train predeclared pilot models using only one search view per direction."""
import argparse
import copy
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from data import OUT,ViewData,sha
from model import ResponseEncoder,masks,objective,response_errors,reconstruction,vicreg

MODES=('reconstruction_only','response','response_aux','context_only')


def validate(model,data,indices,w,hidden_a,hidden_b,mode):
    model.eval();parts={};latent=[]
    # Consistency variance/covariance is computed on the complete validation set,
    # while response and reconstruction losses are exactly recording-balanced.
    projections_a=[];projections_b=[]
    with torch.no_grad():
        for start in range(0,len(indices),256):
            ii=indices[start:start+256];x=data.x[ii];c=data.context[ii]
            ta={k:v[ii] for k,v in data.targets.items()}
            za,pa,ha=model(x,c,hidden_a[start:start+len(ii)])
            zb,pb,hb=model(x,c,hidden_b[start:start+len(ii)])
            ra,rb=response_errors(ha,ta),response_errors(hb,ta)
            rr=(reconstruction(ha['rec'],x,hidden_a[start:start+len(ii)])+reconstruction(hb['rec'],x,hidden_b[start:start+len(ii)]))/2
            local={k:(ra[k]+rb[k])/2 for k in ra};local['rec']=rr
            for key,values in local.items():parts[key]=parts.get(key,0.)+float(np.dot(w[start:start+len(ii)],values.numpy()))
            projections_a.append(pa);projections_b.append(pb);latent.append(za)
    a,b=torch.cat(projections_a),torch.cat(projections_b)
    # Weighted VICReg counterpart, avoiding validation bias from more curves per work.
    ww=torch.tensor(w,dtype=a.dtype)
    inv=float((ww[:,None]*(a-b).square()).sum()/a.shape[1])
    def moments(x):
        mu=(ww[:,None]*x).sum(0);center=x-mu
        covariance=center.T@(center*ww[:,None])
        spread=torch.sqrt(covariance.diag()+1e-4)
        var=float(F.relu(1-spread).mean())
        cov=float((covariance.square().sum()-covariance.diag().square().sum())/x.shape[1])
        return var,cov
    av,ac=moments(a);bv,bc=moments(b)
    parts.update(invariance=inv,variance=(av+bv)/2,covariance=(ac+bc)/2)
    parts['loss']=parts['rec'] if mode=='reconstruction_only' else parts['shape']+.25*(parts['direction']+parts['peaks']+parts['lag'])+inv+(av+bv)/2+.04*(ac+bc)/2
    if mode in ('response_aux','context_only'):parts['loss']+=.1*parts['rec']
    latent=torch.cat(latent).numpy();parts['median_latent_sd']=float(np.median(np.sqrt(np.maximum(w@latent**2-(w@latent)**2,0))))
    return parts


def run(view,mode,config):
    data=ViewData(view);indices=data.roles['train'];val=data.roles['validation'];vw=data.role_weights['validation']
    train_owners=set(data.arrays['owner'][indices]);val_owners=set(data.arrays['owner'][val]);test_owners=set(data.arrays['owner'][data.roles['test']])
    assert len(train_owners)==241 and len(val_owners)==43 and len(test_owners)==45
    assert not train_owners&(val_owners|test_owners)
    directory=OUT/f'view{view}_{mode}';directory.mkdir(parents=True,exist_ok=True)
    if (directory/'best.pt').exists():
        print('Retaining existing completed',directory.name,flush=True);return
    torch.manual_seed(config['seed']);torch.set_num_threads(config['cpu_threads']);torch.use_deterministic_algorithms(True)
    model=ResponseEncoder(mode=='context_only')
    optim=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    valgen=torch.Generator().manual_seed(config['seed']+2+view)
    va,vb=masks(len(val),valgen),masks(len(val),valgen)
    initial=validate(model,data,val,vw,va,vb,mode)
    common=dict(mode=mode,view=view,config=config,scales=data.scales,
        data_sha256=sha(OUT/'examples.npz'),code_sha256=hashlib.sha256(b''.join((OUT/name).read_bytes() for name in ('data.py','model.py','train.py'))).hexdigest(),
        train_owner_indices=sorted(train_owners),validation_owner_indices=sorted(val_owners),
        parameters=sum(p.numel() for p in model.parameters()))
    torch.save(dict(**common,state_dict=model.state_dict(),epoch=0,validation=initial),directory/'initial.pt')
    best=initial['loss'];best_epoch=0;best_state=copy.deepcopy(model.state_dict());best_metrics=initial;stale=0
    rows=[dict(epoch=0,seconds=0.,train_loss='',**{'validation_'+k:v for k,v in initial.items()})]
    start_time=time.monotonic();gen=torch.Generator().manual_seed(config['seed']+10+view)
    for epoch in range(1,config['epochs']+1):
        model.train();sample=torch.multinomial(torch.tensor(data.role_weights['train'],dtype=torch.double),4096,replacement=True,generator=gen)
        train_parts={};batches=0
        for start in range(0,len(sample),config['batch_size']):
            ii=indices[sample[start:start+config['batch_size']].numpy()];x=data.x[ii];c=data.context[ii]
            hidden_a,hidden_b=masks(len(ii),gen),masks(len(ii),gen)
            targets={k:v[ii] for k,v in data.targets.items()}
            optim.zero_grad(set_to_none=True)
            loss,parts=objective(mode,model(x,c,hidden_a),model(x,c,hidden_b),x,hidden_a,hidden_b,targets)
            assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip']);optim.step()
            for k,v in parts.items():train_parts[k]=train_parts.get(k,0.)+v
            batches+=1
        result=validate(model,data,val,vw,va,vb,mode)
        rows.append(dict(epoch=epoch,seconds=time.monotonic()-start_time,
            **{'train_'+k:v/batches for k,v in train_parts.items()},**{'validation_'+k:v for k,v in result.items()}))
        if result['loss']<best-1e-6:
            best=result['loss'];best_epoch=epoch;best_state=copy.deepcopy(model.state_dict());best_metrics=result;stale=0
        else:stale+=1
        fields=sorted(set(k for row in rows for k in row))
        with (directory/'history.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        if epoch==1 or epoch%2==0:
            print(f'view={view} {mode} epoch={epoch} train={train_parts["loss"]/batches:.4f} val={result["loss"]:.4f} best={best_epoch} time={time.monotonic()-start_time:.1f}s',flush=True)
        if stale>=config['patience'] and epoch>=config['minimum_epochs']:break
    torch.save(dict(**common,state_dict=best_state,epoch=best_epoch,validation=best_metrics),directory/'best.pt')
    summary=dict(**{k:v for k,v in common.items() if k not in ('scales',)},initial=initial,best=best_metrics,best_epoch=best_epoch,
        epochs_run=epoch,seconds=time.monotonic()-start_time,train_curves=len(indices),validation_curves=len(val),
        test_used_for_training_or_selection=False)
    (directory/'training.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Completed',directory.name,'best epoch',best_epoch,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--view',type=int,choices=(0,1));parser.add_argument('--mode',choices=MODES)
    args=parser.parse_args();config=json.loads((OUT/'protocol.json').read_text())['training']
    for view in (args.view,) if args.view is not None else (0,1):
        for mode in (args.mode,) if args.mode else MODES:run(view,mode,config)
