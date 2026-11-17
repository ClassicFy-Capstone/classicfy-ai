"""Frozen-encoder pedal diagnostics; train probes only on train works.

Validation selects regularization/epochs. Test is reporting only. A failed
readout is not proof of absent information. Original AE is never modified.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

spec = importlib.util.spec_from_file_location('common_runner', Path(__file__).with_name('train_selected_common_ae.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def regress(x, y, split):
    tr, va, te = (split == k for k in ('train', 'validation', 'test'))
    center, scale = x[tr].mean(0), x[tr].std(0)
    scale[scale < 1e-8] = 1
    a = (x-center)/scale
    ym = y[tr].mean(0)
    gram = a[tr].T @ a[tr]
    eig, vectors = np.linalg.eigh(gram)
    rhs = vectors.T @ a[tr].T @ (y[tr]-ym)
    choices = []
    for alpha in (.1, 1., 10., 100., 1000., 10000.):
        coef = vectors @ (rhs/(np.maximum(eig, 0)+alpha)[:, None])
        pred = a @ coef+ym
        choices.append((np.mean((pred[va]-y[va])**2, axis=0), pred))
    which = np.argmin(np.array([v[0] for v in choices]),axis=0)
    pred = np.stack([choices[i][1][:, j] for j, i in enumerate(which)],axis=1)
    mse = np.mean((pred[te]-y[te])**2,axis=0)
    baseline = np.mean((y[te]-ym)**2,axis=0)
    variance = np.var(y[te],axis=0)
    return {'test_mse':mse.tolist(), 'train_mean_baseline_mse':baseline.tolist(),
            'test_skill_vs_train_mean':(1-mse/np.maximum(baseline,1e-12)).tolist(),
            'test_r2':(1-mse/np.maximum(variance,1e-12)).tolist(),
            'selected_alpha':[float((.1,1,10,100,1000,10000)[i]) for i in which]}


def nonlinear_summary_probe(z, y, split):
    torch.manual_seed(20261113)
    tr, va, te = (np.flatnonzero(split==s) for s in ('train','validation','test'))
    xm,xs=z[tr].mean(0),z[tr].std(0);xs[xs<1e-8]=1
    ym,ys=y[tr].mean(0),y[tr].std(0);ys[ys<1e-8]=1
    x=torch.tensor((z-xm)/xs,dtype=torch.float32)
    target=torch.tensor((y-ym)/ys,dtype=torch.float32)
    model=nn.Sequential(nn.Linear(64,64),nn.GELU(),nn.Linear(64,6))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    rng=np.random.default_rng(20261113);best=float('inf');stale=0
    for epoch in range(1,101):
        model.train()
        for batch in np.array_split(rng.permutation(tr),int(np.ceil(len(tr)/64))):
            optimizer.zero_grad()
            loss=(model(x[batch])-target[batch]).square().mean()
            loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad(): value=float((model(x[va])-target[va]).square().mean())
        if value<best-1e-5:
            best=value;stale=0;best_epoch=epoch
            state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else: stale+=1
        if stale>=10: break
    model.load_state_dict(state)
    with torch.no_grad(): prediction=model(x[te]).numpy()*ys+ym
    mse=np.mean((prediction-y[te])**2,axis=0)
    baseline=np.mean((y[te]-ym)**2,axis=0)
    return {'best_epoch':best_epoch,'completed_epochs':epoch,'test_mse':mse.tolist(),
        'test_skill_vs_train_mean':(1-mse/baseline).tolist(),
        'test_r2':(1-mse/y[te].var(0)).tolist(),
        'protocol':'Frozen AE; 64-64-6 GELU MLP; train-fit scaling, validation-selected epoch; normalized residual mean/std targets; one seed.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    out=args.run/'pedal_diagnostics';out.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(20261113)
    manifest=json.loads((args.run/'dataset.json').read_text(encoding='utf-8'))
    model,ckpt=runner.load_model(args.run/'best.pt')
    random_model=runner.TemporalAutoencoder().eval()
    seq=[]
    with np.load(args.run/'input_sequences.npz',allow_pickle=False) as z:
        for r in manifest['sequences']:
            seq.append({**r,'values':z[r['key']+'/values'],'valid':z[r['key']+'/valid']})
    dataset=runner.AllObservedWindows(seq)
    split=np.array([seq[i]['split'] for i,start in dataset.items])
    work=np.array([seq[i]['work'] for i,start in dataset.items])
    keys=['z','no_pedal_z','random_z','mask','x','m','prediction']
    arrays={k:[] for k in keys}
    with torch.no_grad():
        for x,m in runner.DataLoader(dataset,batch_size=64):
            latent=model.encode(x,m); no=x.clone();no[:,4:]=0
            vals=[latent,model.encode(no,m),random_model.encode(x,m),m[:,0].float(),x,m,
                  model.decoder(latent).reshape_as(x)]
            for k,v in zip(keys,vals):arrays[k].append(v.numpy())
    a={k:np.concatenate(v) for k,v in arrays.items()}
    count=a['m'].sum(2)
    means=(a['x']*a['m']).sum(2)/count
    variance=(((a['x']-means[:,:,None])**2)*a['m']).sum(2)/count
    targets=np.concatenate([means[:,4:],np.sqrt(variance[:,4:])],axis=1).astype(float)
    names=['depth_mean','down_ratio_mean','changes_mean','depth_std','down_ratio_std','changes_std']
    results={'checkpoint_sha256':runner.digest(args.run/'best.pt'), 'target_names':names,
             'protocol':'All windows; work/recording/score-disjoint original splits. Probe hyperparameters selected only on validation. Pedal targets are normalized residuals, not raw CC events.',
             'summary_ridge':{},'distribution':{}}
    for key in ('z','no_pedal_z','random_z','mask'):
        results['summary_ridge'][key]=regress(a[key].astype(float),targets,split)
    nonpedal=np.concatenate([means[:,:4],np.sqrt(variance[:,:4]),count[:,:1]],axis=1)
    results['summary_ridge']['nonpedal_statistics']=regress(nonpedal.astype(float),targets,split)
    print(json.dumps({'summary_ridge':results['summary_ridge']}),flush=True)
    for sp in ('train','validation','test'):
        rows=split==sp; channels=[]
        for ch in range(7):
            mask=a['m'][rows,ch];x=a['x'][rows,ch][mask];pred=a['prediction'][rows,ch][mask]
            channels.append({'zero_fraction':float(np.mean(np.abs(x)<1e-8)),
                'mean':float(x.mean()),'std':float(x.std()),'q_abs':np.quantile(abs(x),[.5,.9,.99,1]).tolist(),
                'mse':float(np.mean((x-pred)**2)), 'prediction_std':float(pred.std()),
                'correlation':float(np.corrcoef(x,pred)[0,1]),
                'nonzero_mse':float(np.mean((x[abs(x)>1e-8]-pred[abs(x)>1e-8])**2)) if np.any(abs(x)>1e-8) else None})
        results['distribution'][sp]=channels
    # Hold encoder fixed and refit a small nonlinear, pedal-only decoder.
    tr,va,te=(np.flatnonzero(split==k) for k in ('train','validation','test'))
    mu=a['z'][tr].mean(0);sd=a['z'][tr].std(0);sd[sd<1e-8]=1
    z=torch.tensor((a['z']-mu)/sd);target=torch.tensor(a['x'][:,4:]);mask=torch.tensor(a['m'][:,4:])
    decoder=nn.Sequential(nn.Linear(64,128),nn.GELU(),nn.Linear(128,192))
    optimizer=torch.optim.AdamW(decoder.parameters(),lr=.001,weight_decay=.001)
    def metrics(indices):
        decoder.eval()
        with torch.no_grad():
            prediction=decoder(z[indices]).reshape(-1,3,64)
            errors=torch.where(mask[indices],prediction-target[indices],0).square().sum((0,2))
            return (errors/mask[indices].sum((0,2))).numpy()
    best=float('inf');stale=0;history=[];rng=np.random.default_rng(20261113)
    for epoch in range(1,101):
        decoder.train()
        for batch in np.array_split(rng.permutation(tr),int(np.ceil(len(tr)/64))):
            optimizer.zero_grad();pred=decoder(z[batch]).reshape(-1,3,64)
            loss=torch.where(mask[batch],pred-target[batch],0).square().sum()/mask[batch].sum()
            loss.backward();optimizer.step()
        val=metrics(va);history.append({'epoch':epoch,'validation_mse':val.tolist()})
        if val.mean()<best-1e-5:
            best=float(val.mean());stale=0;best_epoch=epoch
            state={k:v.detach().clone() for k,v in decoder.state_dict().items()}
        else:stale+=1
        if stale>=10:break
    decoder.load_state_dict(state)
    results['frozen_z_pedal_decoder']={'best_epoch':best_epoch,'completed_epochs':epoch,
        'validation_mse':metrics(va).tolist(),'test_mse':metrics(te).tolist(),'history':history}
    # Relative input sensitivity, keeping masks fixed (zero is the reference residual).
    with torch.no_grad():
        original=torch.tensor(a['z'][te]);altered=torch.tensor(a['no_pedal_z'][te])
        results['zero_pedal_input_sensitivity']={'mean_cosine_distance':float((1-torch.nn.functional.cosine_similarity(original,altered)).mean()),
            'relative_l2_change':float(((original-altered).norm(dim=1)/original.norm(dim=1).clamp_min(1e-8)).mean())}
    # Measure objective gradient norms on encoder across fixed training batches.
    params=list(model.input_projection.parameters())+list(model.blocks.parameters())+list(model.bottleneck.parameters())
    norms=[];losses=[]
    for indices in np.array_split(tr[:640],20):
        x=torch.tensor(a['x'][indices]);m=torch.tensor(a['m'][indices]);pred=model(x,m)[0]
        per=((pred-x).square()*m).sum((0,2))/m.sum((0,2))
        groups=[per[i] for i in range(4)]+[per[4:].mean()]
        ns=[]
        for loss in groups:
            gs=torch.autograd.grad(loss/5,params,retain_graph=True)
            ns.append(float(torch.sqrt(sum(g.square().sum() for g in gs))))
        norms.append(ns);losses.append([float(v.detach()) for v in groups])
    results['encoder_gradient_groups']={'groups':['tempo','rubato','dynamics','articulation','pedal'],
        'mean_weighted_gradient_norm':np.mean(norms,axis=0).tolist(),'mean_group_mse':np.mean(losses,axis=0).tolist(),
        'scope':'first 640 train windows, model eval mode; descriptive, not full-training gradient history'}
    runner.dump(out/'report.json',results)
    np.savez_compressed(out/'probe_inputs.npz',z=a['z'],targets=targets,split=split,work=work)
    torch.save({'state_dict':state,'input_mean':torch.tensor(mu),'input_scale':torch.tensor(sd)},out/'frozen_pedal_decoder.pt')
    runner.dump(out/'nonlinear_summary_probe.json',nonlinear_summary_probe(a['z'],targets,split))
    print(json.dumps({k:v for k,v in results.items() if k!='summary_ridge'}),flush=True)


if __name__=='__main__':main()
