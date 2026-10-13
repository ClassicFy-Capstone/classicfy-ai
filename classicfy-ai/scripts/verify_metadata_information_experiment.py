"""Independently recompute readouts with augmented least squares and CSV scores."""

import argparse
from collections import defaultdict, Counter
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import lstsq, solve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'classicfy-ai/src'))
from audit_asap_metadata import read_csv, sha256


def independent_weights(rows):
    counts = Counter((r['family_id'],r['piece']) for r in rows)
    families = {r['family_id'] for r in rows}
    work_counts = Counter(family for family,piece in counts)
    return np.array([1/(len(families)*work_counts[r['family_id']]*counts[r['family_id'],r['piece']]) for r in rows])


def direct_fit(x,y,weights,alpha,query, *, augmented=True):
    mean = np.sum(x*weights[:,None],axis=0)
    scale = np.sqrt(np.sum((x-mean)**2*weights[:,None],axis=0))
    active = scale>1e-12
    z = (x[:,active]-mean[active])/scale[active]
    mean_y = float(np.sum(y*weights))
    if augmented:
        matrix = np.vstack((z*np.sqrt(weights)[:,None],np.sqrt(alpha)*np.eye(active.sum())))
        target = np.r_[(y-mean_y)*np.sqrt(weights),np.zeros(active.sum())]
        beta = lstsq(matrix,target,lapack_driver='gelsy')[0]
    else:
        matrix=z.T@(z*weights[:,None])+alpha*np.eye(active.sum())
        beta=solve(matrix,z.T@((y-mean_y)*weights),assume_a='pos')
    return ((query[:,active]-mean[active])/scale[active])@beta+mean_y


def independent_scores(rows,truth,prediction):
    pieces = defaultdict(list)
    for i,r in enumerate(rows):pieces[r['piece']].append(i)
    works=[]
    for piece,ids in sorted(pieces.items()):
        a,b=np.array(truth)[ids],np.array(prediction)[ids]
        works.append(dict(piece=piece,family=rows[ids[0]]['family_id'],
            global_mse=np.mean((a-b)**2),within_mse=np.mean((a-a.mean()-b+b.mean())**2)))
    def macro(field):
        families=defaultdict(list)
        for r in works:families[r['family']].append(r[field])
        return np.mean([np.mean(v) for v in families.values()])
    return macro('global_mse'),macro('within_mse'),works


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design',type=Path,default=ROOT/'classicfy-ai/analysis/metadata_information_design')
    parser.add_argument('--out',type=Path,default=ROOT/'classicfy-ai/analysis/metadata_information_results')
    parser.add_argument('--runs',type=Path,default=ROOT.parent/'datasets/metadata_information_run01')
    args=parser.parse_args()
    execution=json.loads((args.out/'execution.json').read_text())
    for name,expected in execution['csv_sha256'].items():
        if sha256(args.out/name)!=expected:raise ValueError('Output changed')
    if sha256(args.runs/'inputs.npz')!=execution['inputs_archive_sha256']:raise ValueError('Inputs changed')
    if sha256(args.design/'protocol.json')!=execution['frozen_design_protocol_sha256']:raise ValueError('Design changed')
    protocol=json.loads((args.design/'protocol.json').read_text())
    all_rows={r['key']:r for r in read_csv(args.design/'target_preflight.csv')}
    folds=read_csv(args.design/'fold_manifest.csv')
    main_predictions=read_csv(args.out/'predictions.csv')
    controls=read_csv(args.runs/'control_predictions.csv')
    predictions=main_predictions+controls
    prediction_lookup={(r['condition'],r['kind'],r['encoder_seed'],r['control_seed'],r['target'],r['key']):r for r in predictions}
    if len(prediction_lookup)!=len(predictions):raise ValueError('Duplicate prediction row')
    with np.load(args.runs/'inputs.npz',allow_pickle=False) as z:
        index={k:i for i,k in enumerate(z['keys'])};blocks={k:z[k].copy() for k in z.files if k!='keys'}
    matrix_names={'A0':(), 'A1':('Q',),'B':('Q','B'),'C':('Q','B','T'),'D':('Q','B','T','R')}
    full_checks,cv_checks,max_error,max_cv_error,rows_checked=0,0,0.,0.,0
    for number,run in enumerate(execution['runs']):
        path=args.runs/run['file']
        if sha256(path)!=run['archive_sha256']:raise ValueError('Readout archive changed')
        with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        rows=[all_rows[k] for k in a['keys']]
        targets=('change_concentration_cv',) if run['group']=='dynamics' else ('coupling_local_strength','coupling_local_variation')
        if any(r[f'{target}_eligible']!='True' for r in rows for target in targets):raise ValueError('Ineligible row')
        expected_keys={r['key'] for r in all_rows.values() if r[f'{targets[0]}_eligible']=='True'}
        if set(a['keys'])!=expected_keys:raise ValueError('Different comparison cohort')
        train=a['train_ids'];training=[rows[i] for i in train];weights=independent_weights(training)
        if not np.allclose(weights,a['train_weights'],atol=1e-14):raise ValueError('Training weights differ')
        train_families={r['family_id'] for r in training}
        test_families={r['family_id'] for r in rows if r['evaluation_role']=='test'}
        val_families={r['family_id'] for r in rows if r['evaluation_role']=='validation'}
        if train_families&test_families or train_families&val_families or test_families&val_families:raise ValueError('Family leakage')
        names=matrix_names.get(run['condition'],('Q','B',f"{'cnn64' if run['condition']=='E_CNN' else 'bilstm64'}_{run['encoder_seed']}"))
        ids=np.array([index[r['key']] for r in rows])
        numeric=[]
        added='T' if run['condition']=='C' else 'R'
        for name in names:
            values=blocks[name][ids]
            if run['kind']=='shuffle' and name==added:
                src=a['source_indices']
                if sorted(src.tolist())!=list(range(len(rows))):raise ValueError('Not a permutation')
                for i,j in enumerate(src):
                    if any(rows[i][f]!=rows[j][f] for f in ('evaluation_role','piece','grid')):raise ValueError('Shuffle stratum violated')
                values=values[src]
            elif run['kind']=='noise' and name==added:
                values=np.random.default_rng(run['control_seed']).standard_normal(values.shape)
            numeric.append(values)
        numeric=np.column_stack(numeric) if numeric else np.empty((len(rows),0))
        np.testing.assert_allclose(a['x'][:,int(a['raw_metadata_columns']):],numeric,atol=0,rtol=0)
        for column in range(int(a['raw_metadata_columns'])):
            kind,label=str(a['metadata_columns'][column]).split('=',1)
            observed=[r['composer'] if kind=='composer' else r['work_type'] if kind=='type' else r['composer']+'::'+r['work_type'] for r in rows]
            np.testing.assert_equal(a['x'][:,column],np.array([value==label for value in observed],float))
            if label not in [observed[i] for i in train]:raise ValueError('Metadata vocabulary leakage')
        for j,target in enumerate(targets):
            observed=np.array([float(r[target]) for r in rows])
            if target=='change_concentration_cv':observed=np.log1p(observed)
            np.testing.assert_allclose(observed,a['y'][:,j],atol=0,rtol=0)
            recalculated=direct_fit(a['x'][train],observed[train],weights,a['selected_alphas'][j],a['x'])
            error=float(np.max(abs(recalculated-a['predictions'][:,j])))
            max_error=max(max_error,error);full_checks+=1
            if error>1e-7:raise ValueError(f'Independent augmented fit differs {error}')
            for i,r in enumerate(rows):
                public=prediction_lookup[run['condition'],run['kind'],str(run['encoder_seed']),str(run['control_seed']),target,r['key']]
                if float(public['truth'])!=observed[i] or abs(float(public['prediction'])-a['predictions'][i,j])>1e-12:raise ValueError('CSV prediction differs')
                rows_checked+=1
            # Main model CV candidates are independently solved using Cholesky.
            # Extra full-training metadata columns absent in a fold are constant
            # zero and dropped; this is equivalent to that fold's vocabulary.
            if run['kind']=='main':
                mapping={r['key']:int(r['inner_fold']) for r in folds if r['target']==target and r['role']=='train'}
                fold_ids=np.array([mapping[r['key']] for r in training])
                cv=np.zeros((len(train),len(protocol['ridge_alphas'])))
                for fold in range(5):
                    fit_ids=train[fold_ids!=fold];eval_ids=train[fold_ids==fold]
                    fit_rows=[rows[i] for i in fit_ids]
                    for k,alpha in enumerate(protocol['ridge_alphas']):
                        values=direct_fit(a['x'][fit_ids],observed[fit_ids],independent_weights(fit_rows),alpha,a['x'][eval_ids],augmented=False)
                        cv[fold_ids==fold,k]=values
                        cv_checks+=1
                err=float(np.max(abs(cv-a['cv_oof_predictions'][:,:,j])));max_cv_error=max(max_cv_error,err)
                if err>1e-7:raise ValueError('Independent CV prediction differs')
        if (number+1)%30==0:print('Audited readouts',number+1,'/',len(execution['runs']),flush=True)
    # Recompute all public metrics from the actual CSV predictions.
    grouped=defaultdict(list)
    for r in predictions:grouped[tuple(r[f] for f in ('condition','kind','encoder_seed','control_seed','target','role'))].append(r)
    metrics=read_csv(args.out/'metrics.csv')+read_csv(args.out/'controls.csv')
    metric_checks=0
    for metric in metrics:
        key=tuple(metric[f] for f in ('condition','kind','encoder_seed','control_seed','target','role'))
        local=grouped[key];rows=[all_rows[r['key']] for r in local]
        a,b,_=independent_scores(rows,[float(r['truth']) for r in local],[float(r['prediction']) for r in local])
        if max(abs(a-float(metric['global_mse'])),abs(b-float(metric['within_mse'])))>1e-12:raise ValueError('Independent metric differs')
        metric_checks+=1
    # OOF aggregate losses, every selected alpha, and paired bootstrap intervals.
    cv_checks_selected=0
    for run in execution['runs']:
        with np.load(args.runs/run['file'],allow_pickle=False) as z:
            weights=z['train_weights'];pred=z['cv_oof_predictions'];truth=z['y'][z['train_ids']]
            losses=np.sum(weights[:,None,None]*(pred-truth[:,None,:])**2,axis=0)
            np.testing.assert_allclose(losses,z['cv_scores'],atol=1e-12,rtol=0)
            for j in range(losses.shape[1]):
                chosen=max(alpha for alpha,loss in zip(protocol['ridge_alphas'],losses[:,j]) if loss<=losses[:,j].min()+1e-12)
                if chosen!=z['selected_alphas'][j]:raise ValueError('Alpha selection differs')
                cv_checks_selected+=1
    bootstrap_checks=0
    for record in read_csv(args.out/'bootstrap.csv'):
        if record['interval_status']!='ok':continue
        def scores(condition,seed):
            local=grouped[condition,'main',seed,'0',record['target'],record['role']]
            _,_,works=independent_scores([all_rows[r['key']] for r in local],[float(r['truth']) for r in local],[float(r['prediction']) for r in local])
            groups=defaultdict(list)
            for r in works:groups[r['family']].append(r[record['metric']])
            return {k:np.mean(v) for k,v in groups.items()}
        before=scores(record['reference'],'0');after=scores(record['condition'],record['encoder_seed'])
        keys=sorted(before);a=np.array([before[k] for k in keys]);b=np.array([after[k] for k in keys])
        ids=np.random.default_rng(protocol['uncertainty']['seed']).integers(0,len(keys),(2000,len(keys)))
        av,bv=a[ids].mean(axis=1),b[ids].mean(axis=1)
        reduction=100*(av-bv)/av
        np.testing.assert_allclose(np.percentile(reduction,[2.5,97.5]),[float(record['reduction_ci_low']),float(record['reduction_ci_high'])],atol=1e-8)
        bootstrap_checks+=1
    result=dict(readout_archives=len(execution['runs']),independent_augmented_readouts=full_checks,
        main_cv_linear_solves=cv_checks,independently_checked_alpha_choices=cv_checks_selected,
        checked_prediction_rows=rows_checked,independent_metric_rows=metric_checks,bootstrap_intervals=bootstrap_checks,
        maximum_augmented_prediction_error=max_error,maximum_cv_prediction_error=max_cv_error,
        source_code_sha256=sha256(__file__),execution_sha256=sha256(args.out/'execution.json'))
    (args.out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
