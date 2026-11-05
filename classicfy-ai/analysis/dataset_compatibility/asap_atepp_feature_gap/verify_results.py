"""Independent arithmetic, source-event and model-input checks for the audit."""
import csv
import json
import sys
from collections import defaultdict

import numpy as np

from analyze import OUT, ROOT, DATASETS, read, load_midi, CHANNELS
from embedding.data import PerformanceSequence, WindowDataset


def main():
    checks=[]
    summary=json.loads((OUT/'summary.json').read_text())
    full=json.loads((OUT/'all_asap_conversion_summary.json').read_text())
    max_d=max_p=0.0
    with np.load(OUT/'all_asap_quarter_raw.npz',allow_pickle=False)as z:
        meta=json.loads(str(z['manifest']))
        assert len(meta)==full['converted']==1035
        assert len({r['key']for r in meta})==len(meta)
        assert len({r['audit']['midi_sha256']for r in meta})==len(meta)
        for i,r in enumerate(meta):
            q,t,x,m=[z[f'{i}/{k}']for k in ['q','times','raw','mask']]
            np.testing.assert_allclose(np.diff(q),1,rtol=0,atol=1e-12)
            assert np.all(np.diff(t)>0) and x.shape==m.shape==(7,len(q)-1)
            assert np.isfinite(x[m]).all() and np.isnan(x[~m]).all()
            np.testing.assert_allclose(x[0,m[0]],np.log2(.5/np.diff(t))[m[0]],rtol=0,atol=1e-12)
        checks.extend(['1035 distinct source MIDI hashes; no duplicated performance',
                       'all raw grids have quarter-note step 1, increasing performance times and consistent masks',
                       'tempo is independently checked against 120-BPM quarter reference'])
        # Samples cover direct anchors, subdivided beats and multi-quarter intervals.
        chosen=[]
        for condition in [lambda r:r['audit']['inferred_quarter_boundaries']==0,
                          lambda r:r['audit']['native_q_step_median']>1.1,
                          lambda r:r['audit']['native_q_step_median']<.9]:
            chosen.append(next(i for i,r in enumerate(meta)if condition(r)))
        for i in chosen:
            r=meta[i];t=z[f'{i}/times'];x=z[f'{i}/raw'];m=z[f'{i}/mask']
            midi=load_midi(DATASETS/'ASAP'/r['key'])
            times=np.array([e.time for e in midi.pedals]);vals=np.array([e.value for e in midi.pedals])
            bins=np.flatnonzero(m[2]&m[4])
            for bi in bins[np.linspace(0,len(bins)-1,min(12,len(bins))).astype(int)]:
                left,right=t[bi:bi+2]
                velocities=[n.velocity for n in midi.notes if left<=n.start<right]
                d=np.mean(velocities)/127
                max_d=max(max_d,abs(d-x[2,bi]))
                cuts=np.r_[left,times[(times>left)&(times<right)],right]
                area=0.0
                for a,b in zip(cuts[:-1],cuts[1:]):
                    j=np.searchsorted(times,a,side='right')-1
                    area+=(b-a)*(vals[j]/127 if j>=0 else 0)
                max_p=max(max_p,abs(area/(right-left)-x[4,bi]))
        assert max_d<1e-12 and max_p<1e-10
        checks.append('Dynamics and pedal depth independently recalculated from source note/CC64 events in 36 sampled bins')
    sequences=[]
    with np.load(OUT/'all_asap_quarter_normalized.npz',allow_pickle=False)as z:
        meta=json.loads(str(z['manifest']))
        assert len(meta)==full['normalized']==968
        for i,r in enumerate(meta):
            q,x,m=[z[f'{i}/{k}']for k in ['q','values','valid']]
            assert tuple(r['channels'])==CHANNELS and x.shape==m.shape==(7,len(q)-1)
            assert m.dtype==bool and np.isfinite(x).all() and np.all(x[~m]==0)
            sequences.append(PerformanceSequence(r['key'],r['work'],'quarter',x,m))
    windows=WindowDataset(sequences,window_size=64,stride=32,minimum_coverage=.5)
    assert len(windows)>0
    checks.append('Normalized quarter cache passes the existing seven-channel WindowDataset with 64-quarter windows')
    # Direct pair comparison, independently of scipy.rankdata used by analysis.
    pred=read(OUT/'domain_predictions.csv')
    results=read(OUT/'domain_separability.csv')
    for result in results:
        selected=[r for r in pred if r['features']==result['features']and r['stage']==result['stage']]
        workgroups=defaultdict(list)
        for r in selected:workgroups[r['work']].append(r)
        values=[]
        for work,rs in workgroups.items():
            assert len({r['heldout_fold']for r in rs})==1
            a=np.array([float(r['score'])for r in rs if r['dataset']=='ASAP'])
            t=np.array([float(r['score'])for r in rs if r['dataset']=='ATEPP'])
            delta=t[:,None]-a[None,:]
            values.append(float(((delta>0)+.5*(delta==0)).mean()))
        assert abs(np.mean(values)-float(result['heldout_work_macro_auc']))<1e-12
    checks.append('Held-out-work AUC independently recomputed by all cross-dataset score pairs; work stays in one fold')
    for features in {r['features']for r in pred}:
        sets=[]
        for stage in ['raw','standardized']:
            sets.append({(r['dataset'],r['key'],r['work'])for r in pred
                         if r['features']==features and r['stage']==stage})
        assert sets[0]==sets[1]
    checks.append('Raw/standardized classification uses exactly the same performances in each feature subset')
    result=dict(status='passed',checks=checks,max_independent_dynamics_error=max_d,
                max_independent_pedal_depth_error=max_p,
                normalized_sequences=len(sequences),eligible_64_quarter_windows=len(windows),
                sequences_with_eligible_windows=len(sequences)-len(windows.skipped_keys),
                caveat='These checks verify conversion and arithmetic, not autoencoder transfer quality or causal attribution of dataset differences.')
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
