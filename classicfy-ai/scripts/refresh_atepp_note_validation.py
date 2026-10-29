"""Reuse accepted alignments to validate a newer R adapter without re-matching."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import argparse
import json
from pathlib import Path
import sys
import os

sys.path.insert(0,os.environ.get('CLASSICFY_VALIDATION_SRC',str(Path(__file__).resolve().parents[1]/'src')))
from validate_atepp_note_pipeline import read_log,dump


@lru_cache(maxsize=4)
def score(path):
    from embedding.note_score_context_data import load_note_score_context
    return load_note_score_context(path)[0]


def refresh(task):
    import numpy as np
    from embedding.note_performance_data import save_note_performance,load_note_performance
    from preprocessing.note_performance_adapters import prepare_atepp_note_performance
    from features.note_performance import inverse_r_physical,summarize_note_performance
    root,out,result,source,row=task
    pid=row['perf_id'];old=Path(result['performance_cache']).parent
    bridge=json.loads((old/'bridge.json').read_text())
    try:
        s,r=prepare_atepp_note_performance(Path(root)/row['score_path'],source['score_midi'],Path(root)/row['midi_path'],
             old/'alignment.npz',bridge,record_key=pid,work_key=result['work_key'],split=result['split'],score_features=score(source['score_cache']))
        p,m=inverse_r_physical(r.values,r.mask,r.local_tau)
        np.testing.assert_allclose(p[m],r.physical[m],rtol=1e-10,atol=1e-10)
        dest=Path(out)/'performances'/pid/'R'
        if not dest.exists():save_note_performance(r,dest)
        # Whitespace is not part of the immutable R fingerprint. Keep this new
        # validation cache compact without changing values, masks or provenance.
        manifest=json.loads((dest/'manifest.json').read_text())
        (dest/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,separators=(',',':')))
        assert load_note_performance(dest).fingerprint==r.fingerprint
        invalid=r.beats.provenance.get('interval_validation',{}).get('out_of_range_indices',[])
        return {**result,'status':'extracted','performance_cache':str(dest),'fingerprint':r.fingerprint,
                'coverage':summarize_note_performance(r),'alignment_audit':r.provenance['alignment'],
                'adapter':r.provenance['adapter'],'masked_estimated_intervals':len(invalid),
                'reused_alignment':str(old/'alignment.npz'),'previous_R_fingerprint':result['fingerprint']}
    except Exception as exc:return {**result,'status':'refresh_error','error':f'{type(exc).__name__}: {exc}'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    rows={r['perf_id']:r for r in __import__('csv').DictReader((a.root/'ATEPP-metadata-1.2.csv').open(encoding='utf-8-sig'))}
    old=read_log(a.previous/'alignments.jsonl');sources={r['score_path']:r for r in read_log(a.previous/'scores.jsonl')}
    for name in ['raw.jsonl','scores.jsonl','scan_summary.json','splits.json','source_integrity.json','alignment_environment.json']:
        target=a.out/name
        if not target.exists():target.write_bytes((a.previous/name).read_bytes())
    path=a.out/'alignments.jsonl';done={r['perf_id'] for r in read_log(path)}
    tasks=[]
    with path.open('a') as stream:
        for result in old:
            if result['perf_id'] in done:continue
            if result['status']!='extracted':stream.write(json.dumps(result)+'\n')
            else:
                row=rows[result['perf_id']];tasks.append((str(a.root),str(a.out),result,sources[row['score_path']],row))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,future in enumerate(as_completed([pool.submit(refresh,t) for t in tasks]),1):
            result=future.result()
            with path.open('a') as stream:stream.write(json.dumps(result,ensure_ascii=False)+'\n')
            if i%100==0 or i==len(tasks):print('refresh R',i,'/',len(tasks),result['status'],flush=True)
    from collections import Counter
    results=read_log(path);dump(a.out/'alignment_summary.json',{'statuses':dict(Counter(r['status'] for r in results)),
         'masked_estimated_intervals':sum(r.get('masked_estimated_intervals',0) for r in results),
         'records_with_masked_intervals':sum(r.get('masked_estimated_intervals',0)>0 for r in results),
         'errors':[r for r in results if r['status']=='refresh_error']})
    if any(r['status']=='refresh_error' for r in results):raise SystemExit(1)


if __name__=='__main__':main()
