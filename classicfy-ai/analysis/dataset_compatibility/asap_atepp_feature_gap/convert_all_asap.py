"""Extend the verified matched-work conversion to all aligned ASAP performances.

Reuses prior matched-work extraction, keeps original data/checkpoints untouched,
and writes a separate quarter-note sequence cache with masks and provenance.
"""
import csv
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from analyze import OUT, ROOT, DATASETS, CHANNELS, read, write, sha, initialize, extract, normalize


def main():
    with np.load(OUT/"asap_quarter_raw.npz",allow_pickle=False)as z:
        manifest=json.loads(str(z["manifest"]))
        records=[dict(**r,**{k:z[f"{i}/{k}"].copy()for k in ["q","times","raw","mask"]})
                 for i,r in enumerate(manifest)]
    used={r["key"]for r in records}
    rows=read(DATASETS/"ASAP/metadata.csv")
    annotations=json.loads((DATASETS/"ASAP/asap_annotations.json").read_text())
    tasks=[]
    for r in rows:
        key=r["midi_performance"]
        if (key in used or not annotations[key]["score_and_performance_aligned"]
                or key=="Beethoven/Piano_Sonatas/18-2/Schmitt01.mid"):
            continue
        title="Toccata"if(r["composer"],r["title"])==("Schumann","Toccata_repeat")else r["title"]
        tasks.append((key,r["composer"]+"::"+title))
    failures=[]
    with ProcessPoolExecutor(max_workers=4,initializer=initialize)as pool:
        for i,r in enumerate(pool.map(extract,tasks,chunksize=8),1):
            if "error"in r:failures.append(r)
            else:records.append(r)
            if i%50==0:print(f"Additional ASAP quarter conversion {i}/{len(tasks)}",flush=True)
    records.sort(key=lambda r:r["key"])
    arrays={};manifest=[]
    for i,r in enumerate(records):
        manifest.append({k:v for k,v in r.items()if not isinstance(v,np.ndarray)})
        arrays.update({f"{i}/{k}":r[k]for k in ["q","times","raw","mask"]})
    np.savez_compressed(OUT/"all_asap_quarter_raw.npz",manifest=np.array(json.dumps(manifest)),**arrays)
    normalized=normalize(records)
    arrays={};manifest=[]
    for i,r in enumerate(normalized):
        manifest.append(dict(key=r["key"],work=r["work"],group=r["group"],channels=CHANNELS,scales=r["scales"]))
        arrays.update({f"{i}/q":r["shared_q"],f"{i}/values":np.where(r["shared_mask"],r["standardized"],0).astype(np.float32),
                       f"{i}/valid":r["shared_mask"],f"{i}/raw":r["shared_raw"]})
    np.savez_compressed(OUT/"all_asap_quarter_normalized.npz",manifest=np.array(json.dumps(manifest)),**arrays)
    write("all_asap_time_axis_audit.csv",[r["audit"]for r in records])
    write("all_asap_conversion_failures.csv",[dict(key=r["key"],work=r["work"],error=r["error"])for r in failures])
    audits=[r["audit"]for r in records]
    native=sum(r["native_intervals"]for r in audits)
    one=sum(r["native_one_quarter_intervals"]for r in audits)
    result=dict(source="All local ASAP aligned performances",metadata_rows=len(rows),
                original_aligned=sum(annotations[r["midi_performance"]]["score_and_performance_aligned"]for r in rows),
                identical_byte_duplicate_excluded=1,converted=len(records),failed=len(failures),
                normalized=len(normalized),native_intervals=native,native_one_quarter_intervals=one,
                native_shorter_intervals=sum(r["native_intervals_shorter_than_quarter"]for r in audits),
                native_longer_intervals=sum(r["native_intervals_longer_than_quarter"]for r in audits),
                native_nonquarter_percent=100*(1-one/native),
                quarter_intervals=sum(r["quarter_intervals"]for r in audits),
                quarter_boundaries=sum(r["quarter_boundaries"]for r in audits),
                inferred_quarter_boundaries=sum(r["inferred_quarter_boundaries"]for r in audits),
                special_or_invalid_quarter_intervals=sum(r["special_or_invalid_quarter_intervals"]for r in audits),
                cache_sha256=sha(OUT/"all_asap_quarter_normalized.npz"),
                behavior="Separate diagnostic inputs; original caches and models unchanged. Per-score/grid references need >=2 performances. Existing 64-beat checkpoints were not trained on this quarter-note input.")
    (OUT/"all_asap_conversion_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
