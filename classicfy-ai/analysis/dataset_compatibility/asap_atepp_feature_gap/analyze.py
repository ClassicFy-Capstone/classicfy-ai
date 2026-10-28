"""Matched-repertoire feature audit and ASAP quarter-note re-extraction.

ATEPP source MIDI/alignments are not local. Its saved, quality-accepted summaries
are used unchanged. Differences are descriptive dataset differences, not isolated
transcription error or autoencoder transfer accuracy.
"""
import argparse
import csv
import hashlib
import json
import sys
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata, ks_2samp

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent))
from _support import REPO_ROOT, DATASETS, source_label
ROOT = REPO_ROOT.parent
AI = REPO_ROOT / "classicfy-ai"
AN = AI / "analysis"
sys.path.insert(0, str(AI / "src"))
from features import (BeatSequence, PieceFeatureInput, extract_dynamics,
                      extract_pedaling, extract_articulation, fit_residual_scale)
from preprocessing import ASAPLoader, load_midi, load_match
from quarter_grid import score_tempo_map, seconds_to_quarters, quarter_boundaries

CHANNELS = ("tempo", "rubato", "dynamics", "articulation", "pedal_depth", "pedal_down_ratio", "pedal_changes")
SEED = 20261028
LOADER = None


def read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write(name, rows):
    if not rows:
        if name.endswith("conversion_failures.csv"):
            with (OUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
                csv.writer(stream, lineterminator="\n").writerow(["key", "work", "error"])
        return
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return float("nan")


def initialize():
    global LOADER
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    LOADER = ASAPLoader(DATASETS / "ASAP", DATASETS / "nASAP")


def extract(task):
    key, work = task
    try:
        sample = LOADER.get_sample(key)
        q = seconds_to_quarters(sample.score_beats, score_tempo_map(str(sample.score_path)))
        grid, times, good, exact = quarter_boundaries(q, sample.performance_beats, sample.score_beat_types)
        midi = load_midi(sample.performance_path)
        d = extract_dynamics(midi, times).sequence
        p = extract_pedaling(midi, times)
        values = np.full((7, len(good)), np.nan)
        mask = np.zeros(values.shape, bool)
        values[0] = np.log2(.5 / np.diff(times))
        mask[0] = good
        values[1] = values[0] - np.median(values[0, good]) if good.any() else np.nan
        mask[1] = good
        values[2], mask[2] = d.values, d.mask & good
        if sample.robust_note_alignment is True and sample.note_alignment_path:
            a = extract_articulation(load_match(sample.note_alignment_path), times).sequence
            values[3], mask[3] = a.values, a.mask & good
        if midi.pedals:
            for i, seq in enumerate([p.depth, p.down_ratio, p.changes], 4):
                values[i], mask[i] = seq.values, seq.mask & good
        # Apply the same missing-CC64 policy as the ATEPP normalization adapter.
        values[~mask] = np.nan
        native_dynamics = extract_dynamics(midi, sample.performance_beats).sequence
        ref = PieceFeatureInput.from_asap_sample(sample, native_dynamics).piece_key
        levels = sorted({event.value for event in midi.pedals})
        audit = dict(key=key, work=work, status="converted", reason="",
                     native_intervals=len(q)-1, quarter_intervals=len(good),
                     native_one_quarter_intervals=int(np.isclose(np.diff(q), 1, atol=2e-5, rtol=0).sum()),
                     native_intervals_longer_than_quarter=int((np.diff(q) > 1+2e-5).sum()),
                     native_intervals_shorter_than_quarter=int((np.diff(q) < 1-2e-5).sum()),
                     native_q_step_min=float(np.min(np.diff(q))), native_q_step_median=float(np.median(np.diff(q))),
                     native_q_step_max=float(np.max(np.diff(q))),
                     inferred_quarter_boundaries=int((~exact).sum()), quarter_boundaries=len(grid),
                     special_or_invalid_quarter_intervals=int((~good).sum()),
                     robust_articulation=sample.robust_note_alignment,
                     pedal_events=len(midi.pedals), pedal_value_levels=len(levels),
                     pedal_binary_values_only=bool(levels and set(levels) <= {0, 127}),
                     midi_sha256=sha(sample.performance_path),
                     score_sha256=sha(sample.score_path))
        source_path, separator, variant = ref.partition("::")
        group = Path(source_path).relative_to(DATASETS / "ASAP").as_posix()
        if separator:
            group += "::" + variant
        return dict(key=key, work=work, group=group,
                    q=grid, times=times, raw=values, mask=mask, audit=audit)
    except Exception as error:
        return dict(key=key, work=work, error=f"{type(error).__name__}: {error}")


def summarize(values, mask):
    x = values[mask & np.isfinite(values)]
    return (float(x.mean()), float(np.ptp(np.percentile(x, [5, 95]))), len(x)) if len(x) else (np.nan, np.nan, 0)


def normalize(records):
    groups = defaultdict(list)
    for record in records:
        groups[record["group"]].append(record)
    outputs = []
    for group, members in groups.items():
        lo = max(r["q"][0] for r in members)
        hi = min(r["q"][-1] for r in members)
        if len(members) < 2 or hi <= lo:
            continue
        width = int(hi-lo)
        xs, masks = [], []
        for r in members:
            i = int(lo-r["q"][0])
            xs.append(r["raw"][:, i:i+width])
            masks.append(r["mask"][:, i:i+width])
        x, mask = np.stack(xs), np.stack(masks)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            common = np.nanmedian(np.where(mask, x, np.nan), axis=0)
        support = mask.sum(axis=0)
        valid = mask & (support[None, :, :] >= 2)
        residual = x - common[None, :, :]
        normalized = np.full(x.shape, np.nan)
        scales = []
        for ci, channel in enumerate(CHANNELS):
            scale = fit_residual_scale([BeatSequence(v[ci], m[ci]) for v,m in zip(residual,valid)],
                                       method="std" if ci >= 4 else "mad")
            normalized[:, ci] = residual[:, ci] / scale.value
            scales.append(dict(channel=channel, value=scale.value, method=scale.method))
        for i, r in enumerate(members):
            outputs.append(dict(**r, shared_q=np.arange(lo,hi+1), shared_raw=x[i],
                                shared_mask=valid[i], standardized=normalized[i], scales=scales))
    return outputs


def summary_rows(records):
    rows = []
    for record in records:
        for ci, channel in enumerate(CHANNELS):
            rm, rr, n = summarize(record["raw"][ci], record["mask"][ci])
            sm, sr, sn = summarize(record.get("standardized", record["raw"])[ci],
                                  record.get("shared_mask", record["mask"])[ci])
            if "standardized" not in record:
                sm = sr = np.nan
                sn = 0
            # The raw comparator uses the same shared crop as normalization when available.
            if "shared_raw" in record:
                rm, rr, n = summarize(record["shared_raw"][ci], record["shared_mask"][ci])
            rows.append(dict(dataset="ASAP", key=record["key"], work=record["work"],
                             group=record["group"], channel=channel, raw_mean=rm, raw_range=rr,
                             standardized_mean=sm, standardized_range=sr,
                             total_intervals=len(record.get("shared_q",record["q"]))-1,
                             raw_valid=n, standardized_valid=sn))
    return rows


def auc(labels, scores):
    labels = np.asarray(labels, bool)
    n, m = labels.sum(), (~labels).sum()
    return float((rankdata(scores)[labels].sum() - n*(n+1)/2)/(n*m))


def compare(rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for field in ["raw_mean", "raw_range", "standardized_mean", "standardized_range"]:
            value = finite_number(r[field])
            if np.isfinite(value):
                grouped[r["work"], r["channel"], field][r["dataset"]].append(value)
    details = []
    for (work, channel, field), parts in sorted(grouped.items()):
        a, t = np.asarray(parts["ASAP"]), np.asarray(parts["ATEPP"])
        if min(len(a),len(t)) < 2:
            continue
        merged = np.r_[a,t]
        scale = np.ptp(np.percentile(merged,[25,75]))/1.3489795
        if scale < 1e-12:
            scale = np.std(merged)
        p = auc(np.r_[np.zeros(len(a)),np.ones(len(t))],merged)
        details.append(dict(work=work, channel=channel, field=field,
                            asap_n=len(a), atepp_n=len(t), asap_median=float(np.median(a)),
                            atepp_median=float(np.median(t)),
                            median_difference=float(np.median(t)-np.median(a)),
                            robust_shift=float((np.median(t)-np.median(a))/scale) if scale>1e-12 else np.nan,
                            probability_atepp_greater=p, rank_separation=abs(2*p-1),
                            ks=float(ks_2samp(a,t,method="asymp").statistic)))
    write("per_work_differences.csv", details)
    summary = []
    for channel in CHANNELS:
        for field in ["raw_mean","raw_range","standardized_mean","standardized_range"]:
            ds=[d for d in details if d["channel"]==channel and d["field"]==field]
            if not ds:
                continue
            summary.append(dict(channel=channel, field=field, works=len(ds),
                                asap_work_median=float(np.median([d["asap_median"] for d in ds])),
                                atepp_work_median=float(np.median([d["atepp_median"] for d in ds])),
                                median_within_work_difference=float(np.median([d["median_difference"] for d in ds])),
                                median_abs_robust_shift=float(np.nanmedian([abs(d["robust_shift"]) for d in ds])),
                                median_rank_separation=float(np.median([d["rank_separation"] for d in ds])),
                                median_ks=float(np.median([d["ks"] for d in ds]))))
    write("feature_gap_summary.csv", summary)
    return details, summary


def domain_check(rows, *, shuffle_labels=False):
    by = defaultdict(dict)
    for r in rows:
        by[r["dataset"],r["key"],r["work"]][r["channel"]] = r
    results=[]
    predictions=[]
    for subset, channels in [("dynamics_articulation", CHANNELS[2:4]), ("all_channels", CHANNELS)]:
        for stage in ["raw", "standardized"]:
            entries=[]
            for (dataset,key,work), parts in by.items():
                if any(c not in parts for c in channels):
                    continue
                values=np.array([finite_number(parts[c][f"{stage}_{m}"]) for c in channels for m in ["mean","range"]])
                peer_stage = "standardized" if stage == "raw" else "raw"
                peer = np.array([finite_number(parts[c][f"{peer_stage}_{m}"])
                                 for c in channels for m in ["mean", "range"]])
                if np.isfinite(values).all() and np.isfinite(peer).all():
                    entries.append((dataset,key,work,values))
            counts=Counter((e[2],e[0]) for e in entries)
            works=sorted({e[2] for e in entries if min(counts[e[2],"ASAP"],counts[e[2],"ATEPP"])>=2})
            if len(works)<10:
                continue
            entries=[e for e in entries if e[2] in works]
            x=np.stack([e[3]for e in entries]); y=np.array([e[0]=="ATEPP" for e in entries],float)
            g=np.array([e[2]for e in entries]); scores=np.full(len(y),np.nan)
            if shuffle_labels:
                rng=np.random.default_rng(SEED+1)
                for work in works:
                    ix=np.flatnonzero(g==work)
                    y[ix]=rng.permutation(y[ix])
            order=np.random.default_rng(SEED).permutation(works)
            folds={w:i%5 for i,w in enumerate(order)}
            for fold in range(5):
                test=np.array([folds[w]==fold for w in g]);train=~test
                weights=np.array([1/counts[w,"ATEPP" if label else "ASAP"]
                                  for w,label in zip(g,y)])[train]
                weights/=weights.sum()
                center=np.sum(x[train]*weights[:,None],axis=0)
                scale=np.sqrt(np.sum((x[train]-center)**2*weights[:,None],axis=0))
                scale=np.maximum(scale,1e-6)
                tx=(x[train]-center)/scale
                tx=np.c_[np.ones(len(tx)),tx]; vx=np.c_[np.ones(test.sum()),(x[test]-center)/scale]
                def objective(beta):
                    z=tx@beta;prob=1/(1+np.exp(-np.clip(z,-500,500)))
                    loss=np.sum(weights*(np.logaddexp(0,z)-y[train]*z)) + .01*np.dot(beta[1:],beta[1:])/2
                    grad=tx.T@(weights*(prob-y[train]));grad[1:]+=.01*beta[1:]
                    return loss,grad
                fit=minimize(objective,np.zeros(tx.shape[1]),jac=True,method="L-BFGS-B",
                             options={"maxiter":300,"gtol":1e-8})
                if not fit.success:
                    raise RuntimeError(f"Domain diagnostic did not converge: {fit.message}")
                scores[test]=vx@fit.x
            perwork=[auc(y[g==w],scores[g==w]) for w in works]
            boot=np.random.default_rng(SEED).choice(perwork,size=(1000,len(perwork)),replace=True).mean(axis=1)
            results.append(dict(features=subset,stage=stage,works=len(works),asap_n=int((y==0).sum()),
                                atepp_n=int((y==1).sum()),heldout_work_macro_auc=float(np.mean(perwork)),
                                work_bootstrap_lower=float(np.quantile(boot,.025)),
                                work_bootstrap_upper=float(np.quantile(boot,.975)),
                                interpretation="Dataset separability, not transcription error or transfer accuracy"))
            predictions.extend(dict(features=subset,stage=stage,dataset=e[0],key=e[1],work=e[2],
                                    heldout_fold=folds[e[2]],score=float(s))for e,s in zip(entries,scores))
    suffix="_label_shuffle" if shuffle_labels else ""
    write(f"domain_separability{suffix}.csv",results)
    write(f"domain_predictions{suffix}.csv",predictions)
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--reuse-extraction",action="store_true")
    args=parser.parse_args()
    protocol=dict(seed=SEED,comparison="Matched individual works/movements, source-summary available in ATEPP",
                  source_filters="ASAP aligned; robust articulation only; duplicate bytes removed. ATEPP saved quality-accepted normalized cohort; duplicate MIDI hashes removed.",
                  time_axis="Integer quarter-note grid within annotation coverage, exact score MIDI tempo-map conversion, piecewise-linear performance boundary interpolation, no feature-value interpolation",
                  special_intervals="bR-adjacent bins invalid; non-increasing anchors rejected; no extrapolation",
                  pedal_policy="No CC64 events means missing in both datasets for this diagnostic",
                  statistics="At least 2 performances per dataset/work/channel; equal-weight work medians and rank separation; 5-fold heldout-work logistic dataset classification with equal dataset/work weights and train-only standardization",
                  limits="Different artists, score editions, alignment methods and crops remain. These are observed dataset gaps, not pure transcription errors. ATEPP source MIDI is not local.")
    (OUT/"protocol.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+"\n")
    cross=read(OUT.parent/"asap_atepp_overlap/work_crosswalk.csv")
    amap={};tmap={}
    for r in cross:
        if r["status"]!="same_unit":continue
        title="Toccata" if (r["composer"],r["asap_title"])==("Schumann","Toccata_repeat") else r["asap_title"]
        work=r["composer"]+"::"+title
        amap[r["composer"],r["asap_title"]]=work
        for cid in r["atepp_composition_ids"].split(";"):
            assert cid not in tmap or tmap[cid]==work
            tmap[cid]=work
    atepp=read(AN/"atepp_normalization/feature_normalization/performance_summary.csv")
    available_ids={r["piece"].split("/")[1]for r in atepp}
    available_works={tmap[c]for c in available_ids if c in tmap}
    ameta=read(DATASETS/"ASAP/metadata.csv")
    annotations=json.loads((DATASETS/"ASAP/asap_annotations.json").read_text())
    dropkey="Beethoven/Piano_Sonatas/18-2/Schmitt01.mid"
    tasks=[(r["midi_performance"],amap[r["composer"],r["title"]])for r in ameta
           if (r["composer"],r["title"])in amap and amap[r["composer"],r["title"]]in available_works
           and annotations[r["midi_performance"]]["score_and_performance_aligned"]
           and r["midi_performance"]!=dropkey]
    if args.reuse_extraction:
        with np.load(OUT/"asap_quarter_raw.npz",allow_pickle=False)as z:
            manifest=json.loads(str(z["manifest"]))
            extracted=[dict(**r,**{k:z[f"{i}/{k}"].copy()for k in ["q","times","raw","mask"]})for i,r in enumerate(manifest)]
    else:
        extracted=[];failures=[]
        with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize)as pool:
            for i,r in enumerate(pool.map(extract,tasks,chunksize=8),1):
                if "error"in r:failures.append(r)
                else:extracted.append(r)
                if i%50==0:print(f"Quarter re-extraction {i}/{len(tasks)}",flush=True)
        write("conversion_failures.csv",[dict(key=r["key"],work=r["work"],error=r["error"])for r in failures])
        arrays={};manifest=[]
        for i,r in enumerate(extracted):
            manifest.append({k:v for k,v in r.items()if not isinstance(v,np.ndarray)})
            arrays.update({f"{i}/{k}":r[k]for k in ["q","times","raw","mask"]})
        np.savez_compressed(OUT/"asap_quarter_raw.npz",manifest=np.array(json.dumps(manifest)),**arrays)
    normalized=normalize(extracted)
    nkeys={r["key"]for r in normalized}
    allrecords=normalized+[r for r in extracted if r["key"]not in nkeys]
    asummaries=summary_rows(allrecords)
    tmeta={r["key"]:r for r in read(AN/"atepp_normalization/performance_catalog.csv")}
    hashkeys=defaultdict(list)
    for key,r in tmeta.items():hashkeys[r["midi_sha256"]].append(key)
    tdrop={k for keys in hashkeys.values()for k in sorted(keys)[1:]}
    tsummaries=[]
    for r in atepp:
        cid=r["piece"].split("/")[1]
        if cid not in tmap or tmap[cid]not in {v["work"]for v in extracted}or r["key"]in tdrop:continue
        tsummaries.append(dict(dataset="ATEPP",key=r["key"],work=tmap[cid],group=r["piece"],channel=r["channel"],
                               **{k:finite_number(r[k])for k in ["raw_mean","raw_range","standardized_mean","standardized_range"]},
                               total_intervals=int(r["beats"]),raw_valid=int(r["raw_valid"]),standardized_valid=int(r["standardized_valid"])))
    write("asap_quarter_summary.csv",asummaries)
    write("matched_atepp_summary.csv",tsummaries)
    write("time_axis_audit.csv",[r["audit"]for r in extracted])
    details,gap=compare(asummaries+tsummaries)
    domain=domain_check(asummaries+tsummaries)
    label_shuffle_control=domain_check(asummaries+tsummaries,shuffle_labels=True)
    native=read(AN/"atepp_research/performance_summary.csv")
    tids={r["key"]for r in tsummaries}
    quality=[]
    for dataset,rs in [("ASAP",[r["audit"]for r in extracted]),("ATEPP",[r for r in native if r["perf_id"]in tids])]:
        count=len(rs)
        no=sum(int(r["pedal_events"])==0 for r in rs)
        levels=[int(r["pedal_value_levels"])for r in rs if int(r["pedal_events"])>0]
        quality.append(dict(dataset=dataset,performances=count,no_cc64=no,no_cc64_percent=100*no/count,
                            median_nonempty_pedal_value_levels=float(np.median(levels)),
                            at_most_two_pedal_levels=sum(v<=2 for v in levels),
                            at_most_two_pedal_levels_percent=100*sum(v<=2 for v in levels)/len(levels)))
    write("pedal_source_summary.csv",quality)
    # Emit reusable normalized sequences without replacing any prior training input.
    arrays={};manifest=[]
    for i,r in enumerate(normalized):
        manifest.append(dict(key=r["key"],work=r["work"],group=r["group"],channels=CHANNELS,scales=r["scales"]))
        arrays.update({f"{i}/q":r["shared_q"],f"{i}/values":np.where(r["shared_mask"],r["standardized"],0).astype(np.float32),
                       f"{i}/valid":r["shared_mask"],f"{i}/raw":r["shared_raw"]})
    np.savez_compressed(OUT/"asap_quarter_normalized.npz",manifest=np.array(json.dumps(manifest)),**arrays)
    audit=[r["audit"]for r in extracted]
    totalnative=sum(r["native_intervals"]for r in audit)
    summary=dict(asap_attempted=len(tasks),asap_converted=len(extracted),asap_rejected=len(tasks)-len(extracted),
                 asap_normalized=len(normalized),atepp_performances=len(tids),
                 common_work_keys=len({r["work"]for r in extracted}&{r["work"]for r in tsummaries}),
                 native_intervals=totalnative,native_one_quarter_intervals=sum(r["native_one_quarter_intervals"]for r in audit),
                 native_nonquarter_percent=100*(1-sum(r["native_one_quarter_intervals"]for r in audit)/totalnative),
                 quarter_intervals=sum(r["quarter_intervals"]for r in audit),
                 inferred_boundaries=sum(r["inferred_quarter_boundaries"]for r in audit),
                 quarter_boundaries=sum(r["quarter_boundaries"]for r in audit),
                 special_or_invalid_quarter_intervals=sum(r["special_or_invalid_quarter_intervals"]for r in audit),
                 pedal=quality,domain=domain,label_shuffle_control=label_shuffle_control,
                 source_hashes={source_label(p):sha(p)for p in [DATASETS/"ASAP/metadata.csv",
                     DATASETS/"ASAP/asap_annotations.json",AN/"atepp_normalization/feature_normalization/performance_summary.csv",
                     AN/"atepp_normalization/performance_catalog.csv",OUT.parent/"asap_atepp_overlap/work_crosswalk.csv"]})
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
