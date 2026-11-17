"""Fix one Schubert theme as A; compare nearest/farthest recordings of one Bach passage."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np

from listening_study import archive_index, audio_name, create_audio, digest, write_csv
from prepare_listening_study import resolve_source
from prepare_theme_comparison import boundary_ids, passage
from preprocessing import ASAPLoader
from validate_embedding import BLOCKS, collect_groups, summarize, weighted_distances
from validate_feature_normalization import collect_sample, init_worker, source_signature, write_cache


ANCHOR_WORK = "Schubert/Impromptu_op.90_D.899/3"
ANCHOR_KEY = ANCHOR_WORK + "/LeeSH08M.mid"
TARGET_WORK = "Bach/Prelude/bwv_857"
LABELS = {ANCHOR_WORK: "슈베르트 · 즉흥곡 Op.90 No.3 · 첫 주제 1–8마디",
          TARGET_WORK: "바흐 · 전주곡 BWV 857 · 1–8마디"}


def work_vectors(workspace, work, cache, metadata, loader):
    asap, nasap = workspace / "datasets/ASAP", workspace / "datasets/nASAP"
    if not cache.exists():
        init_worker(asap, nasap)
        records = [collect_sample(key) for key in sorted(metadata)
                   if str(Path(key).parent) == work and loader.get_sample(key).aligned]
        write_cache(cache, records, source_signature(asap, nasap))
    groups, _, scales, provenance = collect_groups(SimpleNamespace(
        asap_root=asap, nasap_root=nasap, cache=cache,
        min_performances=5, min_beats=32, min_coverage=.5))
    if len(groups) != 1 or groups[0]["piece"] != work + "/midi_score.mid":
        raise ValueError("Expected one unambiguous score grid per work")
    group = groups[0]
    sample = loader.get_sample(group["keys"][0])
    end = sample.score_beats.index(sample.score_downbeats[8])
    indices = np.flatnonzero(group["shared_indices"] < end)
    if end != 32 or len(indices) < 24:
        raise ValueError("Require at least 24/32 common valid opening intervals")
    values = group["values"][:, :, indices]
    return group["keys"], np.array([summarize(v) for v in values]), dict(
        reference_recordings=len(group["keys"]), common_intervals=len(indices), total_intervals=end,
        scales=scales, provenance=provenance, raw_cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest()), values


def select_recordings(anchor, keys, vectors):
    if len(keys) < 2:
        raise ValueError("Need two recordings of the candidate work")
    distances = weighted_distances(anchor[None, :], vectors, "all")[0]
    rows = []
    for i, key in enumerate(keys):
        rows.append(dict(key=key, distance=float(distances[i]),
                         **{block: float(weighted_distances(anchor[None, :], vectors[i:i+1], "only_" + block)[0, 0])
                            for block in BLOCKS}))
    rows.sort(key=lambda row: (row["distance"], row["key"]))
    if rows[-1]["distance"] - rows[0]["distance"] <= 1e-12:
        raise ValueError("All candidate distances are tied; no near/far comparison")
    return rows


def build(args):
    ai = Path(__file__).resolve().parents[1]
    workspace = ai.parent.parent
    loader = ASAPLoader(workspace / "datasets/ASAP", workspace / "datasets/nASAP")
    with (workspace / "datasets/ASAP/metadata.csv").open(encoding="utf-8-sig", newline="") as stream:
        metadata = {row["midi_performance"]: row for row in csv.DictReader(stream)}
    args.out.mkdir(parents=True, exist_ok=True)
    reference = {}
    work_data = {}
    for role, work in (("anchor", ANCHOR_WORK), ("candidate", TARGET_WORK)):
        work_data[work] = work_vectors(workspace, work, args.out / f"{role}_features_raw.npz", metadata, loader)
        reference[work] = work_data[work][2]
    akeys, avectors, _, aval = work_data[ANCHOR_WORK]
    anchor = avectors[akeys.index(ANCHOR_KEY)]
    keys, vectors, _, values = work_data[TARGET_WORK]
    boundaries = {work: boundary_ids(workspace / "datasets/nASAP" / work / "xml_score.musicxml", quarters)
                  for work, quarters in ((ANCHOR_WORK, 8), (TARGET_WORK, 4))}
    candidates = []
    for i, key in enumerate(keys):
        if audio_name(metadata[key]):
            passage(loader.get_sample(key), boundaries[TARGET_WORK])
            candidates.append(i)
    ordered = select_recordings(anchor, [keys[i] for i in candidates], vectors[candidates])
    near, far = ordered[0]["key"], ordered[-1]["key"]
    ranked = [dict(rank=i + 1, **row) for i, row in enumerate(ordered)]
    write_csv(args.out / "candidate_ranking.csv", ranked)
    selection = dict(anchor_key=ANCHOR_KEY, candidate_work=TARGET_WORK, near_key=near, far_key=far,
                     candidate_count=len(candidates), ranking=ranked, reference=reference,
                     rule="Fix Schubert LeeSH08M measures 1–8 first; fix Bach BWV857 measures 1–8; nearest and farthest among that work's available recordings. Per-work full-piece reference normalization, 14D summaries, five equal feature blocks. No post-listening reselection.")
    (args.out / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n")
    np.savez_compressed(args.out / "comparison_vectors.npz", anchor=anchor, keys=np.array(keys), vectors=vectors,
                        anchor_values=aval[akeys.index(ANCHOR_KEY)], candidate_values=values)
    print(f"Selected {len(candidates)} candidates: nearest {ordered[0]['distance']:.6f}, farthest {ordered[-1]['distance']:.6f}", flush=True)
    if args.plan_only:
        return
    archive = archive_index(workspace / "datasets/listening_evaluation/cache") if args.download else None
    clips = []
    for key in (ANCHOR_KEY, near, far):
        work = str(Path(key).parent)
        row, sample = metadata[key], loader.get_sample(key)
        start, end = passage(sample, boundaries[work])
        source, offset = resolve_source(args.audio_root, row, workspace)
        if source is None and archive is None:
            raise ValueError("Source WAV missing; supply --audio-root or --download")
        if row["end"] and float(row["start"] or 0) + end > float(row["end"]):
            raise ValueError("Passage exceeds the ASAP audio trim")
        spec = dict(key=key, part=1, measure_start=1, measure_end_inclusive=8,
                    audio="audio/" + digest([key, 1, 8, "cross-theme-v1"])[:16] + ".wav",
                    asap_start_seconds=start, asap_end_seconds=end,
                    source_start_seconds=offset + start, source_end_seconds=offset + end)
        receipt = create_audio(audio_name(row), [spec], args.pack, args.out / "receipts", local=source, archive=archive)
        clips.append(receipt["clips"][0])
        print(f"Audio ready {len(clips)}/3", flush=True)
    public = dict(schema_version=1, task_count=1, participants={}, feedback={"url": "feedback.json"}, section={
        "mode": "cross", "work_title": "A와 닮은 연주를 찾아보세요.",
        "label": "기준 A의 첫 주제 · 다른 한 작품의 후보 X/Y",
        "description": "먼저 슈베르트의 기준 A를 듣고, 바흐의 X와 Y 중 연주 성향이 더 비슷하게 들리는 쪽을 고르세요. X/Y는 같은 작품의 같은 1–8마디입니다. 선율 자체보다 빠르기의 흐름·강약·연결감·울림을 비교하세요."})
    feedback = dict(participants={}, interpretation="피처 기준의 가까움·멀어짐과 청취 판단을 비교합니다. 사람의 유사도 판단에 대한 객관적 정답은 아닙니다.")
    orders = {}
    for number, order in ((1, (far, near)), (2, (near, far)), (3, (far, near))):
        mapping = dict(zip(("X", "Y"), order))
        orders[f"P{number}"] = dict(reference_key=ANCHOR_KEY, x_key=order[0], y_key=order[1])
        stimuli = {}
        for label, key in {"A": ANCHOR_KEY, **mapping}.items():
            clip = next(c for c in clips if c["key"] == key)
            stimuli[label] = [dict(audio=clip["audio"], part=1, label=LABELS[str(Path(key).parent)])]
        public["participants"][f"P{number}"] = [dict(task_id="C01", mode="cross", stimuli=stimuli)]
        expected = {}
        for field, metric in (("overall", "distance"), *((name.lower(), name) for name in BLOCKS)):
            d = {label: next(r[metric] for r in ordered if r["key"] == key) for label, key in mapping.items()}
            choice = "tie" if abs(d["X"] - d["Y"]) <= 1e-12 else min(d, key=d.get)
            expected[field] = dict(choice=choice, distances=d)
        feedback["participants"][f"P{number}"] = expected
    public["study_id"] = digest(dict(public=public, clip_hashes=[c["audio_sha256"] for c in clips], selection=selection))[:20]
    feedback["study_id"] = public["study_id"]
    for name in ("index.html", "app.js", "style.css"):
        shutil.copyfile(ai / "tools/listening_evaluation" / name, args.pack / name)
    (args.pack / "study.js").write_text("window.LISTENING_STUDY = " + json.dumps(public, ensure_ascii=False) + ";\n")
    (args.pack / "feedback.json").write_text(json.dumps(feedback, ensure_ascii=False, indent=2) + "\n")
    (args.pack / "LICENSE.txt").write_text(
        "MAESTRO v2.0.0, Google LLC / International Piano-e-Competition; ASAP mapping.\n"
        "CC BY-NC-SA 4.0 https://creativecommons.org/licenses/by-nc-sa/4.0/\n"
        "https://magenta.withgoogle.com/datasets/maestro\nhttps://github.com/fosfrancesco/asap-dataset\n"
        "Modification: excerpts only; no gain normalization, resampling, speed changes or synthesis.\n")
    manifest = dict(study_id=public["study_id"], public=public, selection=selection, schedules=orders, clips=clips,
                    boundary_policy="XML note IDs through nASAP, checked against ASAP downbeats within 150ms; 80ms pre-roll; stop before measure 9. Original PCM, speed and level preserved.",
                    interpretation="One selected exploratory cross-work comparison; embedding agreement is not preference accuracy or AE performance.")
    (args.out / "organizer.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Pack: {args.pack}\nStudy ID: {public['study_id']}")


def main():
    workspace = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--out", type=Path, default=workspace / "datasets/listening_cross_work_theme")
    parser.add_argument("--pack", type=Path, default=workspace / "datasets/listening_cross_work_theme/pack")
    args = parser.parse_args()
    try:
        build(args)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, str(error) + "\n")


if __name__ == "__main__":
    main()
