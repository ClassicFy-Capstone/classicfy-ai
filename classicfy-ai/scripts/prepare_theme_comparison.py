"""Prepare two recordings of Schubert D899/3, opening theme excerpt, measures 1–8."""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np

from listening_study import archive_index, audio_name, create_audio, digest, write_csv
from prepare_listening_study import resolve_source
from preprocessing import ASAPLoader
from preprocessing.note_alignment import load_match
from validate_embedding import BLOCKS, collect_groups, summarize, weighted_distances
from validate_feature_normalization import collect_sample, init_worker, source_signature, write_cache


WORK = "Schubert/Impromptu_op.90_D.899/3"
SECTION = {
    "work_title": "슈베르트 · 즉흥곡 Op.90 No.3",
    "label": "첫 주제의 1–8마디 · 두 연주 비교",
    "description": "X와 Y는 같은 작품의 1–8마디입니다. 선율의 호흡, 빠르기 변화, 강약, 음의 연결과 울림을 비교하세요. 같은 악보 범위라 연주에 따라 재생 길이는 다릅니다.",
}


def boundary_ids(score, expected_quarters=8):
    """Use XML note IDs to avoid assuming match beats equal XML quarter notes."""
    measures = ET.parse(score).getroot().find("part").findall("measure")[:9]
    if [m.get("number") for m in measures] != [str(i) for i in range(1, 10)]:
        raise ValueError("Expected consecutive opening measures 1–9")
    if any(m.find(".//repeat") is not None for m in measures):
        raise ValueError("Opening contains a repeat; review the occurrence mapping")
    divisions, boundaries = 1, {}
    for measure in measures:
        divisions = int(measure.findtext("attributes/divisions", str(divisions)))
        cursor = previous = extent = Fraction(0)
        ids, pitches = [], set()
        for item in measure:
            duration = Fraction(int(item.findtext("duration", "0")), divisions)
            if item.tag == "backup":
                cursor -= duration
            elif item.tag == "forward":
                cursor += duration
                extent = max(extent, cursor)
            elif item.tag == "note":
                onset = previous if item.find("chord") is not None else cursor
                pitch = tuple(item.findtext("pitch/" + field, "0") for field in ("step", "alter", "octave"))
                tied = any(tie.get("type") == "stop" for tie in item.findall("tie"))
                if item.find("rest") is None and onset == 0 and item.get("id") and not tied and pitch not in pitches:
                    # Two written voices can share one piano attack and one match.
                    ids.append(item.get("id") + "-1")
                    pitches.add(pitch)
                if item.find("chord") is None:
                    previous = cursor
                    cursor += duration
                extent = max(extent, onset + duration)
        if extent != expected_quarters:
            raise ValueError(f"Expected {expected_quarters} XML quarter notes per measure")
        if measure.get("number") in {"1", "9"}:
            if not ids:
                raise ValueError("Missing note IDs at the passage boundary")
            boundaries[measure.get("number")] = ids
    return boundaries


def passage(sample, boundaries):
    if not sample.aligned or not sample.robust_note_alignment or not sample.note_alignment_path:
        raise ValueError("Both recordings require aligned, robust nASAP matches")
    alignment = load_match(sample.note_alignment_path)
    matched = {score.note_id: performed for score, performed in alignment.matches}
    starts = []
    for measure in ("1", "9"):
        if not set(boundaries[measure]).issubset(matched):
            raise ValueError(f"Unmatched boundary notes at measure {measure}")
        starts.append(min(matched[n].onset for n in boundaries[measure]))
    if len(sample.performance_downbeats) < 9:
        raise ValueError("Not enough downbeat annotations")
    for onset, annotated in zip(starts, (sample.performance_downbeats[0], sample.performance_downbeats[8])):
        if abs(onset - annotated) > .15:
            raise ValueError("XML note boundary disagrees with ASAP downbeat")
    # Include the opening attack; stop before any note of measure 9.
    return max(0., starts[0] - .08), starts[1]


def select_contrast(workspace, out, metadata, loader, boundaries):
    """Rank all audio-mapped pairs on the exact passage, five equally weighted blocks."""
    asap = workspace / "datasets/ASAP"
    nasap = workspace / "datasets/nASAP"
    cache = out / "theme_features_raw.npz"
    if not cache.exists():
        init_worker(asap, nasap)
        keys = sorted(k for k in metadata if str(Path(k).parent) == WORK)
        records = [collect_sample(key) for key in keys if loader.get_sample(key).aligned]
        write_cache(cache, records, source_signature(asap, nasap))
    groups, _, scales, provenance = collect_groups(SimpleNamespace(
        asap_root=asap, nasap_root=nasap, cache=cache,
        min_performances=5, min_beats=32, min_coverage=.5))
    if len(groups) != 1 or groups[0]["piece"] != WORK + "/midi_score.mid":
        raise ValueError("Expected one common score grid for this work")
    group = groups[0]
    sample = loader.get_sample(group["keys"][0])
    end_index = sample.score_beats.index(sample.score_downbeats[8])
    selected = np.flatnonzero(group["shared_indices"] < end_index)
    if end_index != 32 or len(selected) < 24:
        raise ValueError("Opening passage requires at least 24/32 common valid intervals")
    vectors = np.array([summarize(values[:, selected]) for values in group["values"]])
    distances = weighted_distances(vectors, vectors, "all")
    candidates = []
    for i, key in enumerate(group["keys"]):
        if not audio_name(metadata[key]):
            continue
        other = loader.get_sample(key)
        np.testing.assert_array_equal(sample.score_beats, other.score_beats)
        passage(other, boundaries)
        candidates.append(i)
    pairs = []
    for position, i in enumerate(candidates):
        for j in candidates[position + 1:]:
            pairs.append(dict(key_x=group["keys"][i], key_y=group["keys"][j],
                              distance=float(distances[i, j]),
                              **{name: float(weighted_distances(vectors[i:i+1], vectors[j:j+1], "only_" + name)[0, 0])
                                 for name in BLOCKS}))
    if not pairs:
        raise ValueError("Not enough audio-mapped recordings")
    pairs.sort(key=lambda row: (-row["distance"], row["key_x"], row["key_y"]))
    ranked = [dict(rank=i + 1, **row) for i, row in enumerate(pairs)]
    write_csv(out / "theme_pair_ranking.csv", ranked)
    selection = dict(rule="Maximum first-theme excerpt distance; five equal-weight feature blocks, 14 mean/range coordinates; full-work reference normalization.",
                     reference_recordings=len(group["keys"]), audio_recordings=len(candidates), pairs=len(pairs),
                     common_intervals=len(selected), total_intervals=end_index, winner=ranked[0],
                     scales=scales, provenance=provenance,
                     raw_cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest())
    (out / "theme_selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n")
    np.savez_compressed(out / "theme_vectors.npz", keys=np.array(group["keys"]), vectors=vectors,
                        values=group["values"][:, :, selected], interval_indices=group["shared_indices"][selected])
    print(f"Selected rank 1/{len(pairs)} from {len(candidates)} recordings: {pairs[0]['key_x']} / {pairs[0]['key_y']} (distance {pairs[0]['distance']:.6f})", flush=True)
    return [pairs[0]["key_x"], pairs[0]["key_y"]], selection


def build(args):
    ai = Path(__file__).resolve().parents[1]
    workspace = ai.parent.parent
    loader = ASAPLoader(workspace / "datasets/ASAP", workspace / "datasets/nASAP")
    with (workspace / "datasets/ASAP/metadata.csv").open(encoding="utf-8-sig", newline="") as stream:
        metadata = {row["midi_performance"]: row for row in csv.DictReader(stream)}
    score = workspace / "datasets/nASAP" / WORK / "xml_score.musicxml"
    boundaries = boundary_ids(score)
    args.out.mkdir(parents=True, exist_ok=True)
    keys, selection = select_contrast(workspace, args.out, metadata, loader, boundaries)
    if args.plan_only:
        return
    archive = archive_index(workspace / "datasets/listening_evaluation/cache") if args.download else None
    clips = []
    for key in keys:
        sample = loader.get_sample(key)
        start, end = passage(sample, boundaries)
        row = metadata[key]
        source, offset = resolve_source(args.audio_root, row, workspace)
        if source is None and archive is None:
            raise ValueError("Source WAV missing; supply --audio-root or --download")
        if row["end"] and float(row["start"] or 0) + end > float(row["end"]):
            raise ValueError("Passage exceeds the ASAP audio trim")
        spec = dict(key=key, part=1, measure_start=1, measure_end_inclusive=8,
                    audio="audio/" + digest([key, 1, 8, "theme-v1"])[:16] + ".wav",
                    asap_start_seconds=start, asap_end_seconds=end,
                    source_start_seconds=offset + start, source_end_seconds=offset + end)
        receipt = create_audio(audio_name(row), [spec], args.pack, args.out / "receipts", local=source, archive=archive)
        clips.append(receipt["clips"][0])
        print(f"Ready {len(clips)}/2: {end-start:.2f}s", flush=True)
    public = dict(schema_version=1, task_count=1, section=SECTION, participants={})
    for number, order in ((1, keys), (2, keys[::-1]), (3, keys)):
        stimuli = {label: [{"audio": next(c["audio"] for c in clips if c["key"] == key),
                            "part": 1, "label": SECTION["label"]}]
                   for label, key in zip(("X", "Y"), order)}
        public["participants"][f"P{number}"] = [dict(task_id="T01", mode="pair", stimuli=stimuli)]
    public["study_id"] = digest(dict(public=public, hashes=[c["audio_sha256"] for c in clips]))[:20]
    args.pack.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "app.js", "style.css"):
        shutil.copyfile(ai / "tools/listening_evaluation" / name, args.pack / name)
    (args.pack / "study.js").write_text("window.LISTENING_STUDY = " + json.dumps(public, ensure_ascii=False) + ";\n")
    (args.pack / "LICENSE.txt").write_text(
        "MAESTRO v2.0.0, Google LLC / International Piano-e-Competition; ASAP mapping.\n"
        "CC BY-NC-SA 4.0 https://creativecommons.org/licenses/by-nc-sa/4.0/\n"
        "https://magenta.withgoogle.com/datasets/maestro\nhttps://github.com/fosfrancesco/asap-dataset\n"
        "Modification: excerpts only; no gain normalization, resampling, speed changes or synthesis.\n")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = dict(study_id=public["study_id"], public=public, clips=clips,
                    score_sha256=hashlib.sha256(score.read_bytes()).hexdigest(), boundary_note_ids=boundaries,
                    selection=selection,
                    boundary_policy="XML note IDs mapped through nASAP; 80ms opening pre-roll; stop at earliest measure-9 onset; check ASAP downbeats within 150ms.",
                    interpretation="Opening theme excerpt, not a complete formal section or an AE evaluation. No human responses collected.")
    (args.out / "theme_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Pack: {args.pack}\nStudy ID: {public['study_id']}")


def main():
    workspace = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download only the needed MAESTRO prefixes")
    parser.add_argument("--plan-only", action="store_true", help="Rank pairs without downloading audio or generating a pack")
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--pack", type=Path, default=workspace / "datasets/listening_theme_contrast/pack")
    parser.add_argument("--out", type=Path, default=workspace / "datasets/listening_theme_contrast")
    args = parser.parse_args()
    try:
        build(args)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, str(error) + "\n")


if __name__ == "__main__":
    main()
