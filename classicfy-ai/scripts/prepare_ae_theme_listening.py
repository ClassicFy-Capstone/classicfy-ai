"""Frozen AE retrieval and score-grounded theme excerpts for local listening.

Run from the repository root. The encode stage stores resumable, hash-bound
chunks outside Git. It never trains the model or fits a new scaler.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from fractions import Fraction
import re
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch

AI = Path(__file__).resolve().parents[1]
WORKSPACE = AI.parents[1]
sys.path.insert(0, str(AI / "src"))
from embedding.note_ae_indexed_data import IndexedNoteAECorpus, file_sha
from embedding.note_ae_training import load_note_ae_model
from embedding.note_performance_data import load_note_performance
from preprocessing.musicxml_score import _root as read_score_root, read_musicxml


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def encode_catalogue(args):
    torch.set_num_threads(1)
    corpus = IndexedNoteAECorpus(args.corpus)
    model, payload, _, _ = load_note_ae_model(args.checkpoint, corpus=corpus)
    protocol = {"checkpoint_sha256": file_sha(args.checkpoint),
                "epoch": payload["epoch"], "corpus_manifest_sha256": file_sha(args.corpus / "manifest.json"),
                "pooling": "arithmetic mean of all full 8-measure window e vectors, then L2 normalization",
                "distance": "1 - cosine similarity", "device": "cpu", "batch_size": args.batch_size}
    protocol_id = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    protocol["id"] = protocol_id
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "encoding_protocol.json", protocol)
    items = sorted(corpus.items.values(), key=lambda v: (v["window"]["N"], v["window"]["window_id"]))
    chunks = args.out / "chunks"
    chunks.mkdir(exist_ok=True)
    vectors = []
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(items), 1024):
            block = items[start:start + 1024]
            ids = np.array([v["window"]["window_id"] for v in block])
            path = chunks / f"{start:06d}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    if str(saved["protocol_id"]) != protocol_id or not np.array_equal(saved["window_id"], ids):
                        raise ValueError("Stale embedding chunk: use another --out directory")
                    e = saved["e"].copy()
            else:
                batches = []
                for offset in range(0, len(block), args.batch_size):
                    batch = corpus.collate(block[offset:offset + args.batch_size])
                    batches.append(model.encode_batch(batch)["e"].cpu().numpy())
                e = np.concatenate(batches)
                temp = path.with_suffix(".tmp.npz")
                np.savez_compressed(temp, window_id=ids, e=e, protocol_id=np.array(protocol_id))
                temp.replace(path)
            if e.shape != (len(block), model.config.latent_dim) or not np.isfinite(e).all():
                raise ValueError("Invalid encoded vectors")
            vectors.append(e)
            progress = {"status": "encoding", "windows": start + len(block), "total": len(items),
                        "seconds": round(time.monotonic() - started, 1)}
            write_json(args.out / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    vectors = np.concatenate(vectors)
    groups = defaultdict(list)
    for index, item in enumerate(items):
        groups[item["window"]["performance_id"]].append(index)
    pooled, rows = [], []
    for performance_id, indices in sorted(groups.items()):
        indices.sort(key=lambda i: items[i]["window"]["q_start"])
        vector = vectors[indices].astype(np.float64).mean(axis=0)
        norm = np.linalg.norm(vector)
        if norm <= 1e-12:
            raise ValueError("Zero pooled embedding")
        pooled.append(vector / norm)
        item = items[indices[0]]
        rows.append({**{k: item["window"][k] for k in
                        ("performance_id", "work_id", "recording_id", "split")},
                     "record_key": item["record_key"], "windows": len(indices)})
    np.savez_compressed(args.out / "catalogue_embeddings.npz", e=np.array(pooled),
                        performance_id=np.array([r["performance_id"] for r in rows]),
                        protocol_id=np.array(protocol_id))
    write_json(args.out / "catalogue.json", {"protocol": protocol, "recordings": rows})
    write_json(args.out / "progress.json", {"status": "completed", "windows": len(items),
                                            "recordings": len(rows), "epoch": payload["epoch"]})


def rank_candidates(rows, embeddings, anchor_index):
    """Exclude the same movement and byte-identical recording before ranking."""
    anchor = rows[anchor_index]
    distances = np.clip(1 - embeddings @ embeddings[anchor_index], 0, 2)
    indices = [i for i, row in enumerate(rows)
               if row["work_id"] != anchor["work_id"]
               and row["recording_id"] != anchor["recording_id"]]
    return sorted(indices, key=lambda i: (float(distances[i]), rows[i]["performance_id"])), distances


def select_near_far(rows, embeddings, anchor_index):
    """Nearest other work B, then farthest FROM A in B's work C.

    C is not selected by distance from B. Identical MIDI recordings cannot form
    the B/C comparison, even if their performance IDs differ.
    """
    order, distances = rank_candidates(rows, embeddings, anchor_index)
    if not order:
        raise ValueError("No different-work candidate")
    near = order[0]
    pool = [i for i, row in enumerate(rows)
            if row["work_id"] == rows[near]["work_id"]
            and row["recording_id"] != rows[near]["recording_id"]
            and row["recording_id"] != rows[anchor_index]["recording_id"]]
    if not pool:
        raise ValueError("Nearest work has no distinct comparison recording")
    far = min(pool, key=lambda i: (-float(distances[i]), rows[i]["performance_id"]))
    if distances[far] <= distances[near] + 1e-12:
        raise ValueError("No strictly farther recording in the nearest work")
    return near, far, pool, order, distances


def public_pack(pack):
    """Keep distances, identities of candidates and B/C roles out of initial JS.

    Result files are fetched only after the listener commits a choice. This is
    UI blinding, not access control: source YouTube players may show identities.
    """
    opaque = {key: hashlib.sha256((pack.get("storage_id", pack["id"]) + key).encode()).hexdigest()[:16]
              for key in pack["clips"]}
    public = {"schema": "ae-theme-choice-public-v2", "id": pack["id"],
              "storage_id": pack.get("storage_id", pack["id"]),
              "epoch": pack["model"]["epoch"], "catalogue": pack["catalogue"], "cases": [], "clips": {}}
    for key, clip in pack["clips"].items():
        public["clips"][opaque[key]] = {name: clip[name] for name in
                                        ("video_id", "youtube_url", "track_duration_seconds", "theme",
                                         "start_seconds", "end_seconds")}
    results = {}
    for case in pack["cases"]:
        public["cases"].append({"id": case["id"], "label": case["label"],
                                "anchor": opaque[case["anchor"]],
                                "candidates": sorted(opaque[c["clip"]] for c in case["candidates"]),
                                "result_url": f"results/{case['id']}.json",
                                "anchor_artist": pack["clips"][case["anchor"]]["artist"]})
        results[case["id"]] = {
            "pack_id": pack["id"], "case_id": case["id"],
            "near_clip": opaque[case["near"]], "far_clip": opaque[case["far"]],
            "distance_reference": "A", "distance_scope": pack["distance_scope"],
            "far_work_pool": case["far_work_pool"],
            "recordings": {opaque[key]: {name: pack["clips"][key][name] for name in
                                         ("performance_id", "artist", "title", "album", "album_date")}
                           for key in [case["anchor"]] + [c["clip"] for c in case["candidates"]]},
            "candidates": [{**candidate, "clip": opaque[candidate["clip"]]} for candidate in case["candidates"]]}
    return public, results


def theme_bounds(source, annotation):
    """Map explicit XML measure labels to the FIRST occurrence on the used route.

    XML labels are not assumed to be zero-based indices or literature bar numbers.
    Source windows carry the actual repeat-expanded quarter coordinates.
    """
    xml_path = Path(source["xml_path"])
    if file_sha(xml_path) != annotation["score_sha256"]:
        raise ValueError("Theme annotation score edition/hash changed")
    if file_sha(source["windows_file"]) != source["windows_sha256"]:
        raise ValueError("Theme window/repeat route source changed")
    root, _ = read_score_root(xml_path)
    # ElementTree's wildcard also accepts scores with a MusicXML namespace.
    part = root.find("{*}part")
    measures = part.findall("{*}measure")
    labels = [m.get("number") for m in measures]
    start_label, end_label = annotation["xml_measure_start"], annotation["xml_measure_end_inclusive"]
    if labels.count(start_label) != 1 or labels.count(end_label) != 1:
        raise ValueError("Ambiguous MusicXML measure labels")
    a, b = labels.index(start_label), labels.index(end_label)
    if a > b:
        raise ValueError("Reversed annotated measure range")
    visits = {}
    for window in json.loads(Path(source["windows_file"]).read_text()):
        for visit in window["measure_visits"]:
            if visit["key"] in visits and visits[visit["key"]] != visit:
                raise ValueError("Inconsistent repeated measure visit")
            visits[visit["key"]] = visit
    selected = [visits[f"{index}@1"] for index in range(a, b + 1)]
    for left, right in zip(selected, selected[1:]):
        if Fraction(left["q_start"]) + Fraction(left["duration_q"]) != Fraction(right["q_start"]):
            raise ValueError("Theme crosses a noncontiguous repeat route")
    start = float(Fraction(selected[0]["q_start"]))
    end = float(Fraction(selected[-1]["q_start"]) + Fraction(selected[-1]["duration_q"]))
    if annotation.get("omit_initial_rests"):
        if start != 0:
            raise ValueError("Initial-rest omission requires a theme starting at score origin")
        start = min(float(note.onset) for note in read_musicxml(xml_path).notes)
        if not 0 <= start < end:
            raise ValueError("First score attack is outside the annotated theme")
    return start, end


def time_at_quarter(performance, quarter):
    """Use observed note onset at the boundary, else valid beat interpolation.

    Never silently clamp/extrapolate a score position outside the accepted map.
    The median chord onset estimates its attack center; the boundary at the very
    start includes the first aligned attack (including any aligned grace note).
    """
    onsets = [m["onset_sec"] for m in performance.metadata
              if m.get("is_attack") and "onset_sec" in m
              and abs(float(m["score_onset_q"]) - quarter) < 1e-8]
    if onsets:
        return float(min(onsets) if quarter == 0 else np.median(onsets)), "aligned-note-onset"
    beat = performance.beats
    if not len(beat.q) or not beat.q[0] <= quarter <= beat.q[-1]:
        raise ValueError("Theme boundary outside the observed time map")
    interval = min(int(np.searchsorted(beat.q, quarter, side="right")) - 1, len(beat.valid) - 1)
    if not beat.valid[interval]:
        raise ValueError("Theme boundary uses an invalid estimated beat")
    return float(np.interp(quarter, beat.q, beat.seconds)), "valid-beat-interpolation"


def youtube_id(link):
    url = urlparse(link)
    if url.hostname in ("youtube.com", "www.youtube.com", "m.youtube.com"):
        value = parse_qs(url.query).get("v", [""])[0]
    elif url.hostname == "youtu.be":
        value = url.path.lstrip("/")
    else:
        raise ValueError("Not a YouTube recording URL")
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        raise ValueError("Invalid YouTube recording ID")
    return value


def build_pack(args):
    previous_path = args.destination / "listening.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else None
    public_path = args.ui / "data.js"
    previous_public = (json.loads(public_path.read_text().removeprefix("window.LISTENING_DATA = ").strip().removesuffix(";"))
                       if public_path.exists() else None)
    catalogue = json.loads((args.out / "catalogue.json").read_text())
    rows = catalogue["recordings"]
    protocol = catalogue["protocol"]
    with np.load(args.out / "catalogue_embeddings.npz", allow_pickle=False) as saved:
        if str(saved["protocol_id"]) != protocol["id"] or list(saved["performance_id"]) != [r["performance_id"] for r in rows]:
            raise ValueError("Catalogue identity mismatch")
        embeddings = saved["e"].copy()
    if not np.isfinite(embeddings).all() or not np.allclose(np.linalg.norm(embeddings, axis=1), 1):
        raise ValueError("Catalogue embeddings are not finite unit vectors")
    if file_sha(args.checkpoint) != protocol["checkpoint_sha256"] or file_sha(args.corpus / "manifest.json") != protocol["corpus_manifest_sha256"]:
        raise ValueError("Encoding source changed")
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    index_path = Path(manifest["source_index"])
    if file_sha(index_path) != manifest["source_index_sha256"]:
        raise ValueError("Source index changed")
    sources = {s["performance_id"]: s for s in json.loads(index_path.read_text())["sources"]}
    metadata = {r["perf_id"].zfill(5): r for r in read_csv(args.metadata)}
    availability = {r["perf_id"].zfill(5): r for r in read_csv(args.availability)}
    themes = json.loads(args.themes.read_text())
    if len({c["id"] for c in themes["cases"]}) != len(themes["cases"]):
        raise ValueError("Duplicate listening case IDs")
    by_id = {r["performance_id"]: i for i, r in enumerate(rows)}
    clips = {}

    def clip(i):
        row = rows[i]
        if row["performance_id"] in clips:
            return row["performance_id"]
        source = sources[row["performance_id"]]
        if source["work_key"] != row["work_id"] or source["recording_id"] != row["recording_id"]:
            raise ValueError("Catalogue/source identity mismatch")
        annotation = themes["works"][row["work_id"]]
        meta = metadata[row["record_key"]]
        if f"ATEPP/{meta['composition_id']}" != row["work_id"]:
            raise ValueError("Recording metadata work mismatch")
        performance = load_note_performance(source["performance_cache"])
        if performance.fingerprint != source["R_fingerprint"]:
            raise ValueError("Performance source changed")
        q_start, q_end = theme_bounds(source, annotation)
        start, start_method = time_at_quarter(performance, q_start)
        end, end_method = time_at_quarter(performance, q_end)
        if not 0 <= start < end <= float(meta["track_duration"]):
            raise ValueError("Invalid excerpt time/duration")
        snapshot = availability.get(row["record_key"])
        video_id = youtube_id(meta["youtube_links"])
        if snapshot and snapshot["youtube_video_id"] != video_id:
            raise ValueError("Availability refers to another recording")
        clips[row["performance_id"]] = {
            **row, "composer": meta["composer"], "title": meta["track"], "artist": meta["artist"],
            "video_id": video_id, "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
            "track_duration_seconds": float(meta["track_duration"]),
            "album": meta["album"], "album_date": meta["album_date"],
            "theme": annotation, "q_start": q_start, "q_end": q_end,
            "start_seconds": round(start, 3), "end_seconds": round(end, 3),
            "time_mapping": {"start_method": start_method, "end_method": end_method,
                             "basis": "ATEPP MIDI clock, estimated score alignment",
                             "youtube_offset_seconds": 0, "youtube_offset_verified": False,
                             "audio_listened_by_agent": False},
            "availability": {"state": "previously_playable" if snapshot else "unverified",
                             "checked_at_utc": snapshot["youtube_checked_at_utc"] if snapshot else None,
                             "full_playback_verified": False},
            "provenance": {"score_sha256": file_sha(source["xml_path"]),
                           "R_fingerprint": performance.fingerprint,
                           "R_array_sha256": file_sha(Path(source["performance_cache"]) / "performance.npz"),
                           "windows_sha256": file_sha(source["windows_file"])}}
        return row["performance_id"]

    cases = []
    for definition in themes["cases"]:
        ai = by_id[definition["anchor"]]
        near, far, far_pool, order, distances = select_near_far(rows, embeddings, ai)
        near_clip, far_clip = clip(near), clip(far)
        if (clips[near_clip]["provenance"]["score_sha256"] != clips[far_clip]["provenance"]["score_sha256"]
                or clips[near_clip]["q_start"] != clips[far_clip]["q_start"]
                or clips[near_clip]["q_end"] != clips[far_clip]["q_end"]):
            raise ValueError("B/C must use the identical score edition and theme interval")
        ranks = {i: rank for rank, i in enumerate(order, 1)}
        candidates = [{"clip": clip(ci), "role": role, "global_rank": ranks[ci],
                       "cosine_distance_to_anchor": float(distances[ci]),
                       "same_artist_as_anchor": metadata[rows[ci]["record_key"]]["artist"] == metadata[rows[ai]["record_key"]]["artist"]}
                      for ci, role in [(near, "near"), (far, "far")]]
        ranked_rows = []
        for rank, ci in enumerate(order, 1):
            r = rows[ci]
            m = metadata[r["record_key"]]
            ranked_rows.append({"rank": rank, **r, "cosine_distance": float(distances[ci]),
                                "artist": m["artist"], "composer": m["composer"], "title": m["track"]})
        with (args.out / f"ranked_{rows[ai]['record_key']}.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(ranked_rows[0]))
            writer.writeheader()
            writer.writerows(ranked_rows)
        cases.append({"id": definition["id"], "label": definition["label"], "anchor": clip(ai),
                      "candidate_count": len(order), "candidate_works": len({rows[i]["work_id"] for i in order}),
                      "near": near_clip, "far": far_clip, "far_work_pool": len(far_pool),
                      "candidates": candidates})
    pack = {"schema": "ae-theme-choice-organizer-v2", "model": protocol,
            "storage_id": themes.get("storage_id", protocol["id"]),
            "catalogue": {"recordings": len(rows), "works": len({r['work_id'] for r in rows}),
                          "full_windows": sum(r["windows"] for r in rows)},
            "selection": "B = nearest to A across all other works; C = farthest FROM A among recordings of B's work, excluding identical MIDI recordings; no availability reranking",
            "distance_scope": "whole-recording mean embedding, NOT theme-only distance or evidence of perceived similarity",
            "themes_sha256": file_sha(args.themes), "metadata_sha256": file_sha(args.metadata),
            "cases": cases, "clips": clips}
    pack["id"] = hashlib.sha256(json.dumps(pack, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    work_questions = defaultdict(list)
    for case in cases:
        for work in {clips[case["anchor"]]["work_id"], clips[case["near"]]["work_id"]}:
            work_questions[work].append(case["id"])
    if themes.get("max_questions_per_work") is not None and any(
            len(ids) > themes["max_questions_per_work"] for ids in work_questions.values()):
        raise ValueError("A work is reused across listening questions")
    if previous is not None and previous["id"] != pack["id"]:
        write_json(args.destination / "history" / f"{previous['id']}.json", previous)
    write_json(args.destination / "listening.json", pack)
    args.ui.mkdir(parents=True, exist_ok=True)
    public, results = public_pack(pack)
    # Migrate only exactly unchanged questions. Removed/replaced questions keep
    # their previous browser save and are archived separately from active scores.
    compatible = (previous is not None and previous_public is not None
                  and previous_public.get("storage_id", previous_public["id"]) == public["storage_id"]
                  and all(previous[k] == pack[k] for k in ("model", "selection", "distance_scope")))
    migrations = []
    if compatible:
        unchanged = {c["id"] for c in previous["cases"] if c in cases and all(
            previous["clips"][key] == clips.get(key) for key in [c["anchor"], c["near"], c["far"]])}
        earlier = previous_public.get("migration_sources", [
            {"pack_id": key, "case_ids": [c["id"] for c in previous["cases"]]}
            for key in previous_public.get("compatible_pack_ids", [])])
        for source in earlier + [{"pack_id": previous["id"], "case_ids": list(unchanged)}]:
            keep = sorted(unchanged.intersection(source["case_ids"]))
            if keep and source["pack_id"] != pack["id"]:
                migrations.append({"pack_id": source["pack_id"], "case_ids": keep})
    public["migration_sources"] = migrations
    public["compatible_pack_ids"] = sorted({source["pack_id"] for source in migrations})
    (args.ui / "data.js").write_text("window.LISTENING_DATA = " + json.dumps(public, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c") + ";\n", encoding="utf-8")
    for case_id, result in results.items():
        write_json(args.ui / f"results/{case_id}.json", result)
    write_json(args.destination / "verification.json", {
        "status": "passed", "pack_id": pack["id"], "checkpoint_epoch": protocol["epoch"],
        "catalogue_recordings": len(rows), "encoded_windows": sum(r["windows"] for r in rows),
        "cases": len(cases), "unique_recordings": len(clips),
        "unique_question_works": len(work_questions),
        "maximum_questions_per_work": max(map(len, work_questions.values())),
        "work_question_counts": {work: len(ids) for work, ids in sorted(work_questions.items())},
        "nearest_global_ranks": [case["candidates"][0]["global_rank"] for case in cases],
        "near_far_same_work": True, "far_is_farthest_from_A_in_B_work": True,
        "near_far_identical_score_theme": True,
        "initial_page_has_no_model_answer_or_distances": True,
        "anchor_candidate_same_work_or_recording_pairs": 0, "source_hashes_checked": True,
        "theme_boundaries_within_observed_alignment": True,
        "human_listening_completed": False, "youtube_offsets_verified": False})
    print(f"Prepared {len(cases)} cases / {len(clips)} recordings: {args.ui / 'index.html'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=WORKSPACE / "diagnostics/full_corpus_reconstruction/run01/corpus")
    parser.add_argument("--checkpoint", type=Path, default=WORKSPACE / "diagnostics/full_corpus_reconstruction/run01/training_resume01/best.pt")
    parser.add_argument("--out", type=Path, default=WORKSPACE / "diagnostics/ae_theme_listening/run01")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--stage", choices=("encode", "build", "all"), default="all")
    parser.add_argument("--metadata", type=Path, default=WORKSPACE / "datasets/ATEPP/ATEPP-metadata-1.2.csv")
    parser.add_argument("--availability", type=Path, default=AI / "analysis/atepp_youtube_availability/ae_metadata_playable.csv")
    parser.add_argument("--themes", type=Path, default=AI / "analysis/ae_theme_listening/themes.json")
    parser.add_argument("--destination", type=Path, default=AI / "analysis/ae_theme_listening")
    parser.add_argument("--ui", type=Path, default=AI / "tools/ae_theme_listening")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.stage in ("encode", "all"):
        encode_catalogue(args)
    if args.stage in ("build", "all"):
        build_pack(args)


if __name__ == "__main__":
    main()
