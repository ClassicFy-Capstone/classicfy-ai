"""Audit local ASAP MIDI identity and trace confirmed matches into saved analyses."""
import csv
import hashlib
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import pretty_midi

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent))
from _support import REPO_ROOT, DATASETS, source_label
ROOT = REPO_ROOT.parent
DATA = DATASETS / "ASAP"
ANALYSIS = REPO_ROOT / "classicfy-ai/analysis"


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def write(name, rows):
    if not rows:
        return
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    rows = list(csv.DictReader((DATA / "metadata.csv").open(encoding="utf-8-sig")))
    annotations = json.loads((DATA / "asap_annotations.json").read_text())
    catalog, groups = [], {name: defaultdict(list) for name in ["bytes", "events", "shifted_events", "shifted_notes"]}
    warnings.simplefilter("ignore", RuntimeWarning)
    for index, row in enumerate(rows, 1):
        key = row["midi_performance"]
        path = DATA / key
        midi = pretty_midi.PrettyMIDI(str(path))
        notes = sorted((float(n.start), float(n.end), int(n.pitch), int(n.velocity))
                       for instrument in midi.instruments if not instrument.is_drum for n in instrument.notes)
        pedals = sorted((float(c.time), int(c.value))
                        for instrument in midi.instruments if not instrument.is_drum
                        for c in instrument.control_changes if c.number == 64)
        # Ignore track/container metadata; preserve note timing, velocity, length and CC64.
        # Times rounded to one microsecond. A second key removes constant leading silence.
        first = notes[0][0] if notes else 0.0
        def rounded_notes(offset):
            return sorted((round(a-offset, 6), round(b-offset, 6), p, v) for a,b,p,v in notes)
        def rounded_pedals(offset):
            return sorted((round(t-offset, 6), v) for t,v in pedals)
        hashes = {
            "bytes": hashlib.sha256(path.read_bytes()).hexdigest(),
            "events": digest([rounded_notes(0), rounded_pedals(0)]),
            "shifted_events": digest([rounded_notes(first), rounded_pedals(first)]),
            "shifted_notes": digest(rounded_notes(first)),
        }
        item = dict(key=key, composer=row["composer"], title=row["title"],
                    midi_score=row["midi_score"], notes=len(notes), pedals=len(pedals),
                    aligned=annotations[key]["score_and_performance_aligned"],
                    beats=len(annotations[key]["performance_beats"]),
                    source=row["maestro_midi_performance"], start=row["start"], end=row["end"],
                    **{f"sha256_{k}": v for k,v in hashes.items()})
        catalog.append(item)
        for method, value in hashes.items():
            groups[method][value].append(item)
        if index % 100 == 0:
            print(f"Parsed {index}/{len(rows)}", flush=True)
    write("performance_identity_catalog.csv", catalog)
    duplicates = []
    for method, values in groups.items():
        for value, items in values.items():
            if len(items) > 1:
                for item in items:
                    duplicates.append(dict(method=method, identity_hash=value, group_size=len(items), **item))
    write("duplicate_members.csv", duplicates)
    summary = dict(total_performances=len(rows), groups={
        method: [dict(hash=value, keys=[item["key"] for item in items])
                 for value, items in values.items() if len(items) > 1]
        for method, values in groups.items()},
        metadata_sha256=hashlib.sha256((DATA / "metadata.csv").read_bytes()).hexdigest(),
        scope="Local ASAP complete MIDI files; exact bytes and note/CC64 events at microsecond precision. Does not exhaustively detect edited/transcribed/partially cropped recordings.")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
