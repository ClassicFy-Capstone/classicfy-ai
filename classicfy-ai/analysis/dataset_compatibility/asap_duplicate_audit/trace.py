"""Trace confirmed identical ASAP files into saved feature-analysis tables."""
import csv
import json
from collections import defaultdict
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent))
from _support import REPO_ROOT, DATASETS, source_label
ROOT = REPO_ROOT.parent
AN = REPO_ROOT / "classicfy-ai/analysis"


def main():
    summary = json.loads((OUT / "summary.json").read_text())
    duplicate_keys = {key for group in summary["groups"]["bytes"] for key in group["keys"]}
    names = ["dynamics_pedaling/performance_summary.csv", "articulation/performance_summary.csv",
             "feature_normalization/performance_summary.csv", "_embedding_validation/embeddings.csv",
             "metadata_style_validation/performances.csv", "tempo/performance_summary.csv",
             "rubato/performance_summary.csv"]
    result = []
    for name in names:
        rows = list(csv.DictReader((AN / name).open(encoding="utf-8-sig")))
        by_key = defaultdict(list)
        for line, row in enumerate(rows, 2):
            if row.get("key") in duplicate_keys:
                by_key[row["key"]].append((line, row))
        for key in sorted(duplicate_keys):
            matches = by_key[key]
            row = matches[0][1] if matches else {}
            result.append(dict(table=name, key=key, included=bool(matches),
                               rows=len(matches), lines=";".join(str(line) for line, _ in matches),
                               group=row.get("work", row.get("piece", "")),
                               beat_intervals=row.get("beat_count", row.get("beats", "")),
                               dynamics_mean=row.get("dynamics_mean", ""),
                               pedal_depth_mean=row.get("pedal_depth_mean", ""),
                               articulation_mean=row.get("articulation_mean", ""),
                               flags=row.get("flags", "")))
    with (OUT / "analysis_membership.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
