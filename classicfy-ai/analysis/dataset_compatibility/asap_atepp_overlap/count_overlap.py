"""Count repertoire overlap from complete local metadata, preserving source units.

Run from any directory with Python 3. No input data or repository files change.
The checked mapping CSV records each decision and its source track labels.
"""
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent))
from _support import REPO_ROOT, DATASETS, source_label
ROOT = REPO_ROOT.parent
ASAP = DATASETS / "ASAP/metadata.csv"
ATEPP = DATASETS / "ATEPP/ATEPP-metadata-1.2.csv"


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write(name, rows):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def norm(text):
    return "".join(c for c in unicodedata.normalize("NFKD", text.casefold())
                   if not unicodedata.combining(c))


def main():
    asap, atepp = read(ASAP), read(ATEPP)
    aw, tw = defaultdict(list), defaultdict(list)
    for row in asap:
        aw[row["composer"], row["title"]].append(row)
    for row in atepp:
        tw[row["composition_id"]].append(row)
    assert len({r["midi_performance"] for r in asap}) == len(asap)
    assert len({r["perf_id"] for r in atepp}) == len(atepp)

    # Explicit metadata-ID crosswalk. IDs are source IDs, not inferred recording IDs.
    fixed = {
        "Chopin": {
            "Ballades_1": [1611], "Ballades_2": [1668], "Ballades_3": [1637],
            "Ballades_4": [1658], "Barcarolle": [1630, 1652], "Berceuse_op_57": [1672],
            "Polonaises_53": [1617, 1680], "Scherzos_20": [1657],
            "Scherzos_31": [1659], "Scherzos_39": [1677],
            "Sonata_2_1st": [1452], "Sonata_2_2nd": [1454],
            "Sonata_2_3rd": [1453], "Sonata_2_4th": [1451],
            "Sonata_3_2nd": [1461], "Sonata_3_3rd": [1458], "Sonata_3_4th": [1459],
        },
        "Haydn": {
            "Keyboard_Sonatas_32-1": [1076], "Keyboard_Sonatas_46-1": [1064],
            "Keyboard_Sonatas_48-1": [1067], "Keyboard_Sonatas_48-2": [1068],
            "Keyboard_Sonatas_50-1": [1072],
        },
        "Liszt": {
            "Annees_de_pelerinage_2_1_Gondoliera": [105],
            "Concert_Etude_S145_2": [116], "Mephisto_Waltz": [1503], "Sonata": [1495],
            "Transcendental_Etudes_1": [119], "Transcendental_Etudes_3": [124],
            "Transcendental_Etudes_5": [123], "Transcendental_Etudes_10": [121],
            "Transcendental_Etudes_11": [117],
        },
        "Mozart": {
            "Fantasie_475": [1565], "Piano_Sonatas_11-3": [1148],
            "Piano_Sonatas_12-1": [1118], "Piano_Sonatas_12-2": [1117],
            "Piano_Sonatas_12-3": [1116], "Piano_Sonatas_8-1": [1154],
        },
        "Rachmaninoff": {
            "Preludes_op_23_4": [80], "Preludes_op_23_6": [77],
            "Preludes_op_32_10": [9], "Preludes_op_32_5": [10],
        },
        "Ravel": {
            "Gaspard_de_la_Nuit_1_Ondine": [721], "Miroirs_3_Une_Barque": [713],
            "Miroirs_4_Alborada_del_gracioso": [716], "Pavane": [1543],
        },
        "Schubert": {
            "Impromptu_op.90_D.899_1": [910], "Impromptu_op.90_D.899_2": [911],
            "Impromptu_op.90_D.899_3": [913], "Impromptu_op.90_D.899_4": [912],
            "Impromptu_op142_1": [918], "Impromptu_op142_3": [917],
            "Moment_Musical_no_1": [865], "Moment_musical_no_3": [861],
            "Piano_Sonatas_664-1": [870], "Piano_Sonatas_664-2": [872],
            "Piano_Sonatas_664-3": [871], "Piano_Sonatas_894-2": [885],
        },
        "Schumann": {
            "Arabeske": [1517], "Toccata": [1518], "Toccata_repeat": [1518],
            "Kreisleriana_1": [220], "Kreisleriana_2": [215], "Kreisleriana_3": [218],
            "Kreisleriana_4": [217], "Kreisleriana_5": [219], "Kreisleriana_6": [214],
            "Kreisleriana_7": [221],
        },
        "Scriabin": {"Etudes_op_8_11": [127], "Sonatas_5": [1512]},
        "Debussy": {"Images_Book_1_1_Reflets_dans_lEau": [646]},
    }
    etudes = {
        10: {1: 1412, 2: 1413, 3: 1416, 4: 1420, 5: 1410, 7: 1415, 8: 1414,
             10: 1417, 12: 1411},
        25: {1: 1429, 2: 1426, 4: 1427, 5: 1430, 8: 1425, 10: 1421, 11: 1431, 12: 1428},
    }
    for op, numbers in etudes.items():
        for number, cid in numbers.items():
            fixed["Chopin"][f"Etudes_op_{op}_{number}"] = [cid]

    # Beethoven's conventional sonata number and movement; all decisions explicit,
    # including titles whose movement numbers are absent or incorrect in metadata.
    beethoven = {
        "1-1": [1183], "2-1": [1209], "3-1": [1177], "3-2": [1179],
        "4-1": [1202], "5-1": [1309], "7-1": [1168], "7-2": [1167],
        "7-3": [1165], "7-4": [1166], "8-1": [1241], "8-2": [1242], "8-3": [1243],
        "9-1": [1186], "9-2": [1185], "9-3": [1187],
        "10-1": [1225], "11-1": [1300], "11-2": [1301], "11-3": [1303],
        "12-1": [1247], "13-4": [1258], "14-3": [1304],
        "15-1": [1251], "15-4": [1248], "16-1": [1315], "16-2": [1316],
        "17-1": [1253], "17-2": [1252], "17-3": [1254],
        "18-1": [1227], "18-2": [1229], "18-3": [1230], "18-4": [1228],
        "21-1": [1213], "21-2": [1215], "21-3": [1214],
        "22-1": [1235], "22-2": [1236], "23-1": [1223],
        "24-1": [1263], "24-2": [1264],
        "26-1": [1197], "26-2": [1196], "26-3": [1198],
        "27-1": [1260], "27-2": [1259], "28-1": [1240], "28-2": [1238],
        "29-2": [1233], "29-3": [1231], "29-4": [1234],
        "30-1": [1206], "31-1": [1190], "31-2": [1194], "32-1": [1308],
    }
    fixed["Beethoven"] = {"Piano_Sonatas_" + k: v for k, v in beethoven.items()}

    # Scope differs: ASAP combines material that ATEPP separates into movements.
    segmented = {
        ("Beethoven", "Piano_Sonatas_31-3_4"): [1195, 1193],
        ("Schubert", "Wanderer_fantasie"): [896, 894, 897, 895],
    }
    result = []
    for (composer, title), rows in sorted(aw.items()):
        note = "checked composer, catalogue/sonata number, and movement/individual piece"
        ids = list(map(str, fixed.get(composer, {}).get(title, [])))
        if composer == "Bach" and "_bwv_" in title:
            kind, bwv = title.split("_bwv_")
            for cid, tracks in tw.items():
                if tracks[0]["composer"] != "Johann Sebastian Bach":
                    continue
                # Last BWV occurrence avoids matching book-wide ranges (870–893).
                for track in tracks:
                    label = norm(track["track"])
                    numbers = re.findall(r"bwv\s*\.?\s*(\d+)", label)
                    if numbers and numbers[-1] == bwv and kind.casefold() in label:
                        # All matching metadata rows must describe the individual type.
                        if "prelude and fugue" not in label and "prelude & fugue" not in label:
                            ids.append(cid)
                            break
            ids = sorted(set(ids), key=int)
            note = "BWV number + Prelude/Fugue; book-range numbers ignored"
        status = "same_unit" if ids else "not_matched"
        if (composer, title) in segmented:
            ids = list(map(str, segmented[composer, title]))
            status = "split_across_atepp_units"
            note = "same musical material, but ASAP combines ATEPP movement units"
        selected = [row for cid in ids for row in tw[cid]]
        assert all(cid in tw for cid in ids)
        result.append(dict(
            composer=composer, asap_title=title, status=status,
            asap_performances=len({row["midi_performance"] for row in rows}),
            atepp_composition_ids=";".join(ids), atepp_id_count=len(ids),
            atepp_performances=len({row["perf_id"] for row in selected}),
            atepp_example_tracks=" || ".join(min((r["track"] for r in tw[cid]), key=len)
                                             for cid in ids),
            asap_score_paths=";".join(sorted({row["xml_score"] for row in rows})), note=note,
        ))

    write("work_crosswalk.csv", result)
    write("unmatched_asap_works.csv", [r for r in result if r["status"] == "not_matched"])
    allids, strictids = set(), set()
    for row in result:
        ids = set(filter(None, row["atepp_composition_ids"].split(";")))
        allids.update(ids)
        if row["status"] == "same_unit":
            strictids.update(ids)
    catalog = [dict(composition_id=cid, composer=tw[cid][0]["composer"],
                    example_track=min((r["track"] for r in tw[cid]), key=len),
                    performances=len(tw[cid]), same_unit_match=cid in strictids,
                    asap_titles=";".join(r["asap_title"] for r in result
                                         if cid in r["atepp_composition_ids"].split(";")))
               for cid in sorted(allids, key=int)]
    write("matched_atepp_compositions.csv", catalog)
    performances = [dict(**r, overlap_scope="same_unit" if r["composition_id"] in strictids
                        else "split_across_atepp_units")
                    for r in atepp if r["composition_id"] in allids]
    write("matched_atepp_performances.csv", performances)
    asap_status = {(r["composer"], r["asap_title"]): r["status"] for r in result}
    write("matched_asap_performances.csv", [dict(**r, overlap_scope=asap_status[r["composer"], r["title"]])
          for r in asap if asap_status[r["composer"], r["title"]] != "not_matched"])

    summaries = []
    for composer in sorted({key[0] for key in aw}):
        selected = [r for r in result if r["composer"] == composer]
        cids = {cid for r in selected for cid in r["atepp_composition_ids"].split(";") if cid}
        summaries.append(dict(
            composer=composer, asap_total_works=len(selected),
            asap_same_unit_works=sum(r["status"] == "same_unit" for r in selected),
            asap_split_works=sum(r["status"] == "split_across_atepp_units" for r in selected),
            asap_total_performances=sum(r["asap_performances"] for r in selected),
            asap_performances_on_overlapping_works=sum(r["asap_performances"] for r in selected
                                                       if r["status"] != "not_matched"),
            atepp_matching_ids=len(cids),
            atepp_performances_on_overlapping_works=sum(len(tw[cid]) for cid in cids),
        ))
    write("composer_summary.csv", summaries)
    # A conservative candidate search, not an assertion of recording identity.
    # ASAP has surname/initial file labels rather than a full artist-name column.
    identity_audit = []
    row_lookup = {(r["composer"], r["asap_title"]): r for r in result}
    for artist in sorted({r["artist"] for r in atepp}):
        surname = re.sub(r"[^a-z]", "", norm(artist.split()[-1]))
        candidates = [r for r in asap
                      if re.sub(r"[^a-z]", "", norm(Path(r["midi_performance"]).stem)).startswith(surname)]
        same_work_pairs = []
        for candidate in candidates:
            ids = row_lookup[candidate["composer"], candidate["title"]]["atepp_composition_ids"].split(";")
            same_work_pairs.extend((candidate["midi_performance"], r["perf_id"])
                                   for r in atepp if r["artist"] == artist and r["composition_id"] in ids)
        identity_audit.append(dict(
            atepp_artist=artist, asap_surname_prefix_candidates=len(candidates),
            same_work_candidate_pairs=len(same_work_pairs),
            asap_candidate_paths=";".join(r["midi_performance"] for r in candidates),
            limitation="surname-prefix heuristic only; not audio/MIDI recording verification",
        ))
    write("performance_identity_audit.csv", identity_audit)
    summary = dict(
        scope="Complete local ASAP metadata and complete official ATEPP 1.2 metadata; no feature/quality filtering",
        asap_total_work_keys=len(aw), asap_total_performances=len(asap),
        atepp_total_composition_ids=len(tw), atepp_total_performances=len(atepp),
        asap_same_unit_work_keys=sum(r["status"] == "same_unit" for r in result),
        asap_same_unit_performances=sum(r["asap_performances"] for r in result if r["status"] == "same_unit"),
        atepp_same_unit_composition_ids=len(strictids),
        atepp_same_unit_performances=sum(len(tw[cid]) for cid in strictids),
        same_unit_unique_musical_works=len({
            (r["composer"], "Toccata" if r["composer"] == "Schumann"
             and r["asap_title"] == "Toccata_repeat" else r["asap_title"])
            for r in result if r["status"] == "same_unit"
        }),
        asap_split_work_keys=sum(r["status"] == "split_across_atepp_units" for r in result),
        asap_overlapping_work_keys=sum(r["status"] != "not_matched" for r in result),
        asap_performances_on_overlapping_works=sum(r["asap_performances"] for r in result if r["status"] != "not_matched"),
        atepp_overlapping_composition_ids=len(allids),
        atepp_performances_on_overlapping_works=sum(len(tw[cid]) for cid in allids),
        source_work_aliases={"ASAP": "Schumann Toccata and Toccata_repeat refer to Op.7; source keys retained",
                             "ATEPP": "Chopin Barcarolle Op.60 and Polonaise Op.53 each have two matching IDs; source IDs retained"},
        recording_overlap={"confirmed_identical_recordings": 0,
                           "artist_surname_and_same_work_candidate_pairs": sum(r["same_work_candidate_pairs"] for r in identity_audit),
                           "exhaustive_recording_deduplication": False,
                           "reason": "No shared recording identifier; ASAP has no full artist-name field and local ATEPP contains metadata only. Zero confirmed matches does not establish zero true overlap."},
        sources=[dict(path=source_label(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                 for p in [ASAP, ATEPP]],
    )
    for prefix in ["asap", "atepp"]:
        works = summary[f"{prefix}_overlapping_work_keys" if prefix == "asap" else "atepp_overlapping_composition_ids"]
        denom = summary[f"{prefix}_total_work_keys" if prefix == "asap" else "atepp_total_composition_ids"]
        summary[f"{prefix}_overlapping_work_percent"] = 100 * works / denom
        summary[f"{prefix}_overlapping_repertoire_performance_percent"] = 100 * summary[f"{prefix}_performances_on_overlapping_works"] / summary[f"{prefix}_total_performances"]
        nworks = summary[f"{prefix}_same_unit_work_keys" if prefix == "asap" else "atepp_same_unit_composition_ids"]
        summary[f"{prefix}_same_unit_work_percent"] = 100 * nworks / denom
        summary[f"{prefix}_same_unit_performance_percent"] = 100 * summary[f"{prefix}_same_unit_performances"] / summary[f"{prefix}_total_performances"]
    # One ASAP alias pair and two ATEPP duplicate-ID pairs explain the difference.
    assert summary["asap_same_unit_work_keys"] - 1 == summary["atepp_same_unit_composition_ids"] - 2 == summary["same_unit_unique_musical_works"]
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nUNMATCHED")
    for r in result:
        if r["status"] == "not_matched":
            print(r["composer"], r["asap_title"])


if __name__ == "__main__":
    main()
