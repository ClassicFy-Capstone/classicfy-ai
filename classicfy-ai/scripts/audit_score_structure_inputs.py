"""Step 1 inventory: existing loader reuse and score-structure input coverage.

This does not parse a musical timeline, infer phrases or compute tonal tension.
Measure-label coverage is a diagnostic, not a verified MusicXML/beat mapping.
"""

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import fields
import hashlib
import importlib.metadata
import json
from math import isfinite
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'classicfy-ai/src'))
from preprocessing import ASAPLoader, AsapSample, load_match
from features.beat_grid import build_beat_grid


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write(path, values):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(values[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(values)


def xml_inventory(path):
    root = ET.parse(path).getroot()
    # Namespace handling is for inventory only; it does not infer music timing.
    for node in root.iter():
        node.tag = node.tag.rsplit('}', 1)[-1]
    notes = root.findall('.//note')
    measures = root.findall('.//measure')
    ids = [n.get('id') for n in notes if n.get('id')]
    labels = {m.get('number', '') for m in measures}
    counts = Counter(n.tag for n in root.iter())
    slurs = Counter(n.get('type', '') for n in root.findall('.//slur'))
    return dict(root_type=root.tag, parts=counts['part'], measure_elements=len(measures),
        distinct_measure_labels=len(labels), notes=len(notes), rests=counts['rest'],
        grace=counts['grace'], chord_members=counts['chord'],
        notes_with_ids=len(ids), duplicate_note_ids=len(ids)-len(set(ids)),
        notes_with_voice=sum(n.find('voice') is not None for n in notes),
        notes_with_staff=sum(n.find('staff') is not None for n in notes),
        slur_start=slurs['start'], slur_stop=slurs['stop'], slur_continue=slurs['continue'],
        repeat_signs=counts['repeat'], endings=counts['ending'],
        backups=counts['backup'], forwards=counts['forward'],
        key_signatures=counts['key'], keys_with_mode=sum(k.find('mode') is not None for k in root.findall('.//key')),
        fermatas=counts['fermata'], breath_marks=counts['breath-mark'],
        harmony_symbols=counts['harmony']), labels, set(ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asap-root', type=Path, default=ROOT.parent / 'datasets/ASAP')
    parser.add_argument('--nasap-root', type=Path, default=ROOT.parent / 'datasets/nASAP')
    parser.add_argument('--cohort', type=Path, default=ROOT / 'classicfy-ai/analysis/metadata_style_validation/performances.csv')
    parser.add_argument('--out', type=Path, default=ROOT / 'classicfy-ai/analysis/score_structure_inputs')
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    cohort = rows(args.cohort)
    metadata = {r['midi_performance']:r for r in rows(args.asap_root / 'metadata.csv')}
    nasap_metadata = {r['match_file']:r for r in rows(args.nasap_root / 'metadata.csv') if r['match_file']}
    annotations = json.loads((args.asap_root / 'asap_annotations.json').read_text())
    loader = ASAPLoader(args.asap_root, args.nasap_root)
    work_groups = defaultdict(list)
    for row in cohort:
        work_groups[row['piece'], row['grid']].append(row)
    inventories, label_sets, id_sets = {}, {}, {}
    for piece, grid in sorted(work_groups):
        source = metadata[work_groups[piece, grid][0]['key']]
        path = (args.asap_root / source['xml_score']).resolve()
        if not path.is_relative_to(args.asap_root.resolve()) or not path.is_file():
            raise ValueError('Invalid metadata MusicXML path: ' + source['xml_score'])
        inventory, labels, ids = xml_inventory(path)
        inventories[piece, grid] = dict(piece=piece, grid=grid, xml_score=source['xml_score'],
            xml_sha256=sha(path), performances=len(work_groups[piece, grid]), **inventory)
        label_sets[piece, grid] = labels; id_sets[piece, grid] = ids
    records, issues, nasap_inventories = [], [], {}
    for i, row in enumerate(cohort):
        key = row['key']; sample = loader.get_sample(key); source = metadata[key]; annotation = annotations[key]
        if source['midi_score'] != row['piece']:
            raise ValueError('Cohort score identity differs: ' + key)
        score = build_beat_grid(sample.score_beats); performed = build_beat_grid(sample.performance_beats)
        grid = hashlib.sha256(score.edges.tobytes()+json.dumps(sample.score_beat_types).encode()).hexdigest()[:12]
        if grid != row['grid']:
            raise ValueError('Existing feature grid hash differs: ' + key)
        mapping = annotation.get('downbeats_score_map')
        map_status = ('present_list' if isinstance(mapping, list) else
            'missing_nonfinite' if isinstance(mapping, float) and not isfinite(mapping) else
            'missing' if mapping is None else 'invalid_type')
        labels = label_sets[row['piece'], row['grid']]
        map_entries = mapping if isinstance(mapping, list) else []
        components = [str(value).split('-') for value in map_entries]
        # Preserve raw labels: never silently convert 0-based/1-based numbering.
        direct = sum(all(label in labels for label in item) for item in components)
        unmatched = [{'index': j, 'label': str(map_entries[j])}
            for j, item in enumerate(components) if any(label not in labels for label in item)]
        alignment = load_match(sample.note_alignment_path) if sample.note_alignment_path else None
        unique_notes = {n.note_id:n for n in ([s for s,p in alignment.matches]+list(alignment.deletions))} if alignment else {}
        xml_ids = id_sets[row['piece'], row['grid']]
        exact_ids = sum(n in xml_ids for n in unique_notes)
        base_ids = sum(n.rsplit('-',1)[0] in xml_ids for n in unique_notes)
        relative_match = sample.note_alignment_path.relative_to(args.nasap_root.resolve()).as_posix() if sample.note_alignment_path else ''
        nasap_source = nasap_metadata.get(relative_match)
        nasap_xml = nasap_source['xml_score'] if nasap_source else ''
        nasap_ids = set()
        if nasap_xml:
            if nasap_xml not in nasap_inventories:
                nasap_path = (args.nasap_root / nasap_xml).resolve()
                if not nasap_path.is_relative_to(args.nasap_root.resolve()) or not nasap_path.is_file():
                    raise ValueError('Invalid nASAP MusicXML path: ' + nasap_xml)
                inventory, _, ids = xml_inventory(nasap_path)
                nasap_inventories[nasap_xml] = (dict(xml_score=nasap_xml,
                    xml_sha256=sha(nasap_path), **inventory), ids)
            nasap_ids = nasap_inventories[nasap_xml][1]
        canonical_parent = Path(source['midi_performance']).parent.as_posix()
        changed_parent = bool(relative_match and Path(relative_match).parent.as_posix() != canonical_parent)
        record = dict(key=key, piece=row['piece'], grid=grid, aligned=sample.aligned,
            xml_score=source['xml_score'], score_beats=len(score.edges), performance_beats=len(performed.edges),
            beat_intervals=len(score.mask), positive_score_intervals=int(score.mask.sum()),
            positive_performance_intervals=int(performed.mask.sum()),
            irregular_score_beats=sample.score_beat_types.count('bR'),
            irregular_performance_beats=sample.performance_beat_types.count('bR'),
            downbeats=len(sample.score_downbeats), downbeat_map_entries=len(map_entries),
            downbeat_map_status=map_status,
            downbeat_map_count_equal=len(map_entries)==len(sample.score_downbeats),
            combined_measure_map_entries=sum(len(v)>1 for v in components),
            direct_xml_measure_label_entries=direct,
            unmatched_measure_labels=json.dumps(unmatched, separators=(',', ':')),
            only_initial_zero_label_unmatched=unmatched==[{'index': 0, 'label': '0'}],
            note_alignment=relative_match, robust_note_alignment=sample.robust_note_alignment,
            note_alignment_sha256=sha(sample.note_alignment_path) if sample.note_alignment_path else '',
            match_uses_variant_folder=changed_parent, match_score_notes=len(unique_notes),
            exact_match_ids_in_asap_xml=exact_ids, suffix_stripped_match_ids_in_asap_xml=base_ids,
            nasap_xml_score=nasap_xml,
            exact_match_ids_in_nasap_xml=sum(n in nasap_ids for n in unique_notes),
            suffix_stripped_match_ids_in_nasap_xml=sum(n.rsplit('-',1)[0] in nasap_ids for n in unique_notes))
        records.append(record)
        if not record['downbeat_map_count_equal'] or changed_parent or direct != len(map_entries):
            reasons=[]
            if map_status != 'present_list':reasons.append('downbeat_map_'+map_status)
            if not record['downbeat_map_count_equal']:reasons.append('downbeat_map_count_mismatch')
            if changed_parent:reasons.append('nasap_variant_folder')
            if direct != len(map_entries):reasons.append('some_measure_labels_not_exactly_in_xml')
            issues.append(dict(key=key,piece=row['piece'],reasons='|'.join(reasons),
                downbeat_map_status=map_status,unmatched_measure_labels=record['unmatched_measure_labels']))
        if (i+1)%100==0:print('Audited',i+1,'/',len(cohort),flush=True)
    write(args.out/'works.csv',list(inventories.values()));write(args.out/'performances.csv',records)
    if nasap_inventories:write(args.out/'nasap_works.csv',[v[0] for v in nasap_inventories.values()])
    if issues:write(args.out/'review.csv',issues)
    else:(args.out/'review.csv').write_text('key,piece,reasons\n')
    required=('xml_score','score_key_signatures','downbeats_score_map')
    exposed={f.name for f in fields(AsapSample)}
    packages={}
    for name in ('partitura','parangonar','mido','pretty_midi'):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    summary=dict(status='step1_reuse_and_input_inventory_only',performances=len(records),
        work_grids=len(inventories),xml_files=len({r['xml_score'] for r in inventories.values()}),
        xml_with_slurs=sum(r['slur_start']>0 for r in inventories.values()),
        xml_with_repeat_or_endings=sum(r['repeat_signs']>0 or r['endings']>0 for r in inventories.values()),
        xml_with_note_ids=sum(r['notes_with_ids']>0 for r in inventories.values()),
        xml_with_key_mode=sum(r['keys_with_mode']>0 for r in inventories.values()),
        matching_feature_grids=sum(r['grid']==cohort[i]['grid'] for i,r in enumerate(records)),
        aligned=sum(r['aligned'] for r in records),robust_note_alignments=sum(r['robust_note_alignment'] is True for r in records),
        note_alignment_variants=sum(r['match_uses_variant_folder'] for r in records),
        matching_downbeat_map_counts=sum(r['downbeat_map_count_equal'] for r in records),
        downbeat_map_statuses=dict(Counter(r['downbeat_map_status'] for r in records)),
        missing_downbeat_map_work_grids=len({(r['piece'],r['grid']) for r in records if r['downbeat_map_status'] != 'present_list'}),
        initial_zero_label_unmatched_work_grids=len({(r['piece'],r['grid']) for r in records if r['only_initial_zero_label_unmatched']}),
        exact_xml_measure_label_coverage=sum(r['direct_xml_measure_label_entries'] for r in records)/sum(r['downbeat_map_entries'] for r in records),
        nasap_xml_files=len(nasap_inventories),
        nasap_xml_with_note_ids=sum(v[0]['notes_with_ids']>0 for v in nasap_inventories.values()),
        nasap_suffix_stripped_note_id_coverage=sum(r['suffix_stripped_match_ids_in_nasap_xml'] for r in records)/sum(r['match_score_notes'] for r in records),
        score_grids_with_irregular_beats=len({(r['piece'],r['grid']) for r in records if r['irregular_score_beats']}),
        current_sample_fields=sorted(exposed),not_exposed_in_sample=[f for f in required if f not in exposed],
        packages=packages,
        limitations=['No MusicXML onset timeline or repeat unfolding implemented here',
            'Matching measure labels alone is not proof of timing alignment',
            'nASAP match note IDs cannot be assumed identical to original ASAP MusicXML note IDs',
            'Neither phrase labels nor tonal tension are generated'])
    signature=dict(cohort=sha(args.cohort),metadata=sha(args.asap_root/'metadata.csv'),
        annotations=sha(args.asap_root/'asap_annotations.json'),nasap_metadata=sha(args.nasap_root/'metadata.csv'),
        code={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'classicfy-ai/src/preprocessing/asap_loader.py',
            ROOT/'classicfy-ai/src/preprocessing/note_alignment.py',ROOT/'classicfy-ai/src/features/beat_grid.py']},
        csv_sha256={p.name:sha(p) for p in args.out.glob('*.csv')})
    (args.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (args.out/'provenance.json').write_text(json.dumps(signature,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
