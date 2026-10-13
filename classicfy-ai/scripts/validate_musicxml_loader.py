"""Validate the MusicXML loader against the frozen #16 input inventory."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'classicfy-ai/src'))
from preprocessing import load_musicxml


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def check_raw_fields(path, score):
    """Compare retained fields to source XML separately from timing fixtures."""
    root = ET.parse(path).getroot()
    for node in root.iter():
        node.tag = node.tag.rsplit('}', 1)[-1]
    raw_notes = root.findall('part/measure/note')
    counts = dict(notes=len(raw_notes), measures=len(root.findall('part/measure')),
        rests=len(root.findall('part/measure/note/rest')),
        grace=len(root.findall('part/measure/note/grace')),
        chord_members=len(root.findall('part/measure/note/chord')),
        slurs=len(root.findall('part/measure/note/notations/slur')),
        keys=len(root.findall('part/measure/attributes/key')),
        meters=len(root.findall('part/measure/attributes/time')),
        repeats=len(root.findall('part/measure/barline/repeat')),
        endings=len(root.findall('part/measure/barline/ending')))
    loaded = dict(notes=len(score.notes), measures=len(score.measures),
        rests=sum(n.is_rest for n in score.notes), grace=sum(n.is_grace for n in score.notes),
        chord_members=sum(n.is_chord for n in score.notes), slurs=len(score.slurs),
        keys=len(score.key_signatures), meters=len(score.time_signatures),
        repeats=sum(b.kind=='repeat' for b in score.barline_marks),
        endings=sum(b.kind=='ending' for b in score.barline_marks))
    if counts != loaded:
        raise ValueError(f'Source/loaded element counts differ: {counts} != {loaded}')
    for raw, note in zip(raw_notes, score.notes):
        if raw.get('id') != note.xml_id or raw.findtext('voice') != note.voice:
            raise ValueError(f'Note ID or voice changed: {note.note_id}')
        raw_slurs = raw.findall('notations/slur')
        if [tuple(sorted(s.attrib.items())) for s in raw_slurs] != [s.attributes for s in note.slurs]:
            raise ValueError(f'Slur attributes changed: {note.note_id}')
    if len({n.note_id for n in score.notes}) != len(score.notes):
        raise ValueError('Internal note IDs are not unique')
    if any(n.measure_offset_quarters < 0 or n.duration_quarters < 0 for n in score.notes):
        raise ValueError('Negative note position or duration')
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asap-root', type=Path, default=ROOT.parent/'datasets/ASAP')
    parser.add_argument('--nasap-root', type=Path, default=ROOT.parent/'datasets/nASAP')
    parser.add_argument('--inventory', type=Path, default=ROOT/'classicfy-ai/analysis/score_structure_inputs')
    parser.add_argument('--out', type=Path, default=ROOT/'classicfy-ai/analysis/musicxml_loader')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records, diagnostics = [], []
    for dataset, filename, directory in [('ASAP', 'works.csv', args.asap_root),
                                         ('nASAP', 'nasap_works.csv', args.nasap_root)]:
        with (args.inventory/filename).open(encoding='utf-8-sig', newline='') as stream:
            inventory = list(csv.DictReader(stream))
        for row in inventory:
            path = (directory/row['xml_score']).resolve()
            record = dict(dataset=dataset, xml_score=row['xml_score'], xml_sha256='',
                status='error', error='', notes=0, measures=0, rests=0, grace=0,
                chord_members=0, slurs=0, keys=0, meters=0, repeats=0, endings=0,
                notes_with_xml_ids=0, diagnostics=0)
            try:
                if not path.is_relative_to(directory.resolve()):
                    raise ValueError('XML path is outside the dataset root')
                record['xml_sha256'] = sha(path)
                if record['xml_sha256'] != row['xml_sha256']:
                    raise ValueError('XML differs from the frozen input inventory')
                score = load_musicxml(path)
                record.update(check_raw_fields(path, score))
                record.update(status='ok', notes_with_xml_ids=sum(n.xml_id is not None for n in score.notes),
                              diagnostics=len(score.diagnostics))
                diagnostics.extend(dict(dataset=dataset, xml_score=row['xml_score'],
                    code=d.code, part_id=d.part_id, measure_index=d.measure_index,
                    measure_number=next((m.number for m in score.measures
                        if m.part_id==d.part_id and m.measure_index==d.measure_index), ''),
                    detail=d.detail) for d in score.diagnostics)
            except (ValueError, OSError, ET.ParseError) as exc:
                record['error'] = f'{type(exc).__name__}: {exc}'
            records.append(record)
        print(f'{dataset}: {sum(r["status"]=="ok" for r in records if r["dataset"]==dataset)}/{len(inventory)}', flush=True)
    if not records:
        raise ValueError('Empty score inventory')
    write_csv(args.out/'scores.csv', records, list(records[0]))
    write_csv(args.out/'diagnostics.csv', diagnostics,
        ['dataset', 'xml_score', 'code', 'part_id', 'measure_index', 'measure_number', 'detail'])
    summary = dict(files=len(records), loaded=sum(r['status']=='ok' for r in records),
        errors=sum(r['status']!='ok' for r in records),
        datasets={ds:dict(files=sum(r['dataset']==ds for r in records),
            loaded=sum(r['dataset']==ds and r['status']=='ok' for r in records),
            counts={name:sum(r[name] for r in records if r['dataset']==ds)
                for name in ('notes','measures','rests','grace','chord_members','slurs','keys','meters','repeats','endings')},
            diagnostics=dict(Counter(d['code'] for d in diagnostics if d['dataset']==ds)))
            for ds in ('ASAP', 'nASAP')},
        scope='Written MusicXML loading and field preservation, not beat alignment or phrase/tension inference')
    (args.out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    provenance = dict(inventory={name:sha(args.inventory/name) for name in ('works.csv','nasap_works.csv')},
        code={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),
            ROOT/'classicfy-ai/src/preprocessing/musicxml_loader.py',
            ROOT/'classicfy-ai/src/preprocessing/__init__.py')},
        results={name:sha(args.out/name) for name in ('scores.csv','diagnostics.csv','summary.json')})
    (args.out/'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if summary['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
