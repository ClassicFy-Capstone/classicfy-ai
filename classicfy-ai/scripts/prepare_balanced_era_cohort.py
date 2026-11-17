"""Select 100 work/movement IDs per era from all current ATEPP_refine recordings.

Stratifies by composer work counts, with at least one work per composer, and
retains every metadata recording of each selected composition_id.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
DATASET = PROJECT.parent / 'datasets/ATEPP_refine'
OUTPUT = PROJECT / 'classicfy-ai/analysis/era_refine_balanced_100'
SEED = 20261106
TARGET = 100
ERAS = {
    'classical_baroque': ('고전·바로크', {
        'Johann Sebastian Bach', 'George Frideric Handel', 'Domenico Scarlatti',
        'Franz Joseph Haydn', 'Wolfgang Amadeus Mozart', 'Ludwig van Beethoven'}),
    'romantic': ('낭만', {
        'Frédéric Chopin', 'Robert Schumann', 'Franz Schubert', 'Sergei Rachmaninoff',
        'Franz Liszt', 'Johannes Brahms', 'Pyotr Ilyich Tchaikovsky', 'Modest Mussorgsky',
        'Edvard Grieg', 'Felix Mendelssohn', 'César Franck', 'Alexander Scriabin'}),
    'impressionist_modern': ('인상주의·현대', {
        'Claude Debussy', 'Maurice Ravel', 'Sergei Prokofiev', 'Dmitri Shostakovich',
        'Béla Bartók', 'Anton Webern', 'Paul Hindemith'}),
}
BOUNDARIES = {'Ludwig van Beethoven', 'Sergei Rachmaninoff', 'Alexander Scriabin'}


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def allocate(counts):
    """Proportional integer quotas; enforce one per composer, break ties by name."""
    total = sum(counts.values())
    assert len(counts) <= TARGET <= total
    exact = {c: Fraction(TARGET * n, total) for c, n in counts.items()}
    quotas = {c: max(1, q.numerator // q.denominator) for c, q in exact.items()}
    assert sum(quotas.values()) <= TARGET
    while sum(quotas.values()) < TARGET:
        composer = min((c for c in counts if quotas[c] < counts[c]),
                       key=lambda c: (-(exact[c] - quotas[c]), c))
        quotas[composer] += 1
    return quotas


def prepare(metadata, inventory_path, out):
    rows = read_csv(metadata)
    inventory_rows = read_csv(inventory_path)
    manifest_path = metadata.parent / 'refinement_manifest.json'
    receipt = json.loads(manifest_path.read_text())
    assert receipt['dataset'] == 'ATEPP_refine'
    assert sha256(metadata) == receipt['refined_metadata_sha256'], 'Refine metadata checksum differs'
    assert len(rows) == len({r['perf_id'] for r in rows}) == receipt['retained_performance_count']
    original_path = PROJECT.parent / receipt['source_directory'] / metadata.name
    assert sha256(original_path) == receipt['source_metadata_sha256']
    original_rows = read_csv(original_path)
    original = {r['perf_id']: r for r in original_rows}
    assert len(original_rows) == len(original) == receipt['source_performance_count'] == 11674
    assert all(r == original[r['perf_id']] for r in rows), 'Refine rows differ from original metadata'
    excluded_path = metadata.parent / receipt['exclusion_ids_file']
    assert sha256(excluded_path) == receipt['exclusion_ids_sha256']
    excluded_ids = set(excluded_path.read_text().splitlines())
    assert {r['perf_id'] for r in rows} == set(original) - excluded_ids
    checksums_path = metadata.parent / 'file_checksums.csv'
    checksums = {r['refined_file']: r for r in read_csv(checksums_path)}
    input_paths = [metadata, inventory_path, manifest_path, excluded_path, checksums_path]
    input_hashes = {str(p.resolve()): sha256(p) for p in input_paths}
    inventory = {r['perf_id']: r for r in inventory_rows}
    assert len(inventory_rows) == len(inventory) == len(rows)
    assert set(inventory) == {r['perf_id'] for r in rows}
    assert {p.stem for p in (metadata.parent / 'ATEPP-1.2').rglob('*.mid')} == set(inventory)
    composer_era = {c: (era, label) for era, (label, names) in ERAS.items() for c in names}
    assert len(composer_era) == sum(len(names) for _, names in ERAS.values()) == 25
    assert set(composer_era) == {r['composer'] for r in rows}

    by_work = defaultdict(list)
    for row in rows:
        by_work[row['composition_id']].append(row)
    assert '' not in by_work
    population, by_composer = [], defaultdict(list)
    for composition_id, members in by_work.items():
        assert len({r['composer'] for r in members}) == 1, composition_id
        composer = members[0]['composer']
        era, label = composer_era[composer]
        titles = Counter(r['track'] for r in members)
        representative = min(titles, key=lambda title: (-titles[title], title))
        token = f'{SEED}|{era}|{composer}|{composition_id}'
        work = {
            'composition_id': composition_id, 'era': era, 'era_ko': label,
            'composer': composer, 'representative_track': representative,
            'track_title_variants': len(titles), 'recordings': len(members),
            'artists': len({r['artist_id'] for r in members}),
            'recordings_with_score': sum(bool(r['score_path']) for r in members),
            'has_any_score': any(r['score_path'] for r in members),
            'classification_basis': 'composer_style_proxy',
            'composition_date_verified': False, 'boundary_composer': composer in BOUNDARIES,
            'sampling_sha256': hashlib.sha256(token.encode('utf-8')).hexdigest(),
            'composer_quota': 0, 'selected': False,
        }
        population.append(work)
        by_composer[composer].append(work)

    composer_summary = []
    for era, (label, composers) in ERAS.items():
        counts = {c: len(by_composer[c]) for c in composers}
        quotas = allocate(counts)
        for composer in sorted(composers):
            members = sorted(by_composer[composer], key=lambda w: w['sampling_sha256'])
            for i, work in enumerate(members):
                work['composer_quota'] = quotas[composer]
                work['selected'] = i < quotas[composer]
            composer_summary.append({
                'era': era, 'era_ko': label, 'composer': composer,
                'population_works': len(members), 'selected_works': quotas[composer],
                'population_recordings': sum(w['recordings'] for w in members),
                'selected_recordings': sum(w['recordings'] for w in members if w['selected']),
            })
    order = {era: i for i, era in enumerate(ERAS)}
    population.sort(key=lambda w: (order[w['era']], w['composer'], int(w['composition_id'])))
    selected = [w for w in population if w['selected']]
    selected_ids = {w['composition_id'] for w in selected}
    assert len(selected) == len(selected_ids) == 300
    recordings = []
    for row in rows:
        if row['composition_id'] not in selected_ids:
            continue
        local = inventory[row['perf_id']]
        midi_file = metadata.parent / local['midi_file']
        assert local['midi_file'] == 'ATEPP-1.2/' + row['midi_path']
        assert midi_file.is_file(), row['perf_id']
        assert sha256(midi_file) == checksums[local['midi_file']]['sha256'], row['perf_id']
        with midi_file.open('rb') as stream:
            assert stream.read(4) == b'MThd', row['perf_id']
        if local['score_file']:
            assert (metadata.parent / local['score_file']).is_file(), row['perf_id']
        era, label = composer_era[row['composer']]
        recordings.append(dict(row, era=era, era_ko=label,
                               local_midi_file=local['midi_file'],
                               local_score_file=local['score_file']))
    recordings.sort(key=lambda r: (order[r['era']], int(r['composition_id']), r['perf_id']))

    summary = []
    for era, (label, composers) in ERAS.items():
        members = [w for w in population if w['era'] == era]
        chosen = [w for w in members if w['selected']]
        assert len(chosen) == TARGET
        assert {w['composer'] for w in chosen} == composers
        summary.append({
            'era': era, 'era_ko': label, 'population_composers': len(composers),
            'population_works': len(members),
            'population_recordings': sum(w['recordings'] for w in members),
            'selected_composers': len({w['composer'] for w in chosen}),
            'selected_works': len(chosen),
            'selected_recordings': sum(w['recordings'] for w in chosen),
            'selected_works_with_any_score': sum(w['has_any_score'] for w in chosen),
            'selected_recordings_with_score': sum(w['recordings_with_score'] for w in chosen),
        })
    assert sum(w['recordings'] for w in selected) == len(recordings)
    assert all(sha256(Path(path)) == digest for path, digest in input_hashes.items()), 'Input changed during selection'
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'population_works.csv', population)
    write_csv(out / 'selected_works.csv', selected)
    write_csv(out / 'selected_recordings.csv', recordings)
    write_csv(out / 'composer_summary.csv', composer_summary)
    write_csv(out / 'era_summary.csv', summary)
    for era in ERAS:
        write_csv(out / f'{era}_100.csv', [w for w in selected if w['era'] == era])
    protocol = {
        'schema': 'classicfy-refine-atepp-balanced-era-v1', 'dataset': 'ATEPP_refine',
        'population': {'recordings': len(rows), 'works_movements': len(by_work), 'composers': 25},
        'selection_unit': 'Original composition_id; individual movements/pieces, not necessarily complete multi-movement works.',
        'seed': SEED, 'works_per_era': TARGET,
        'classification': {era: {'label': label, 'composers': sorted(composers)}
                           for era, (label, composers) in ERAS.items()},
        'boundary_policy': 'Beethoven in classical_baroque; Rachmaninoff and all Scriabin in romantic, following the preceding whole-dataset count; no work-specific period verification.',
        'allocation': 'Composer work-count proportional integer quotas: floor exact shares, minimum one; allocate remaining seats to greatest exact_share-minus-current_quota, ties by composer name.',
        'sampling': 'Ascending SHA256 of UTF-8 seed|era|composer|composition_id within composer, take first quota works without replacement.',
        'recording_policy': 'Retain all current refine metadata rows for selected composition_ids; never restore excluded original recordings; no further recording downsampling or deduplication.',
        'upstream_refine_filter': receipt['filter'],
        'filters': 'No additional score availability, quality label, alignment, existing feature availability, feature value, or minimum recording-count filter.',
        'title_policy': 'Most frequent original track title per composition_id, lexicographic tie-break; all original recording titles retained.',
        'feature_status': 'Selection only; raw extraction, normalization and era feature comparisons not performed.',
        'analysis_weighting': 'Era work counts are equal; recording counts are not. Summarize per composition_id before comparing eras.',
        'source_metadata': str(metadata.resolve()), 'source_metadata_sha256': sha256(metadata),
        'source_inventory': str(inventory_path.resolve()), 'source_inventory_sha256': sha256(inventory_path),
        'input_sha256': input_hashes,
        'original_metadata_sha256': sha256(original_path),
        'script_sha256': sha256(Path(__file__).resolve()),
        'selected_recording_files_checked': len(recordings), 'summary': summary,
    }
    (out / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, default=DATASET / 'ATEPP-metadata-1.2.csv')
    parser.add_argument('--inventory', type=Path, default=DATASET / 'local_inventory.csv')
    parser.add_argument('--out', type=Path, default=OUTPUT)
    args = parser.parse_args()
    prepare(args.metadata, args.inventory, args.out)
