"""Freeze an era observation cohort from the existing ATEPP diversity exports.

Uses metadata only; does not extract features, refit normalization, or train models.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path, PureWindowsPath


ANALYSIS = Path(__file__).resolve().parents[1] / 'analysis'
ERA_ORDER = ['early', 'classical_romantic', 'modern_contemporary']
ERA_NAMES = {'early': '고음악', 'classical_romantic': '고전·낭만',
             'modern_contemporary': '근현대'}
# Operational style groups, not verified composition-date labels.
COMPOSERS = {
    'Johann Sebastian Bach': ('early', 'baroque', ''),
    'Franz Joseph Haydn': ('classical_romantic', 'classical', ''),
    'Wolfgang Amadeus Mozart': ('classical_romantic', 'classical', ''),
    'Ludwig van Beethoven': ('classical_romantic', 'classical_romantic_transition', 'beethoven'),
    'Franz Schubert': ('classical_romantic', 'romantic', ''),
    'Robert Schumann': ('classical_romantic', 'romantic', ''),
    'Franz Liszt': ('classical_romantic', 'romantic', ''),
    'Sergei Rachmaninoff': ('classical_romantic', 'late_romantic', 'rachmaninoff'),
    'Claude Debussy': ('modern_contemporary', 'french_turn_of_century', ''),
    'Maurice Ravel': ('modern_contemporary', 'french_turn_of_century', ''),
}


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_recorded_hash(path, expected):
    data = path.read_bytes()
    lf = data.replace(b'\r\n', b'\n')
    variants = {'exact': data, 'lf': lf, 'crlf': lf.replace(b'\n', b'\r\n')}
    for mode, content in variants.items():
        if hashlib.sha256(content).hexdigest() == expected:
            return {'path': str(path.relative_to(ANALYSIS)), 'match': mode,
                    'recorded_sha256': expected, 'actual_sha256': sha256(path)}
    raise ValueError(f'{path}: content differs beyond line endings')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def prepare(out):
    diversity = ANALYSIS / 'atepp_style_diversity'
    normalization = ANALYSIS / 'atepp_normalization'
    sources = {
        'diversity': diversity / 'within_work_diversity.csv',
        'queries': diversity / 'cross_composer_queries.csv',
        'catalog': normalization / 'performance_catalog.csv',
        'embeddings': normalization / '_embedding_validation/embeddings.csv',
        'audit': normalization / 'cohort_audit.csv',
        'stats': diversity / 'stats.json',
        'artifact_manifest': diversity / 'artifact_manifest.csv',
        'normalization_stats': normalization / 'stats.json',
    }
    stats = json.loads(sources['stats'].read_text(encoding='utf-8'))
    verified_inputs = []
    for row in read_csv(sources['artifact_manifest']):
        if row['path'] in {'within_work_diversity.csv', 'cross_composer_queries.csv', 'stats.json'}:
            verified_inputs.append(verify_recorded_hash(diversity / row['path'], row['sha256']))
    expected = {PureWindowsPath(name).name: digest
                for name, digest in stats['input_sha256'].items()}
    for name in ['catalog', 'embeddings', 'normalization_stats']:
        path = sources[name]
        verified_inputs.append(verify_recorded_hash(path, expected[path.name]))

    feature_rows = read_csv(sources['diversity'])
    by_work = defaultdict(list)
    for row in feature_rows:
        by_work[row['piece']].append(row)
    features = {'Tempo', 'Rubato', 'Dynamics', 'Articulation', 'Pedaling'}
    for piece, rows in by_work.items():
        assert len(rows) == 5 and {r['feature'] for r in rows} == features, piece
        assert len({(r['composer'], r['track'], r['performances'], r['shared_beats'])
                    for r in rows}) == 1, piece

    queries = read_csv(sources['queries'])
    embeddings = read_csv(sources['embeddings'])
    catalog_rows = read_csv(sources['catalog'])
    catalog = {r['key']: r for r in catalog_rows}
    assert len(catalog) == len(catalog_rows)
    embedded = {r['key']: r for r in embeddings}
    assert len(embedded) == len(embeddings)
    assert {r['key'] for r in queries} == set(embedded)
    assert len(queries) == len(embedded) == stats['population']['performances']
    assert set(by_work) == {r['piece'] for r in queries}
    assert len(by_work) == stats['population']['works']
    audit = {(r['piece'], r['grid']): r for r in read_csv(sources['audit'])}

    performances, excluded, seen_hashes = [], [], {}
    original_membership = defaultdict(list)
    for query in sorted(queries, key=lambda r: r['key']):
        key, piece = query['key'], query['piece']
        meta, embedding = catalog[key], embedded[key]
        for field in ['composer', 'track', 'artist']:
            assert meta[field] == query[field], (key, field)
        assert meta['piece'] == piece == embedding['piece'], key
        assert meta['midi_sha256'] and key, key
        original_membership[piece].append(key)
        era, subgroup, boundary = COMPOSERS[query['composer']]
        row = {'key': key, 'piece': piece, 'grid': embedding['grid'],
               'track': query['track'], 'composer': query['composer'],
               'artist': query['artist'], 'midi_sha256': meta['midi_sha256'],
               'era': era, 'era_ko': ERA_NAMES[era], 'subgroup': subgroup,
               'classification_basis': 'composer_style_proxy',
               'composition_date_verified': False, 'boundary_case': boundary,
               'in_without_beethoven': boundary != 'beethoven',
               'in_without_rachmaninoff': boundary != 'rachmaninoff',
               'in_without_both_boundaries': not boundary}
        digest = meta['midi_sha256']
        if digest in seen_hashes:
            retained = seen_hashes[digest]
            assert retained['piece'] == piece, 'Cross-work duplicate needs manual review'
            excluded.append(dict(row, reason='duplicate_midi_sha256', retained_key=retained['key']))
        else:
            performances.append(row)
            seen_hashes[digest] = row

    works = []
    for piece in sorted(by_work):
        members = [r for r in performances if r['piece'] == piece]
        source = by_work[piece][0]
        assert len(original_membership[piece]) == int(source['performances']), piece
        assert len({r['grid'] for r in members}) == 1, piece
        grid = members[0]['grid']
        quality = audit[piece, grid]
        assert quality['accepted'] == 'True', piece
        assert int(quality['shared_beats']) == int(source['shared_beats']) >= 32, piece
        assert float(quality['shared_fraction']) >= .5 and len(members) >= 5, piece
        era, subgroup, boundary = COMPOSERS[source['composer']]
        works.append({'piece': piece, 'grid': grid, 'track': source['track'],
                      'composer': source['composer'], 'era': era, 'era_ko': ERA_NAMES[era],
                      'subgroup': subgroup, 'boundary_case': boundary,
                      'classification_basis': 'composer_style_proxy',
                      'composition_date_verified': False,
                      'recordings_original': len(original_membership[piece]),
                      'recordings_retained': len(members),
                      'artists_retained': len({r['artist'].casefold().strip() for r in members}),
                      'shared_beats_original_cohort': int(quality['shared_beats']),
                      'shared_fraction_original_cohort': quality['shared_fraction'],
                      'in_without_beethoven': boundary != 'beethoven',
                      'in_without_rachmaninoff': boundary != 'rachmaninoff',
                      'in_without_both_boundaries': not boundary,
                      'original_keys': '|'.join(original_membership[piece]),
                      'retained_keys': '|'.join(r['key'] for r in members)})

    summaries = []
    for era in ERA_ORDER:
        group = [r for r in works if r['era'] == era]
        summaries.append({'era': era, 'era_ko': ERA_NAMES[era],
                          'composers': len({r['composer'] for r in group}),
                          'work_score_groups': len(group),
                          'recordings_original': sum(r['recordings_original'] for r in group),
                          'recordings_retained': sum(r['recordings_retained'] for r in group)})
    composer_summary = []
    for composer in sorted(COMPOSERS):
        group = [r for r in works if r['composer'] == composer]
        era, subgroup, boundary = COMPOSERS[composer]
        composer_summary.append({'composer': composer, 'era': era, 'era_ko': ERA_NAMES[era],
                                 'subgroup': subgroup, 'boundary_case': boundary,
                                 'work_score_groups': len(group),
                                 'recordings_retained': sum(r['recordings_retained'] for r in group)})
    sensitivity = []
    for name, field in [('primary', None), ('without_beethoven', 'in_without_beethoven'),
                        ('without_rachmaninoff', 'in_without_rachmaninoff'),
                        ('without_both_boundaries', 'in_without_both_boundaries')]:
        for era in ERA_ORDER:
            group = [r for r in works if r['era'] == era and (field is None or r[field])]
            sensitivity.append({'cohort': name, 'era': era, 'era_ko': ERA_NAMES[era],
                                'work_score_groups': len(group),
                                'recordings_retained': sum(r['recordings_retained'] for r in group)})

    assert len(performances) + len(excluded) == len(queries)
    assert len({r['midi_sha256'] for r in performances}) == len(performances)
    assert sum(r['recordings_retained'] for r in works) == len(performances)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in [('works', works), ('performances', performances),
                       ('excluded_performances', excluded), ('era_summary', summaries),
                       ('composer_summary', composer_summary), ('sensitivity_cohorts', sensitivity)]:
        if rows:
            write_csv(out / f'{name}.csv', rows)
    protocol = {
        'schema': 'classicfy-era-cohort-v1', 'dataset': 'ATEPP-1.2',
        'source_population': {k: stats['population'][k] for k in ['works', 'performances', 'composers']},
        'selection': 'All existing five-feature shared-grid eligible groups; no feature-outcome selection.',
        'duplicate_policy': 'Global identical MIDI SHA256: retain lexicographically smallest recording key.',
        'classification': 'Composer-style proxy; composition dates and individual work styles not verified.',
        'composer_mapping': COMPOSERS,
        'classification_reference': 'https://www.boosey.com/composer/sergei+rachmaninoff',
        'weighting': 'Within era, each work/score group has equal total weight; recordings share its weight.',
        'group_unit': 'Existing piece+grid; may be a movement; parent works are not resolved.',
        'coverage': 'Original common seven-channel mask is the reference; original membership retained in works.csv.',
        'execution': 'Metadata selection only; no raw feature extraction, cache inspection, normalization refit, or statistical comparison.',
        'retained': {'groups': len(works), 'recordings': len(performances),
                     'composers': len({r['composer'] for r in works}), 'excluded_recordings': len(excluded)},
        'input_sha256': {str(p.relative_to(ANALYSIS)): sha256(p) for p in sources.values()},
        'recorded_hash_verification': verified_inputs,
        'script_sha256': sha256(Path(__file__).resolve()),
    }
    (out / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'retained': protocol['retained'], 'eras': summaries}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ANALYSIS / 'era_cohort')
    prepare(parser.parse_args().out)
