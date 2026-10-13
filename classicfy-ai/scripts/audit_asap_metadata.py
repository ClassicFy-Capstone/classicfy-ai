"""Audit work-type metadata and candidate availability for a saved CNN cohort.

No CNN distances, feature distances, new training, or performer identity claims.
Unknown titles remain unknown; explicit dataset-name rules are not musicology labels.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from validate_tempo import plt, setup_style

ROOT = Path(__file__).resolve().parents[2]
TYPE_RULES = (
    ('sonata', r'^(?:piano_sonatas?|keyboard_sonatas?|sonatas?)(?:_|$)'),
    ('etude', r'^(?:etudes?|transcendental_etudes?|concert_etudes?|gran_etudes_de_paganini)(?:_|$)'),
    ('prelude', r'^preludes?(?:_|$)'),
    ('fugue', r'^fugues?(?:_|$)'),
    ('ballade', r'^ballades?(?:_|$)'),
    ('impromptu', r'^impromptus?(?:_|$)'),
    ('scherzo', r'^scherzos?(?:_|$)'),
    ('barcarolle', r'^barcarolles?(?:_|$)'),
)
LABELS = dict(sonata='소나타', etude='에튀드', prelude='전주곡', fugue='푸가',
              ballade='발라드', impromptu='즉흥곡', scherzo='스케르초',
              barcarolle='뱃노래', unclassified='미분류')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows to export: {path}')
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def unique_rows(rows, field):
    result = {}
    for row in rows:
        value = row.get(field)
        if not value or value in result:
            raise ValueError(f'Missing or duplicate {field}: {value}')
        result[value] = row
    return result


def named_type(value):
    normalized = re.sub(r'\s+', '_', value.strip()).casefold()
    matches = [name for name, pattern in TYPE_RULES if re.match(pattern, normalized)]
    if len(matches) > 1:
        raise ValueError(f'Overlapping classification rules: {value}')
    return matches[0] if matches else ''


def classify_work(title, folder):
    """Require non-conflicting title/folder evidence, retaining both sources."""
    parts = folder.strip('/').split('/')
    subgroup = parts[1] if len(parts) > 1 else ''
    title_type, folder_type = named_type(title), named_type(subgroup)
    if title_type and folder_type and title_type != folder_type:
        label, status = 'unclassified', 'conflicting_title_folder'
    elif title_type and folder_type:
        label, status = title_type, 'title_and_folder_agree'
    elif title_type:
        label, status = title_type, 'title_only'
    elif folder_type:
        label, status = folder_type, 'folder_only'
    else:
        label, status = 'unclassified', 'no_explicit_type_label'
    return dict(work_type=label, classification_status=status,
                title_type=title_type, folder_type=folder_type, source_subgroup=subgroup)


def work_family(composer, title):
    """Conservative exclusions for named sonata movements and Bach BWV pairs.

    Catalogue collections (e.g. all op.10 etudes) are not merged. This is a
    documented operational grouping, not a universal composition identifier.
    """
    match = re.match(r'^((?:Piano_Sonatas?|Keyboard_Sonatas?|Sonatas?)_\d+)[_-]'
                     r'(?:\d+(?:st|nd|rd|th)?|[ivx]+)(?:_|$)', title, re.IGNORECASE)
    if match:
        return f'{composer}::{match.group(1)}', 'sonata_movement_group'
    match = re.fullmatch(r'(?:Fugue|Prelude)_bwv_(\d+)', title, re.IGNORECASE)
    if composer == 'Bach' and match:
        return f'Bach::BWV_{match.group(1)}', 'bach_prelude_fugue_catalogue_group'
    return f'{composer}::{title}', 'composer_title_identity'


def join_cohort(cohort, metadata, split):
    """Exact key/score/grid metadata join, never infer an alias for a missing key."""
    indexed = unique_rows(metadata, 'midi_performance')
    unique_rows(cohort, 'key')
    rows = []
    for row in cohort:
        key = row['key']
        if key not in indexed:
            raise ValueError(f'CNN key missing from ASAP metadata: {key}')
        source = indexed[key]
        if row['piece'] != source['midi_score']:
            raise ValueError(f'Score identity mismatch: {key}')
        if row['split'] not in ('train', 'validation', 'test') or split.get(row['piece']) != row['split']:
            raise ValueError(f'Reference split mismatch: {key}')
        if not all(source.get(field) for field in ('composer', 'title', 'folder')):
            raise ValueError(f'Incomplete semantic metadata: {key}')
        family, rule = work_family(source['composer'], source['title'])
        rows.append(dict(key=key, piece=row['piece'], grid=row['grid'], split=row['split'],
            windows=int(row['windows']), composer=source['composer'], title=source['title'],
            folder=source['folder'], work_id=f'{source["composer"]}::{source["title"]}',
            family_id=family, family_rule=rule, **classify_work(source['title'], source['folder'])))
    # One canonical title cannot silently become several types across variants.
    labels = defaultdict(set)
    for row in rows:
        labels[row['work_id']].add(row['work_type'])
    if any(len(values) != 1 for values in labels.values()):
        raise ValueError('Canonical work has inconsistent type labels')
    return rows


def validate_vectors(cohort, vectors, keys, manifest):
    if keys.tolist() != [r['key'] for r in cohort]:
        raise ValueError('CSV and NPZ key order differs')
    if vectors.shape != (len(cohort), 128) or not np.isfinite(vectors).all():
        raise ValueError('Expected finite 128D saved vectors')
    exported = np.array([[float(row[f'z{i}']) for i in range(128)] for row in cohort])
    if not np.allclose(exported, vectors, atol=1e-12, rtol=0):
        raise ValueError('CSV vector export differs from NPZ')
    if np.any(np.linalg.norm(vectors, axis=1) <= 1e-12):
        raise ValueError('Saved vector has undefined cosine normalization')
    if len(manifest) != len(cohort):
        raise ValueError('NPZ identity manifest length differs')
    for row, identity in zip(cohort, manifest, strict=True):
        expected = {field: row[field] for field in ('key', 'piece', 'grid', 'split')}
        expected['windows'] = int(row['windows'])
        if identity != expected:
            raise ValueError(f'NPZ identity manifest differs: {row["key"]}')
    return float(np.max(np.abs(exported - vectors)))


def counts(rows):
    return dict(performances=len(rows), works=len({r['work_id'] for r in rows}),
                families=len({r['family_id'] for r in rows}), composers=len({r['composer'] for r in rows}))


def summarize(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    return [dict(zip(fields, identity, strict=True)) | counts(group)
            for identity, group in sorted(groups.items())]


def scope_rows(rows):
    return [('all', rows), ('train', [r for r in rows if r['split'] == 'train'])]


def candidate_availability(rows):
    """Count metadata-only candidates before observing ANY similarity outcome.

    Both B (different type) and C (same type) have the query composer. A/B/C
    belong to distinct conservative families. Counts do not establish that B
    will be feature-close or C feature-far under a later style threshold.
    """
    query_rows, directed = [], []
    for scope, pool in scope_rows(rows):
        for query in pool:
            eligible = [r for r in pool if r['composer'] == query['composer']
                        and r['family_id'] != query['family_id'] and r['work_type'] != 'unclassified']
            same = [r for r in eligible if r['work_type'] == query['work_type']]
            different = [r for r in eligible if r['work_type'] != query['work_type']]
            sc = Counter(r['family_id'] for r in same)
            dc = Counter(r['family_id'] for r in different)
            bc_pairs = len(same) * len(different) - sum(sc[f] * dc[f] for f in sc)
            supported = query['work_type'] != 'unclassified' and bc_pairs > 0
            reason = ('unclassified_query' if query['work_type'] == 'unclassified' else
                      'no_same_type_other_family' if not same else
                      'no_different_type_same_composer' if not different else
                      'no_distinct_B_C_families' if not bc_pairs else 'supported')
            query_rows.append(dict(scope=scope, key=query['key'], composer=query['composer'],
                work_type=query['work_type'], work_id=query['work_id'], family_id=query['family_id'],
                same_type_candidates=len(same), same_type_candidate_families=len(sc),
                different_type_candidates=len(different), different_type_candidate_families=len(dc),
                distinct_family_B_C_pairs=bc_pairs, metadata_supported=bool(supported), reason=reason))
            if not supported:
                continue
            for other_type in sorted({r['work_type'] for r in different}):
                others = [r for r in different if r['work_type'] == other_type]
                oc = Counter(r['family_id'] for r in others)
                available = len(same) * len(others) - sum(sc[f] * oc[f] for f in sc)
                if available:
                    directed.append(dict(scope=scope, key=query['key'], composer=query['composer'],
                        query_type=query['work_type'], other_type=other_type,
                        work_id=query['work_id'], family_id=query['family_id'],
                        possible_B_C_pairs=available))
    return query_rows, directed


def pilot_options(rows, directed):
    """Prioritize train coverage with >=2 families/type and both directions.

    This is a data-availability decision, not a search for favorable CNN cases.
    """
    train = [r for r in rows if r['split'] == 'train' and r['work_type'] != 'unclassified']
    options = []
    for composer in sorted({r['composer'] for r in train}):
        local = [r for r in train if r['composer'] == composer]
        for a, b in combinations(sorted({r['work_type'] for r in local}), 2):
            ca, cb = counts([r for r in local if r['work_type'] == a]), counts([r for r in local if r['work_type'] == b])
            da = [r for r in directed if r['scope'] == 'train' and r['composer'] == composer
                  and r['query_type'] == a and r['other_type'] == b]
            db = [r for r in directed if r['scope'] == 'train' and r['composer'] == composer
                  and r['query_type'] == b and r['other_type'] == a]
            supported = ca['families'] >= 2 and cb['families'] >= 2 and bool(da) and bool(db)
            options.append(dict(composer=composer, type_a=a, type_b=b,
                type_a_performances=ca['performances'], type_b_performances=cb['performances'],
                type_a_works=ca['works'], type_b_works=cb['works'],
                type_a_families=ca['families'], type_b_families=cb['families'],
                supported_queries_a=len(da), supported_queries_b=len(db),
                total_supported_queries=len(da) + len(db), bidirectional_pilot=bool(supported)))
    options.sort(key=lambda r: (not r['bidirectional_pilot'], -r['total_supported_queries'],
                               r['composer'], r['type_a'], r['type_b']))
    return options


def work_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['work_id']].append(row)
    result = []
    for work, group in sorted(grouped.items()):
        source = group[0]
        result.append({field: source[field] for field in ('work_id', 'family_id', 'family_rule',
            'composer', 'title', 'work_type', 'classification_status', 'title_type', 'folder_type')} |
            dict(folders='|'.join(sorted({r['folder'] for r in group})),
                 splits='|'.join(sorted({r['split'] for r in group})), **counts(group)))
    return result


def family_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['family_id']].append(row)
    return [dict(family_id=family, composer=group[0]['composer'],
                 family_rule=group[0]['family_rule'],
                 work_types='|'.join(sorted({r['work_type'] for r in group})),
                 splits='|'.join(sorted({r['split'] for r in group})),
                 spans_splits=len({r['split'] for r in group}) > 1, **counts(group))
            for family, group in sorted(groups.items())]


def figures(rows, out):
    setup_style()
    types = sorted({r['work_type'] for r in rows}, key=lambda t: (t == 'unclassified', LABELS[t]))
    composers = sorted({r['composer'] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, field, label in zip(axes, ('performances', 'works'), ('연주 수', '작품/악장 수')):
        values = [counts([r for r in rows if r['work_type'] == t])[field] for t in types]
        ax.barh(np.arange(len(types)), values,
                color=['#999999' if t == 'unclassified' else '#2878c8' for t in types])
        ax.set_yticks(np.arange(len(types)), [LABELS[t] for t in types])
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.18)
        ax.set_xlabel(label)
        for y, value in enumerate(values):
            ax.text(value + max(values) * .015, y, str(value), va='center', fontsize=11)
    fig.suptitle('현재 임베딩 대상의 작품 유형 분포', fontsize=21, fontweight='bold')
    fig.text(.5, .025, 'ASAP 570연주 · 작품 유형은 title/folder의 명시적 이름에서 파생 · 미분류 보존 · 전체 split 기술통계', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .07, 1, .92))
    save_figure(fig, out, 'type_distribution')
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, field, title in zip(axes, ('works', 'performances'), ('작품/악장 수', '연주 수')):
        matrix = np.array([[counts([r for r in rows if r['composer'] == c and r['work_type'] == t])[field]
                            for t in types] for c in composers])
        ax.imshow(np.log1p(matrix), cmap='Blues', aspect='auto')
        for y, x in np.ndindex(matrix.shape):
            ax.text(x, y, str(matrix[y, x]), ha='center', va='center', fontsize=10,
                    color='white' if matrix[y, x] > np.expm1(np.log1p(matrix.max()) * .6) else '#222222')
        ax.set_yticks(np.arange(len(composers)), composers)
        ax.set_xticks(np.arange(len(types)), [LABELS[t] for t in types], rotation=45, ha='right')
        ax.set_title(title, loc='left', fontsize=17)
    fig.suptitle('작곡가 × 작품 유형: 후보 구성 확인', fontsize=21, fontweight='bold')
    fig.text(.5, .025, '숫자는 실제 count · 색은 log1p(count) · 같은 작곡가에서 여러 유형이 있어야 작곡가를 고정한 비교가 가능', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .07, 1, .93))
    save_figure(fig, out, 'composer_type_matrix')


def save_figure(fig, out, name):
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(out / f'{name}.{ext}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asap-root', type=Path, default=ROOT.parent / 'datasets/ASAP')
    parser.add_argument('--run', type=Path, default=ROOT.parent / 'datasets/temporal_embedding_cluster_run01')
    parser.add_argument('--features', type=Path, default=ROOT / 'classicfy-ai/analysis/_feature_search_roles/embeddings.csv')
    parser.add_argument('--previous-audit', type=Path, default=ROOT / 'classicfy-ai/analysis/temporal_neighbors/stats.json')
    parser.add_argument('--out', type=Path, default=ROOT / 'classicfy-ai/analysis/metadata_style_validation')
    args = parser.parse_args()
    reference = json.loads((args.run / 'dataset.json').read_text())
    metadata_path = args.asap_root / 'metadata.csv'
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.asap_root, text=True).strip()
    if revision != reference['source_provenance']['asap_revision']:
        raise ValueError('ASAP revision differs from the saved run')
    # Detect working-tree changes to the metadata, not just the committed revision.
    committed_metadata = subprocess.check_output(['git', 'show', f'{revision}:metadata.csv'], cwd=args.asap_root)
    if metadata_path.read_bytes() != committed_metadata:
        raise ValueError('ASAP metadata differs from its declared revision')
    cohort = read_csv(args.run / 'embeddings.csv')
    features = read_csv(args.features)
    feature_lookup = unique_rows(features, 'key')
    if set(feature_lookup) != {r['key'] for r in cohort}:
        raise ValueError('Existing feature and saved CNN cohorts differ')
    for row in cohort:
        if any(row[field] != feature_lookup[row['key']][field] for field in ('piece', 'grid')):
            raise ValueError(f'Feature identity differs: {row["key"]}')
    with np.load(args.run / 'embeddings.npz', allow_pickle=False) as archive:
        export_error = validate_vectors(cohort, archive['vectors'], archive['keys'],
                                        json.loads(str(archive['manifest'])))
    rows = join_cohort(cohort, read_csv(metadata_path), reference['split'])
    query_support, directed = candidate_availability(rows)
    options = pilot_options(rows, directed)
    distribution = summarize(rows, ('work_type',))
    matrix = summarize(rows, ('composer', 'work_type'))
    split_distribution = summarize(rows, ('split', 'work_type'))
    works = work_rows(rows)
    families = family_rows(rows)
    review = [r for r in works if r['classification_status'] != 'title_and_folder_agree']
    excluded_metadata = [{field: r[field] for field in ('midi_performance', 'composer', 'title', 'folder')}
                         for r in read_csv(metadata_path) if r['midi_performance'] not in {i['key'] for i in rows}]
    args.out.mkdir(parents=True, exist_ok=True)
    exports = {'performances.csv': rows, 'works.csv': works, 'families.csv': families,
               'type_distribution.csv': distribution, 'composer_type_counts.csv': matrix,
               'split_type_counts.csv': split_distribution, 'candidate_availability.csv': query_support,
               'directed_type_candidates.csv': directed, 'pilot_options.csv': options,
               'classification_review.csv': review, 'outside_embedding_cohort.csv': excluded_metadata}
    for name, records in exports.items():
        write_csv(args.out / name, records)
    figures(rows, args.out)
    previous = json.loads(args.previous_audit.read_text())
    inputs = {'metadata.csv': metadata_path, 'embeddings.csv': args.run / 'embeddings.csv',
              'embeddings.npz': args.run / 'embeddings.npz', 'dataset.json': args.run / 'dataset.json',
              'training.json': args.run / 'training.json', 'best.pt': args.run / 'best.pt',
              'feature_embeddings.csv': args.features, 'previous_audit.json': args.previous_audit}
    input_hashes = {name: sha256(path) for name, path in inputs.items()}
    pilot = next((r for r in options if r['bidirectional_pilot']), None)
    info = dict(**counts(rows), classified_performances=sum(r['work_type'] != 'unclassified' for r in rows),
                unclassified_performances=sum(r['work_type'] == 'unclassified' for r in rows),
                unclassified_works=sum(r['work_type'] == 'unclassified' for r in works),
                metadata_rows=len(read_csv(metadata_path)), all_keys_joined=True,
                outside_embedding_cohort=len(excluded_metadata),
                availability={scope: dict(queries=sum(r['scope'] == scope for r in query_support),
                    supported_queries=sum(r['scope'] == scope and r['metadata_supported'] for r in query_support),
                    reason_counts=dict(Counter(r['reason'] for r in query_support if r['scope'] == scope)))
                    for scope in ('all', 'train')},
                family_split_overlap=sum(r['spans_splits'] for r in families),
                proposed_pilot=pilot)
    protocol = dict(**info, source_run=str(args.run.resolve()), source_provenance=reference['source_provenance'],
        input_sha256=input_hashes, code_sha256=sha256(Path(__file__)), rules=TYPE_RULES,
        canonical_work='ASAP (composer,title); movements remain separate in work counts',
        family_policy='merge named sonata movements and same-BWV Bach prelude/fugue; do not merge opus collections',
        candidate_policy='same composer for A/B/C; C same type, B different type; all three families distinct',
        primary_scope='train only; all-split counts are descriptive availability, not generalization performance',
        pilot_selection='train >=2 families/type and both directions supported; maximize query coverage then lexical ties',
        csv_npz_export_max_error=export_error,
        prior_report_identity=dict(checkpoint_hash_matches=input_hashes['best.pt'] == previous['checkpoint_sha256'],
                                   embedding_archive_hash_matches=input_hashes['embeddings.npz'] == previous['embedding_sha256']),
        limits=['Work types are dataset-name-derived labels, not verified formal analysis or service genre tags.',
                'Same-composer candidate availability does not guarantee feature-close B or feature-far C.',
                'No CNN/feature distances or mapping success were evaluated; no encoder re-training.',
                'Current local saved run differs from prior temporal-neighbor checkpoint/archive hashes; old distance results are not transferred.',
                'Family rules are conservative exclusions, not guaranteed complete musical family resolution; some families span original splits.',
                'No validated performer identities, eras, mood tags, listening labels or recommendation ground truth are introduced.'])
    (args.out / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
    (args.out / 'summary.json').write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n')
    verification = dict(metadata_revision=revision, committed_metadata_bytes_equal=True,
        csv_npz_export_max_error=export_error, feature_cohort_identity_equal=True,
        performance_rows=len(rows), summed_type_performances=sum(r['performances'] for r in distribution),
        summed_type_works=sum(r['works'] for r in distribution),
        csv_sha256={name: sha256(args.out / name) for name in exports})
    (args.out / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
