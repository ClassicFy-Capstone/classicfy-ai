"""Exploratory era comparisons on the fixed 22-per-era expression figures.

Run from the repository root with the existing analysis Python environment.
No feature re-extraction, cohort changes, or figure changes.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import kruskal, mannwhitneyu, rankdata

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/era_refine_expression'
ERAS = ['classical_baroque', 'romantic', 'impressionist_modern']
CHANNELS = ['rubato', 'dynamics', 'pedal_depth', 'pedal_down_ratio', 'pedal_changes']


def holm(pvalues):
    pvalues = np.asarray(pvalues, float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    adjusted[order] = np.minimum(1., np.maximum.accumulate(
        pvalues[order] * np.arange(len(pvalues), 0, -1)))
    return adjusted


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def exact_partitions(sizes):
    """Enumerate every labelled three-group allocation, within-group order ignored."""
    n = sum(sizes)
    partitions = []
    for a in itertools.combinations(range(n), sizes[0]):
        remaining = [i for i in range(n) if i not in a]
        for b in itertools.combinations(remaining, sizes[1]):
            c = [i for i in remaining if i not in b]
            partitions.append((*a, *b, *c))
    return np.asarray(partitions)


def rank_h(ranks, sizes):
    n = sum(sizes)
    result = np.zeros(ranks.shape[:-1])
    start = 0
    for size in sizes:
        result += ranks[..., start:start + size].sum(axis=-1) ** 2 / size
        start += size
    return 12 * result / (n * (n + 1)) - 3 * (n + 1)


def composer_test(groups, partitions):
    sizes = [len(g) for g in groups]
    values = np.concatenate(groups)
    ranks = rankdata(values)
    _, counts = np.unique(values, return_counts=True)
    n = len(values)
    tie_correction = 1 - np.sum(counts ** 3 - counts) / (n ** 3 - n)
    if not tie_correction:
        return 0., 1.
    observed = float(rank_h(ranks, sizes) / tie_correction)
    permuted = rank_h(ranks[partitions], sizes) / tie_correction
    assert np.isclose(observed, kruskal(*groups).statistic, atol=1e-12)
    # Exact enumeration includes the observed assignment; no Monte Carlo +1.
    p = float(np.mean(permuted >= observed - 1e-10))
    return observed, p


def main():
    source = OUT / 'balanced_work_values.csv'
    input_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    rows = list(csv.DictReader(source.open(encoding='utf-8-sig')))
    assert len(rows) == 330
    ids = {r['composition_id'] for r in rows}
    assert len(ids) == 66
    composers = {
        era: sorted({r['composer'] for r in rows if r['era'] == era}) for era in ERAS}
    sizes = [len(composers[e]) for e in ERAS]
    assert sizes == [5, 6, 2]
    partitions = exact_partitions(sizes)
    assert len(partitions) == 36036
    omnibus, descriptions, pairwise, composer_rows = [], [], [], []
    for channel in CHANNELS:
        channel_rows = [r for r in rows if r['channel'] == channel]
        assert {r['composition_id'] for r in channel_rows} == ids
        for kind in ['absolute', 'relative']:
            field = kind + '_display'
            groups, composer_groups = [], []
            for era in ERAS:
                era_rows = [r for r in channel_rows if r['era'] == era]
                values = np.array([float(r[field]) for r in era_rows])
                assert len(values) == 22 and np.isfinite(values).all()
                groups.append(values)
                descriptions.append(dict(channel=channel, kind=kind, era=era, n=22,
                    median=float(np.median(values)), q25=float(np.quantile(values, .25)),
                    q75=float(np.quantile(values, .75)),
                    recordings_used_median=float(np.median([int(r['recordings_used']) for r in era_rows]))))
                cg = []
                for composer in composers[era]:
                    cv = [float(r[field]) for r in era_rows if r['composer'] == composer]
                    representative = float(np.median(cv))
                    cg.append(representative)
                    composer_rows.append(dict(channel=channel, kind=kind, era=era,
                        composer=composer, works=len(cv), median=representative))
                composer_groups.append(np.array(cg))
            h, p = kruskal(*groups)
            # Independent rank-sum identity, including tie correction.
            pooled = np.concatenate(groups)
            _, counts = np.unique(pooled, return_counts=True)
            correction = 1 - np.sum(counts ** 3 - counts) / (66 ** 3 - 66)
            assert np.isclose(h, rank_h(rankdata(pooled), [22] * 3) / correction)
            ch, cp = composer_test(composer_groups, partitions)
            omnibus.append(dict(channel=channel, kind=kind, works=66,
                work_H=float(h), work_p=float(p),
                rank_effect_H_adjusted=max(0., float((h - 2) / 63)),
                composer_H=ch, composer_exact_p=cp, composer_allocations=len(partitions)))
            for a, b in itertools.combinations(range(3), 2):
                u, pp = mannwhitneyu(groups[a], groups[b], alternative='two-sided',
                                    method='asymptotic', use_continuity=True)
                delta = float(2 * u / (22 * 22) - 1)
                # Independent pair counting also checks sign and tie handling.
                manual = float(np.sign(groups[a][:, None] - groups[b][None, :]).mean())
                assert np.isclose(delta, manual)
                pairwise.append(dict(channel=channel, kind=kind, era_a=ERAS[a],
                    era_b=ERAS[b], cliffs_delta_a_minus_b=delta, p=float(pp)))
    for key in ['work_p', 'composer_exact_p']:
        for row, adjusted in zip(omnibus, holm([r[key] for r in omnibus])):
            row[key + '_holm10'] = float(adjusted)
    for row, adjusted in zip(pairwise, holm([r['p'] for r in pairwise])):
        row['p_holm30'] = float(adjusted)
    outputs = {
        'significance_omnibus.csv': omnibus,
        'significance_era_descriptives.csv': descriptions,
        'significance_pairwise.csv': pairwise,
        'significance_composer_values.csv': composer_rows,
    }
    for name, values in outputs.items():
        write_csv(OUT / name, values)
    protocol = dict(
        scope='Exploratory, fixed figure cohort: 22 works per era / 66 total; not all ATEPP.',
        input_sha256=input_hash, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        versions={'numpy': np.__version__, 'scipy': scipy.__version__},
        work_test='Kruskal-Wallis, chi-square approximation, 22 works per era; dependent works remain a limitation.',
        effect='max(0, (H-k+1)/(N-k)); rank effect estimate, not classification accuracy or raw variance explained.',
        composer_test='Median displayed value per composer, ranks, exact H upper-tail permutation; all 36036 allocations at fixed group sizes 5/6/2.',
        composer_limitation='Sensitivity analysis assuming composer exchangeability, not adjustment identifying era independent of composer; only Debussy/Ravel in modern group.',
        multiple_testing='Holm across 10 metrics separately for work and composer omnibus families. Pairwise Mann-Whitney asymptotic two-sided, continuity/tie correction, Holm across all 30 pairs.',
        relative_definition='Local absolute deviation after common-position median subtraction, not signed mean or prior tempo IQR.',
        limitations=['Composer, work/movement, performer/recording dependence; selected score/alignment availability.',
                     'Different recording counts and self-included reference; no fixed-recording-count sensitivity in this run.',
                     'Distribution significance does not imply nonoverlapping eras or held-out classification ability.',
                     'Failure to reject is not evidence of equality.'],
        references=['https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html',
                    'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html'],
        validation=['Exact 66 IDs and 22 per era for all 10 metrics.',
                    'Rank-sum H independently checked against SciPy; pairwise Cliff delta checked by pair counting.'],
        output_sha256={name: hashlib.sha256((OUT / name).read_bytes()).hexdigest() for name in outputs})
    assert input_hash == hashlib.sha256(source.read_bytes()).hexdigest()
    (OUT / 'significance_protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(omnibus, indent=2))


if __name__ == '__main__':
    main()
