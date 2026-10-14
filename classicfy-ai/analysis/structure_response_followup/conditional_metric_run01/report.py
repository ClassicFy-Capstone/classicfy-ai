"""Summarize seed variability and local-vs-cross-work evaluation in figures."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
MODES = ('fused_no_metric', 'separate_no_metric', 'fused_metric', 'separate_metric')
LABELS = ('Score fused\nNo distance loss', 'Score separate\nNo distance loss', 'Score fused\nDistance loss', 'Score separate\nDistance loss')
COLORS = ('#93a1b2', '#6885a7', '#e3ad61', '#188c78')


def read(name):
    with (OUT / name).open() as f: return list(csv.DictReader(f))


def lookup(rows, **keys):
    match = [r for r in rows if all(r.get(k) == v for k, v in keys.items())]
    assert len(match) == 1, keys
    return match[0]


def main():
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 160})
    summary = read('summary.csv'); passage = read('passage_summary.csv'); controls = read('control_summary.csv')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for col, (category, axis) in enumerate((('P', 'E-PN'), ('H', 'E-HL'))):
        for row, scope in enumerate(('within_work', 'cross_work')):
            ax = axes[row, col]
            values = [lookup(summary, scope=scope, role='test', method=m + '_' + category) for m in MODES]
            scores = [float(r[axis]) for r in values]; errors = [float(r[axis + '_seed_sd']) for r in values]
            ax.bar(range(4), scores, yerr=errors, color=COLORS, capsize=4)
            ax.set_xticks(range(4), LABELS, fontsize=9)
            ax.set_ylim(min(35, min(v - e for v, e in zip(scores, errors)) - 5), max(78, max(v + e for v, e in zip(scores, errors)) + 8))
            ax.set_ylabel('Numeric proxy score (not accuracy)')
            ax.set_title(('Within heldout work' if row == 0 else 'Cross-work catalog') + ': ' + ('phrase shape' if col == 0 else 'tension lag'))
            for i, value in enumerate(scores): ax.text(i, value + errors[i] + .9, f'{value:.2f}', ha='center')
            if row == 0: ax.axhline(50, color='#777', ls=':', lw=1, label='Random recording = 50')
            else:
                prior = float(lookup(summary, scope=scope, role='test', method='summary_' + category)[axis])
                ax.axhline(prior, color='#666', ls='--', lw=1, label='Prior segment summary')
            ax.legend(fontsize=8, loc='upper left')
    fig.suptitle('Four predeclared conditions: three-seed mean ± seed SD, both directions\nReused test benchmark: these error bars are not confidence intervals', fontsize=13)
    fig.savefig(OUT / '01_retrieval.png'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for ax, category in zip(axes, ('P', 'H')):
        values = [lookup(passage, role='test', mode=m, category=category) for m in MODES]
        rho = [float(r['spearman']) for r in values]; error = [float(r['spearman_seed_sd']) for r in values]
        ax.bar(range(4), rho, yerr=error, color=COLORS, capsize=4); ax.set_xticks(range(4), LABELS, fontsize=9)
        ax.set_ylim(min(-.1, min(v - e for v, e in zip(rho, error)) - .08), max(.8, max(v + e for v, e in zip(rho, error)) + .12))
        ax.set_ylabel('Mean within-passage distance rank correlation')
        ax.set_title('Phrase' if category == 'P' else 'Tension windows')
        for i, value in enumerate(rho): ax.text(i, value + error[i] + .025, f'{value:.3f}', ha='center')
    fig.suptitle('Observed target distance vs embedding distance on heldout works\nTarget fit diagnostic; not an independent perceptual style test', fontsize=13)
    fig.savefig(OUT / '02_passage_fit.png'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for ax, (category, axis) in zip(axes, (('P', 'E-PN'), ('H', 'E-HL'))):
        actual = lookup(summary, scope='cross_work', role='test', method='separate_metric_' + category)
        shuffled = lookup(controls, scope='cross_work', role='test', mode='separate_metric', kind=category, control='performance_order_permuted')
        balanced = lookup(controls, scope='cross_work', role='test', mode='separate_metric', kind=category, control='three_candidates')
        y = [float(actual[axis]), float(actual[axis + '_same_work']), float(shuffled[axis]), float(balanced[axis])]
        ax.bar(range(4), y, color=('#188c78', '#93a1b2', '#ba8085', '#6885a7'))
        ax.set_xticks(range(4), ('Actual', 'Same returned work\nRandom recording', 'Performance\norder shuffled', 'Candidate-count\nbalanced'), fontsize=9)
        ax.set_ylim(min(35, min(y) - 5), max(75, max(y) + 8)); ax.set_ylabel('Cross-work numeric proxy score'); ax.set_title('Phrase shape' if category == 'P' else 'Tension lag')
        for i, value in enumerate(y): ax.text(i, value + .7, f'{value:.2f}', ha='center')
        ax.axhline(50, ls=':', color='#777', lw=1)
    fig.suptitle('Primary separated model with distance loss: controls\nShuffle: 5 seeds; candidate count: 20 seeds, then mean across training seeds', fontsize=13)
    fig.savefig(OUT / '03_controls.png'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for ax, metric in zip(axes, ('metric', 'shape')):
        for mode, label, color in zip(MODES, LABELS, COLORS):
            histories = []
            for folder in sorted(OUT.glob(f'seed*_view*_{mode}')):
                with (folder / 'history.csv').open() as f: histories.append(list(csv.DictReader(f)))
            epochs = min(len(h) for h in histories)
            values = np.array([[float(r['validation_' + metric]) for r in h[:epochs]] for h in histories])
            ax.plot(range(epochs), values.mean(0), color=color, label=label.replace('\n', ', '))
        ax.set_xlabel('Epoch'); ax.set_ylabel('Group-balanced validation loss')
        ax.set_title('Distance target loss (also measured for no-loss controls)' if metric == 'metric' else 'Normalized response shape reconstruction')
        ax.legend(fontsize=8)
    fig.suptitle('Six runs per condition: both directions × three seeds', fontsize=13)
    fig.savefig(OUT / '04_learning.png'); plt.close(fig)
    print('Rendered four report figures.', flush=True)


if __name__ == '__main__': main()
