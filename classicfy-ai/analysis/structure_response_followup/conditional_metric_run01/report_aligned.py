"""Show the after-results inference diagnostic without replacing primary results."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report import OUT, read, lookup, MODES, LABELS, COLORS


def main():
    rows = read('aligned_summary.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for ax, (category, axis) in zip(axes, (('P', 'E-PN'), ('H', 'E-HL'))):
        selected = [lookup(rows, role='test', method=mode + '_' + category) for mode in MODES]
        y = [float(r[axis]) for r in selected]; error = [float(r[axis + '_seed_sd']) for r in selected]
        ax.bar(range(4), y, yerr=error, color=COLORS, capsize=4)
        ax.set_xticks(range(4), LABELS, fontsize=9); ax.set_ylim(40, 65)
        ax.axhline(50, ls=':', color='#777'); ax.set_ylabel('Within-work numeric proxy score')
        ax.set_title('Phrase shape' if category == 'P' else 'Tension lag')
        for i, value in enumerate(y): ax.text(i, value + error[i] + .7, f'{value:.2f}', ha='center')
    fig.suptitle('Exploratory: compare only identical score passages at retrieval\nDefined after primary results; three-seed mean ± SD, reused test', fontsize=13)
    fig.savefig(OUT / '05_aligned_diagnostic.png', dpi=160); plt.close(fig)
    print('Rendered aligned diagnostic figure.', flush=True)


if __name__ == '__main__': main()
