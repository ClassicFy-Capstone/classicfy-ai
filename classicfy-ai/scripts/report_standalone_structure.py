"""Report standalone and blended arms on every unchanged endpoint."""

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
METHODS = ['B', 'T', 'F', 'P', 'H', 'P+H', 'B+F', 'B+P', 'B+H', 'B+P+H', 'M-CT']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'classicfy-ai/analysis/standalone_structure')
    args = parser.parse_args(); out = args.out
    s = pd.read_csv(out/'summary.csv'); c = pd.read_csv(out/'controls.csv')
    controls = []
    for (role, method, axis), group in c.groupby(['role', 'method', 'axis']):
        values = group.groupby('seed').score.mean()
        controls.append(dict(role=role, method=method, axis=axis, mean=float(values.mean()),
            p05=float(values.quantile(.05)), p95=float(values.quantile(.95)), seeds=len(values)))
    pd.DataFrame(controls).to_csv(out/'control_summary.csv', index=False)
    contrasts = []
    for role in ('test', 'validation'):
        for axis in ('E-L', 'E-T', 'E-R', 'E-S'):
            t = s[(s.role == role)&(s.axis == axis)].set_index('method')
            for added, reference in [('P', 'F'), ('H', 'F'), ('P+H', 'F'),
                                     ('P', 'B+P'), ('H', 'B+H'), ('P+H', 'B+P+H')]:
                contrasts.append(dict(role=role, axis=axis, added=added, reference=reference,
                    delta_points=float(100*(t.loc[added, 'score']-t.loc[reference, 'score']))))
    pd.DataFrame(contrasts).to_csv(out/'contrasts.csv', index=False)
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), layout='constrained')
    labels = {'E-L': 'Level / spread', 'E-T': 'Delta size distribution',
              'E-R': 'Tempo-Dynamics relation', 'E-S': 'Provisional phrase-phase signs'}
    colors = ['#a3a8ad', '#bbbbbb', '#7195af', '#d18b42', '#9873ad', '#b96e76',
              '#bed0dd', '#edcaa1', '#d9c8e2', '#dfb7bb', '#d1d1d1']
    for row, role in enumerate(('test', 'validation')):
        for col, axis in enumerate(('E-L', 'E-T', 'E-R', 'E-S')):
            ax = axes[row, col]
            t = s[(s.role == role)&(s.axis == axis)].set_index('method')
            ax.bar(np.arange(len(METHODS)), t.loc[METHODS, 'score']*100, color=colors)
            ax.axhline(50, color='#555555', ls='--', lw=.8); ax.set_ylim(40, 85)
            ax.set_xticks(np.arange(len(METHODS)), METHODS, rotation=60, ha='right', fontsize=8)
            ax.set_title(role+' / '+axis+'\n'+labels[axis], fontsize=10)
            ax.set_ylabel('Held-out S@5 (0-100)')
    fig.suptitle('Remove global B: standalone context responses on the exact same 335 recordings\n'
        'Three test and three validation families; proxy consistency, no phrase/key ground truth', fontsize=12)
    for ext in ('png', 'pdf'):
        fig.savefig(out/('01_all_endpoints.'+ext), dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    names = ['F', 'P', 'H', 'P+H', 'B+P', 'B+H', 'B+P+H']
    cs = pd.DataFrame(controls)
    for ax, role in zip(axes, ('test', 'validation')):
        t = s[(s.role == role)&(s.axis == 'E-S')].set_index('method')
        balanced = cs[(cs.role == role)&(cs.axis == 'E-S')].set_index('method')
        x = np.arange(len(names))
        ax.scatter(x, t.loc[names, 'score']*100, label='Actual retrieval', marker='D', color='#bb7640')
        ax.scatter(x, t.loc[names, 'same_work_expected']*100, label='Same returned work, random recording', marker='o', color='#708b9c')
        means = balanced.loc[names, 'mean'].to_numpy()*100
        lo = balanced.loc[names, 'p05'].to_numpy()*100; hi = balanced.loc[names, 'p95'].to_numpy()*100
        ax.errorbar(x, means, yerr=[means-lo, hi-means], fmt='s', color='#8d6da1', capsize=3,
            label='3 per work, 20 seeds (p05-p95)')
        ax.axhline(50, color='#555555', ls='--', lw=.8); ax.set_ylim(47, 62)
        ax.set_xticks(x, names, rotation=40, ha='right'); ax.set_title(role+' / E-S')
        ax.set_ylabel('Phrase-phase sign consistency S@5'); ax.legend(fontsize=8, loc='upper left')
    fig.suptitle('Better phase scores alone do not establish a recording-specific advantage\n'
        'Seed ranges describe gallery sampling, not population confidence intervals', fontsize=11)
    for ext in ('png', 'pdf'):
        fig.savefig(out/('02_phase_controls.'+ext), dpi=160)
    plt.close(fig)
    phrase = pd.read_csv(ROOT/'classicfy-ai/analysis/phrase_response/phrase_hypotheses.csv')
    evidence = Counter(e for value in phrase.stop_evidence for e in ast.literal_eval(value))
    tonal = json.loads((ROOT/'classicfy-ai/analysis/tonal_tension/summary.json').read_text())
    tonal.pop('artifact', None)
    audit = dict(phrase_hypotheses=len(phrase), expert_confirmed_phrases=0, stop_evidence=dict(evidence),
        fixed_length_cap_fraction=evidence['fixed_length_cap']/len(phrase),
        tonal_summary=tonal,
        interpretation='Input availability and arithmetic are verified; musical segmentation/local key accuracy is unmeasured. Limited score rules are a modeling choice, not a runtime fallback caused by unavailable XML.')
    (out/'capture_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    p = out/'provenance.json'; prov = json.loads(p.read_text())
    for code in (Path(__file__), Path(__file__).with_name('verify_standalone_structure.py'),
                 Path(__file__).with_name('verify_structure_response_comparison.py')):
        prov['code'][str(code.relative_to(ROOT))] = hashlib.sha256(code.read_bytes()).hexdigest()
    prov['results'] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                      for f in out.iterdir() if f.is_file() and f.name != 'provenance.json'}
    p.write_text(json.dumps(prov, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
