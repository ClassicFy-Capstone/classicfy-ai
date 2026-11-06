"""Test whether between-performance style divergence follows score form in ATEPP.

Run from any directory: python analyze_structure_divergence.py --help.
Score structure comes only from the score MIDI. Divergence comes only from the
standardized ATEPP cache written by validate_atepp_normalization.py.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import stats

from atepp_feature_analysis import resolve_source
from validate_tempo import setup_style, chart, save, plt, BLUE, GREEN, ORANGE, GRAY, write_csv
from preprocessing.score_midi import load_score_midi

HERE = Path(__file__).resolve().parent
CHANNELS = ('rubato', 'dynamics', 'articulation', 'pedal_depth')
STYLE_CHANNELS = ('rubato', 'dynamics', 'articulation')
CHANNEL_NAMES = dict(composite='종합(R·D·A)', rubato='Rubato', dynamics='Dynamics',
                     articulation='Articulation', pedal_depth='Pedal depth')
# Score-only categories first; the two common-pattern categories come from performances.
CATEGORIES = dict(
    theme_inner='주제(반복) 재료 내부', theme_entry='주제 시작 ±1마디', theme_exit='주제 끝 ±1마디',
    nonrecurring='비반복 구간(경과·전개)', principal='도입 주제와 그 재현', novelty='구조 경계(novelty)',
    opening='곡 시작 16박', ending='곡 끝 16박',
    common_slow='공통 감속 지점(연주 기반)', common_slow_split='공통 감속(반분 교차검증)', common_peak='공통 셈여림 정점(연주 기반)')
# Non-recurring material is the reference level; overlapping positions get every matching term.
JOINT_MODELS = dict(score=('theme_inner', 'theme_entry', 'theme_exit', 'novelty', 'opening', 'ending'),
                    with_common=('theme_inner', 'theme_entry', 'theme_exit', 'novelty', 'opening', 'ending',
                                 'common_slow', 'common_peak'))
GENRES = (('sonata', 'Sonata'), ('etude', 'Étude|Etude|Études'), ('prelude', 'Prelude|Prélude'),
          ('nocturne', 'Nocturne'), ('mazurka', 'Mazurka'), ('waltz', 'Waltz|Valse'),
          ('variations', 'Variation'), ('ballade', 'Ballade'), ('scherzo', 'Scherzo'),
          ('impromptu', 'Impromptu'), ('fugue', 'Fugue|Prelude and Fugue'))


def parse_args():
    root = HERE.parents[2] / 'ATEPP_dataset'
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=root)
    p.add_argument('--cache', type=Path, default=root / 'features/dual_dtw/atepp_standardized.npz')
    p.add_argument('--audit', type=Path, default=HERE.parent / 'analysis/atepp_research/midi_audit.csv')
    p.add_argument('--out', type=Path, default=HERE.parent / 'analysis/structure_divergence')
    p.add_argument('--min-performances', type=int, default=5)
    p.add_argument('--window', type=int, default=16, help='recurrence window in quarter notes')
    p.add_argument('--threshold', type=float, default=.9, help='recurrence cosine threshold')
    p.add_argument('--preview-seconds', type=float, default=30.)
    return p.parse_args()


def load_groups(cache):
    with np.load(cache, allow_pickle=False) as z:
        manifest = json.loads(str(z['manifest']))
        groups = defaultdict(lambda: dict(values={}, common={}, scales={}))
        for i, record in enumerate(manifest['records']):
            if record['channel'] not in CHANNELS + ('tempo',):
                continue
            g = groups[record['piece']]
            g['grid'] = z[f'{i}/score_beats']
            g['scales'][record['channel']] = record['scale']['value']
            values = np.where(z[f'{i}/standardized_mask'], z[f'{i}/standardized_values'], np.nan)
            g['values'].setdefault(record['channel'], {})[record['key']] = values
            if record['channel'] not in g['common']:
                g['common'][record['channel']] = np.where(z[f'{i}/common_mask'], z[f'{i}/common_values'], np.nan)
    return groups


def pitch_roll(notes, grid):
    """Duration overlap of each pitch with each quarter interval, plus onset counts."""
    n = len(grid) - 1
    roll, onsets = np.zeros((n, 128)), np.zeros(n)
    start = grid[0]
    for note in notes:
        a, b = note.onset_beats - start, note.offset_beats - start
        if b <= 0 or a >= n:
            continue
        if 0 <= a < n:
            onsets[int(a)] += 1
        for q in range(max(int(np.floor(a)), 0), min(int(np.ceil(b)), n)):
            roll[q, note.pitch] += min(b, q + 1) - max(a, q)
    return roll, onsets


def diagonal_mean(m, length):
    k = len(m) - length + 1
    if k <= 0:
        return np.zeros((0, 0))
    out = np.zeros((k, k))
    for d in range(length):
        out += m[d:d + k, d:d + k]
    return out / length


def recurrence(roll, length, threshold):
    """Transposition-invariant repeated material, with one shift per diagonal path."""
    x = roll / np.maximum(np.linalg.norm(roll, axis=1, keepdims=True), 1e-12)
    best = None
    for shift in range(-6, 7):
        y = np.roll(x, shift, axis=1)
        if shift > 0:
            y[:, :shift] = 0
        elif shift < 0:
            y[:, shift:] = 0
        s = diagonal_mean(x @ y.T, length)
        best = s if best is None else np.maximum(best, s)
    k = len(best)
    idx = np.arange(k)
    best[np.abs(idx[:, None] - idx[None, :]) < length] = 0
    n = len(roll)
    strength = np.zeros(n)
    member = np.zeros(n, bool)
    window_strength = best.max(axis=1) if k else np.zeros(0)
    for i, value in enumerate(window_strength):
        strength[i:i + length] = np.maximum(strength[i:i + length], value)
        if value >= threshold:
            member[i:i + length] = True
    # Principal theme: windows in the first 2L quarters with sounding notes and their recurrences.
    principal = np.zeros(n, bool)
    for i in range(min(2 * length, k)):
        if roll[i:i + length].sum() == 0 or window_strength[i] < threshold:
            continue
        principal[i:i + length] = True
        for j in np.flatnonzero(best[i] >= threshold):
            principal[j:j + length] = True
    plain = x @ x.T
    return strength, member, principal, plain


def novelty_boundaries(ssm, size=16):
    n = len(ssm)
    sign = np.sign(np.arange(-size, size) + .5)
    gauss = np.exp(-.5 * (np.arange(-size, size) + .5) ** 2 / (size / 2) ** 2)
    kernel = np.outer(sign, sign) * np.outer(gauss, gauss)
    padded = np.pad(ssm, size)
    nov = np.array([np.sum(padded[i:i + 2 * size, i:i + 2 * size] * kernel) for i in range(n)])
    nov = np.maximum(nov, 0)
    peaks = [i for i in range(1, n - 1) if nov[i] >= nov[i - 1] and nov[i] > nov[i + 1]
             and nov[i] >= np.quantile(nov, .9)]
    keep = []
    for i in sorted(peaks, key=lambda i: -nov[i]):
        if all(abs(i - j) >= 2 * size for j in keep):
            keep.append(i)
    return sorted(keep)


def zone(n, points, radius):
    z = np.zeros(n, bool)
    for p in points:
        z[max(p - radius, 0):min(p + radius + 1, n)] = True
    return z


def run_edges(mask, min_length):
    starts, ends, i = [], [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            if j - i >= min_length:
                starts.append(i)
                ends.append(j - 1)
            i = j
        else:
            i += 1
    return starts, ends


def normal_scores(values):
    out = np.full(len(values), np.nan)
    ok = np.isfinite(values)
    if ok.sum() > 2:
        out[ok] = stats.norm.ppf((stats.rankdata(values[ok]) - .5) / ok.sum())
    return out


def divergence(values, min_valid):
    """Mean pairwise absolute difference across performances at each quarter."""
    v = np.array(list(values.values()))
    out = np.full(v.shape[1], np.nan)
    for q in range(v.shape[1]):
        x = v[:, q][np.isfinite(v[:, q])]
        if len(x) >= min_valid:
            x = np.sort(x)
            m = len(x)
            out[q] = 2 * np.sum((2 * np.arange(1, m + 1) - m - 1) * x) / (m * (m - 1))
    return out


def centered(values):
    """Remove each performance's own level so only local deviations remain."""
    return {k: v - np.nanmedian(v) if np.isfinite(v).any() else v for k, v in values.items()}


def residualize(z, covariates):
    ok = np.isfinite(z) & np.all(np.isfinite(covariates), axis=1)
    out = np.full(len(z), np.nan)
    a = np.column_stack([np.ones(ok.sum()), covariates[ok]])
    beta = np.linalg.lstsq(a, z[ok], rcond=None)[0]
    out[ok] = z[ok] - a @ beta
    return out


def circular_effect(values, label, min_count=8, guard=.05):
    """Observed in-minus-out mean and a circular-shift p value that keeps autocorrelation."""
    ok = np.isfinite(values)
    v, l = values[ok], label[ok].astype(float)
    n, k = len(v), l.sum()
    if k < min_count or n - k < min_count:
        return np.nan, np.nan
    v = v - v.mean()
    sums = np.real(np.fft.ifft(np.fft.fft(v) * np.conj(np.fft.fft(l))))
    effects = sums / k - (-sums) / (n - k)
    obs = effects[0]
    shifts = np.arange(n)
    null = effects[(shifts >= guard * n) & (shifts <= (1 - guard) * n)]
    p = (1 + np.sum(np.abs(null) >= abs(obs))) / (1 + len(null))
    return obs, p


def moving_mean(x, w):
    ok = np.isfinite(x)
    num = np.convolve(np.where(ok, x, 0), np.ones(w), 'valid')
    den = np.convolve(ok.astype(float), np.ones(w), 'valid')
    return np.where(den >= w / 2, num / np.maximum(den, 1), np.nan)


def genre(track):
    for name, pattern in GENRES:
        if re.search(pattern, track):
            return name
    return 'other'


def analyze_piece(piece, group, score_notes, alignment, seconds_per_quarter, args):
    grid = group['grid']
    n = len(grid) - 1
    roll, onsets = pitch_roll(score_notes, grid)
    strength, member, principal, plain = recurrence(roll, args.window, args.threshold)
    starts, ends = run_edges(member, args.window)
    entry, exit_ = zone(n, starts, 4), zone(n, ends, 4)
    nov = zone(n, novelty_boundaries(plain), 4)
    edge = np.zeros(n, bool)
    edge[:16] = True
    edge_end = np.zeros(n, bool)
    edge_end[-16:] = True
    slow = group['common'].get('rubato', np.full(n, np.nan))
    peak = group['common'].get('dynamics', np.full(n, np.nan))
    labels = dict(theme_inner=member & ~entry & ~exit_, theme_entry=entry, theme_exit=exit_,
                  nonrecurring=~member & ~entry & ~exit_, principal=principal, novelty=nov,
                  opening=edge, ending=edge_end,
                  common_slow=slow <= np.nanquantile(slow, .1), common_peak=peak >= np.nanquantile(peak, .9))
    min_valid = max(args.min_performances - 1, 4)
    raw = {c: divergence(centered(group['values'][c]) if c != 'rubato' else group['values'][c], min_valid)
           for c in CHANNELS}
    z = {c: normal_scores(np.log(raw[c] + 1e-6)) for c in CHANNELS}
    z['composite'] = normal_scores(np.nanmean([z[c] for c in STYLE_CHANNELS], axis=0))
    covariates = np.column_stack([np.log1p(onsets), alignment])
    rows = []
    position = np.arange(n) / max(n - 1, 1)
    joint = []
    for model, names in JOINT_MODELS.items():
        x = np.column_stack([labels[k].astype(float) for k in names] + [position, covariates])
        for c, values in z.items():
            ok = np.isfinite(values) & np.all(np.isfinite(x), axis=1)
            a = np.column_stack([np.ones(ok.sum()), x[ok]])
            if ok.sum() < 64 or np.linalg.matrix_rank(a) < a.shape[1]:
                continue
            beta = np.linalg.lstsq(a, values[ok], rcond=None)[0]
            r2 = 1 - np.var(values[ok] - a @ beta) / np.var(values[ok])
            base = np.linalg.lstsq(a[:, [0, -2, -1]], values[ok], rcond=None)[0]
            r2_base = 1 - np.var(values[ok] - a[:, [0, -2, -1]] @ base) / np.var(values[ok])
            joint += [dict(piece=piece, model=model, channel=c, term=t, coef=float(b), r2=r2, r2_covariates=r2_base)
                      for t, b in zip(list(names) + ['position'], beta[1:len(names) + 2])]
    split = split_half_slow(group, covariates, min_valid)
    if split:
        rows.append(dict(piece=piece, channel='composite', category='common_slow_split', kind='adjusted',
                         effect=split[0], p=split[1], share=.1))
    for c, values in z.items():
        resid = residualize(values, covariates)
        for name, label in labels.items():
            for adjusted, v in (('raw', values), ('adjusted', resid)):
                eff, p = circular_effect(v, label)
                rows.append(dict(piece=piece, channel=c, category=name, kind=adjusted,
                                 effect=eff, p=p, share=float(label.mean())))
    w = max(int(round(args.preview_seconds / seconds_per_quarter)), 8)
    smooth = moving_mean(residualize(z['composite'], covariates), w) if n > w + 8 else np.zeros(0)
    window = None
    if len(smooth) and np.isfinite(smooth).any():
        best = int(np.nanargmax(smooth))
        frac = {name: moving_mean(label.astype(float), w) for name, label in labels.items()}
        window = dict(piece=piece, window_quarters=w, start=best, start_fraction=best / n,
                      in_first_window=best < w,
                      **{f'obs_{k}': float(v[best]) for k, v in frac.items()},
                      **{f'exp_{k}': float(np.nanmean(v)) for k, v in frac.items()})
    smooth_t = moving_mean(strength, 8)
    smooth_d = moving_mean(z['composite'], 8)
    ok = np.isfinite(smooth_t) & np.isfinite(smooth_d)
    corr = dict(piece=piece, rho_recurrence=stats.spearmanr(smooth_t[ok], smooth_d[ok])[0] if ok.sum() > 16 else np.nan)
    for a, b in (('rubato', 'dynamics'), ('rubato', 'articulation'), ('dynamics', 'articulation')):
        x, y = moving_mean(z[a], 8), moving_mean(z[b], 8)
        ok = np.isfinite(x) & np.isfinite(y)
        corr[f'rho_{a}_{b}'] = stats.spearmanr(x[ok], y[ok])[0] if ok.sum() > 16 else np.nan
    trace = dict(z=z, labels=labels, strength=strength, onsets=onsets, alignment=alignment, window=window)
    return rows + joint, window, corr, trace, dict(quarters=n, theme_share=float(member.mean()),
                                         principal_share=float(principal.mean()), runs=len(starts))


def split_half_slow(group, covariates, min_valid, seed=0):
    """Locate common slowing in one half of the performances, measure divergence in the other."""
    keys = sorted(group['values']['rubato'])
    if len(keys) < 10:
        return None
    rng = np.random.default_rng(seed)
    order = rng.permutation(keys)
    a, b = order[:len(keys) // 2], order[len(keys) // 2:]
    # Standardized = (raw - common) / scale with one scale per work, so this ranks like half-A's own median.
    slow_a = group['common']['rubato'] / group['scales']['rubato'] + np.nanmedian(
        [group['values']['rubato'][k] for k in a], axis=0)
    label = slow_a <= np.nanquantile(slow_a, .1)
    parts = [normal_scores(np.log(divergence(
        {k: v for k, v in (centered(group['values'][c]) if c != 'rubato' else group['values'][c]).items() if k in b},
        min(min_valid, len(b) - 1)) + 1e-6)) for c in STYLE_CHANNELS]
    resid = residualize(normal_scores(np.nanmean(parts, axis=0)), covariates)
    return circular_effect(resid, label & np.isfinite(slow_a))


def alignment_confidence(keys, raw_dir, grid, score_notes):
    """Matched score-onset share per quarter, averaged over performances."""
    n = len(grid) - 1
    total = np.zeros(n)
    for note in score_notes:
        q = int(np.floor(note.onset_beats - grid[0]))
        if 0 <= q < n:
            total[q] += 1
    shares = []
    for key in keys:
        with np.load(raw_dir / f'{key}.npz', allow_pickle=False) as z:
            onsets = z['match_score_onsets']
        matched = np.zeros(n)
        q = np.floor(onsets - grid[0]).astype(int)
        np.add.at(matched, q[(q >= 0) & (q < n)], 1)
        shares.append(np.where(total > 0, np.minimum(matched / np.maximum(total, 1), 1), np.nan))
    out = np.nanmean(shares, axis=0)
    return np.where(np.isfinite(out), out, np.nanmedian(out))


def seconds_per_quarter(keys, raw_dir):
    values = []
    for key in keys:
        with np.load(raw_dir / f'{key}.npz', allow_pickle=False) as z:
            values.append(np.median(np.diff(z['performance_beats'])))
    return float(np.median(values))


def summarize(rows):
    df = pd.DataFrame(rows).dropna(subset=['effect'])
    out = []
    for (channel, category, kind), g in df.groupby(['channel', 'category', 'kind']):
        e = g.effect.to_numpy()
        w = stats.wilcoxon(e) if len(e) >= 10 and np.any(e != 0) else None
        out.append(dict(channel=channel, category=category, kind=kind, pieces=len(g),
                        median_effect=float(np.median(e)), mean_effect=float(np.mean(e)),
                        positive_share=float(np.mean(e > 0)),
                        sig_positive_share=float(np.mean((g.p < .05) & (e > 0))),
                        sig_negative_share=float(np.mean((g.p < .05) & (e < 0))),
                        wilcoxon_p=float(w.pvalue) if w else np.nan,
                        binom_sig_positive_p=float(stats.binomtest(int(np.sum((g.p < .05) & (e > 0))), len(g), .025,
                                                                   alternative='greater').pvalue)))
    return pd.DataFrame(out)


def figures(out, summary, effects, windows, traces, meta):
    setup_style()
    adj = summary[(summary.kind == 'adjusted') & (summary.channel == 'composite')].set_index('category')
    effects_adj = effects[(effects.kind == 'adjusted') & (effects.channel == 'composite')]
    order = [c for c in CATEGORIES if c in adj.index]

    fig, ax = chart('구조 위치별 연주 간 차이 (작품별 효과 분포)',
                    f'{effects_adj.piece.nunique()}개 작품, 종합 차이(정규점수)의 구간 안-밖 평균 차, 음 밀도·정렬 신뢰도 보정', height=6.6)
    for y, c in enumerate(order):
        e = effects_adj[effects_adj.category == c].effect.dropna()
        jitter = np.random.default_rng(y).uniform(-.18, .18, len(e))
        color = GRAY if c.startswith('common') else BLUE
        ax.scatter(e, y + jitter, s=9, alpha=.35, color=color, linewidths=0)
        ax.plot([e.median()] * 2, [y - .3, y + .3], color=ORANGE, lw=2.5)
    ax.axvline(0, color='#333', lw=1)
    ax.set_yticks(range(len(order)), [CATEGORIES[c] for c in order])
    ax.invert_yaxis()
    ax.set(xlabel='해당 구간의 연주 간 차이 − 나머지 구간 (정규점수)', ylabel='')
    save(fig, out / '01_category_effects.png', '주황 선=작품 중앙값. 회색 범주는 연주 공통 패턴에서 정의한 참고용 위치.')

    fig, ax = chart('피처별로 차이가 커지는 위치', '작품 중앙값 효과(보정 후). 파랑=큼, 빨강=작음', height=6.6)
    chans = ['composite', 'rubato', 'dynamics', 'articulation', 'pedal_depth']
    s = summary[summary.kind == 'adjusted'].pivot(index='category', columns='channel', values='median_effect').loc[order, chans]
    lim = np.nanmax(np.abs(s.to_numpy()))
    ax.imshow(s.to_numpy(), cmap='RdBu', vmin=-lim, vmax=lim, aspect='auto')
    for (i, j), v in np.ndenumerate(s.to_numpy()):
        ax.text(j, i, f'{v:+.2f}', ha='center', va='center', fontsize=10)
    ax.set_xticks(range(len(chans)), [CHANNEL_NAMES[c] for c in chans])
    ax.set_yticks(range(len(order)), [CATEGORIES[c] for c in order])
    ax.grid(False)
    save(fig, out / '02_channel_heatmap.png', '값=해당 구간 안-밖 정규점수 차의 작품 중앙값.')

    keys = ['opening', 'principal', 'theme_inner', 'theme_entry', 'theme_exit', 'nonrecurring', 'novelty', 'ending']
    obs = [windows[f'obs_{k}'].mean() for k in keys]
    exp = [windows[f'exp_{k}'].mean() for k in keys]
    fig, ax = chart('차이 최대 30초 구간은 어디에 걸치는가',
                    f'{len(windows)}개 작품, 최대 구간의 구조 포함 비율 vs 무작위 위치 기대값')
    x = np.arange(len(keys))
    ax.bar(x - .2, exp, .4, color=GRAY, label='무작위 위치 기대')
    ax.bar(x + .2, obs, .4, color=BLUE, label='차이 최대 구간')
    ax.set_xticks(x, [CATEGORIES[k] for k in keys], rotation=25, ha='right')
    ax.set(xlabel='', ylabel='구간 안 포함 비율 (작품 평균)')
    ax.legend()
    save(fig, out / '03_top_window_overlap.png', '30초 길이는 작품별 연주 중앙 템포로 마디 수 환산.', bottom=.1)

    piece = max(traces, key=lambda p: (meta[p]['performances'] >= 10, meta[p]['principal_share'] > .1,
                                       -abs(meta[p]['quarters'] - 400)))
    t = traces[piece]
    fig, ax = chart(f'예시: {meta[piece]["track"][:60]}', f'{meta[piece]["composer"]}, 연주 {meta[piece]["performances"]}개. 음영=구조 위치', height=5.4)
    q = np.arange(len(t['z']['composite']))
    for name, color in (('principal', GREEN), ('nonrecurring', '#cccccc')):
        lab = t['labels'][name]
        ax.fill_between(q, -3, 3, where=lab, color=color, alpha=.18, step='post', label=CATEGORIES[name])
    for name, color in (('theme_entry', ORANGE),):
        ax.fill_between(q, -3, 3, where=t['labels'][name], color=color, alpha=.25, step='post', label=CATEGORIES[name])
    ax.plot(q, t['z']['composite'], color=GRAY, lw=.6, alpha=.6)
    ax.plot(q[3:-4], moving_mean(t['z']['composite'], 8), color=BLUE, lw=2, label='종합 차이 (8박 이동평균)')
    if t['window']:
        s0 = t['window']['start']
        ax.axvspan(s0, s0 + t['window']['window_quarters'], color=BLUE, alpha=.1, label='차이 최대 30초')
    ax.set(xlabel='악보 위치 (4분음표)', ylabel='연주 간 차이 (정규점수)', ylim=(-3, 3))
    ax.legend(fontsize=8, ncol=3, loc='lower left')
    save(fig, out / '04_example_piece.png', '정규점수는 작품 안 순위 기반. 회색 선=4분음표 단위 원값.')
    return piece


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    raw_dir = args.root / 'features/dual_dtw/raw'
    audit = pd.read_csv(args.audit, dtype={'perf_id': str, 'composition_id': str}).fillna('')
    audit = audit[audit.status == 'aligned'].set_index('perf_id')
    groups = load_groups(args.cache)
    rows, windows, corrs, traces, meta = [], [], [], {}, {}
    for piece, group in sorted(groups.items()):
        keys = sorted(group['values']['rubato'])
        if len(keys) < args.min_performances:
            continue
        row = audit.loc[keys[0]]
        notes = load_score_midi(resolve_source(row['local_score'], args.root))
        alignment = alignment_confidence(keys, raw_dir, group['grid'], notes)
        r, w, c, trace, info = analyze_piece(piece, group, notes, alignment, seconds_per_quarter(keys, raw_dir), args)
        meta[piece] = dict(piece=piece, track=row['track'], composer=row['composer'], genre=genre(row['track']),
                           performances=len(keys), **info)
        rows += r
        corrs.append(c)
        traces[piece] = trace
        if w:
            windows.append(w)
        print(f'{piece} n={len(keys)} quarters={info["quarters"]} theme={info["theme_share"]:.2f}', flush=True)
    joint = pd.DataFrame([r for r in rows if 'term' in r])
    rows = [r for r in rows if 'term' not in r]
    joint_summary = pd.DataFrame([
        dict(model=m, channel=c, term=t, pieces=len(g), median_coef=float(g.coef.median()),
             positive_share=float((g.coef > 0).mean()), wilcoxon_p=float(stats.wilcoxon(g.coef).pvalue),
             median_r2=float(g.r2.median()), median_r2_covariates=float(g.r2_covariates.median()))
        for (m, c, t), g in joint.groupby(['model', 'channel', 'term'])])
    joint.to_csv(args.out / 'joint_coefficients.csv', index=False, encoding='utf-8-sig')
    joint_summary.to_csv(args.out / 'joint_summary.csv', index=False, encoding='utf-8-sig')
    effects = pd.DataFrame(rows).merge(pd.DataFrame(meta.values())[['piece', 'genre', 'composer']], on='piece')
    summary = summarize(rows)
    by_genre = pd.concat([summarize(g.to_dict('records')).assign(genre=name)
                          for name, g in effects[(effects.channel == 'composite') & (effects.kind == 'adjusted')].groupby('genre')
                          if g.piece.nunique() >= 8])
    windows = pd.DataFrame(windows)
    corrs = pd.DataFrame(corrs)
    effects.to_csv(args.out / 'piece_effects.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(args.out / 'summary.csv', index=False, encoding='utf-8-sig')
    by_genre.to_csv(args.out / 'summary_by_genre.csv', index=False, encoding='utf-8-sig')
    windows.to_csv(args.out / 'top_windows.csv', index=False, encoding='utf-8-sig')
    corrs.to_csv(args.out / 'correlations.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(meta.values()).to_csv(args.out / 'pieces.csv', index=False, encoding='utf-8-sig')
    example = figures(args.out, summary, effects, windows, traces, meta)
    stats_out = dict(pieces=len(meta), performances=int(sum(m['performances'] for m in meta.values())),
                     window=args.window, threshold=args.threshold, example=example,
                     median_theme_share=float(np.median([m['theme_share'] for m in meta.values()])),
                     median_principal_share=float(np.median([m['principal_share'] for m in meta.values()])),
                     rho_recurrence_median=float(corrs.rho_recurrence.median()),
                     rho_channels_median={c: float(corrs[c].median()) for c in corrs if c.startswith('rho_') and c != 'rho_recurrence'},
                     top_window_in_first=float(windows.in_first_window.mean()) if len(windows) else None)
    (args.out / 'stats.json').write_text(json.dumps(stats_out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(stats_out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
