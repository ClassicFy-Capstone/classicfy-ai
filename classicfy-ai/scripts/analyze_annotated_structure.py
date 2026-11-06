"""Re-test style divergence against expert form and cadence annotations.

Run from any directory: python analyze_annotated_structure.py --help.
BPSD (Beethoven first movements) gives sonata-form sections in measures;
DCML Mozart sonatas give cadences and phrase marks in quarter beats. Both are
mapped onto each ATEPP score grid by aligning score note events, so repeats
that one edition writes out and the other folds are handled by gaps.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import stats

import analyze_structure_divergence as base
from atepp_feature_analysis import resolve_source
from validate_tempo import setup_style, chart, save, BLUE, ORANGE, GRAY, GREEN
from preprocessing.score_midi import load_score_midi

HERE = Path(__file__).resolve().parent
REGIONS = dict(Introduction='서주', FirstGroup='제1주제군', Transition='경과', SecondGroup='제2주제군',
               CadentialGroup='종결군', Development='발전부', Coda='코다')
CADENCES = ('PAC', 'IAC', 'HC', 'EC', 'DC')
ROMAN = dict(I=1, II=2, III=3)


def parse_args():
    data = HERE.parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=data / 'ATEPP_dataset')
    p.add_argument('--bpsd', type=Path, default=data / 'annotations/bpsd/Beethoven_Piano_Sonata_Dataset_v2/2_Annotations')
    p.add_argument('--dcml', type=Path, default=data / 'annotations/dcml/mozart_piano_sonatas')
    p.add_argument('--audit', type=Path, default=HERE.parent / 'analysis/atepp_research/midi_audit.csv')
    p.add_argument('--catalog', type=Path, default=HERE.parent / 'analysis/atepp_normalization/performance_catalog.csv')
    p.add_argument('--out', type=Path, default=HERE.parent / 'analysis/structure_divergence/annotated')
    p.add_argument('--min-match', type=float, default=.6)
    return p.parse_args()


def events(onsets, pitches):
    """Group notes into onset events with a 128-pitch membership vector."""
    order = np.argsort(onsets, kind='stable')
    times, rows = [], []
    for t, p in zip(np.round(np.asarray(onsets)[order], 4), np.asarray(pitches)[order]):
        if not times or t != times[-1]:
            times.append(t)
            rows.append(np.zeros(128, bool))
        rows[-1][int(p)] = True
    return np.array(times), np.array(rows)


def needleman_wunsch(a, b, gap=.5, free_ends=False):
    """Global (or b-free-end) alignment of event sets; returns matched (i, j, jaccard)."""
    n, m = len(a), len(b)
    bi = b.astype(np.int32)
    sizes = bi.sum(1)
    d = np.zeros((n + 1, m + 1), np.float32)
    d[0] = 0 if free_ends else gap * np.arange(m + 1)
    d[:, 0] = gap * np.arange(n + 1)
    steps = gap * np.arange(m + 1)
    for i in range(1, n + 1):
        inter = bi @ a[i - 1].astype(np.int32)
        cost = 1 - inter / np.maximum(sizes + a[i - 1].sum() - inter, 1)
        e = np.minimum(d[i - 1, :-1] + cost, d[i - 1, 1:] + gap)
        f = np.concatenate([[d[i, 0]], e]) - steps
        d[i] = np.minimum.accumulate(f) + steps
    i, j = n, (int(np.argmin(d[n])) if free_ends else m)
    pairs = []
    while i > 0 and j > 0:
        inter = (a[i - 1] & b[j - 1]).sum()
        c = 1 - inter / max((a[i - 1] | b[j - 1]).sum(), 1)
        if np.isclose(d[i, j], d[i - 1, j - 1] + c):
            pairs.append((i - 1, j - 1, 1 - c))
            i, j = i - 1, j - 1
        elif np.isclose(d[i, j], d[i - 1, j] + gap):
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def align_positions(at_times, at_rows, ref_times, ref_rows, ref_pos):
    """Reference position for each ATEPP event; long unmatched ATEPP blocks get a second, free-end pass."""
    pairs = needleman_wunsch(at_rows, ref_rows)
    pos = np.full(len(at_times), np.nan)
    score = np.zeros(len(at_times))
    for i, j, s in pairs:
        if s >= .5:
            pos[i], score[i] = ref_pos[j], s
    gaps = np.flatnonzero(np.isnan(pos))
    for block in np.split(gaps, np.flatnonzero(np.diff(gaps) > 1) + 1):
        if len(block) < 24:
            continue
        for i, j, s in needleman_wunsch(at_rows[block], ref_rows, free_ends=True):
            if s >= .5:
                pos[block[i]], score[block[i]] = ref_pos[j], s
    return pos, float(np.mean(score >= .5))


def quarter_positions(at_times, pos, grid):
    """Median reference position per ATEPP quarter, interpolated only between nearby matches."""
    n = len(grid) - 1
    q = np.floor(at_times - grid[0]).astype(int)
    out = np.full(n, np.nan)
    for k in range(n):
        v = pos[(q == k) & np.isfinite(pos)]
        if len(v):
            out[k] = np.median(v)
    ok = np.flatnonzero(np.isfinite(out))
    for a, b in zip(ok[:-1], ok[1:]):
        if 1 < b - a <= 4 and out[b] >= out[a]:
            out[a + 1:b] = np.interp(np.arange(a + 1, b), [a, b], [out[a], out[b]])
    return out


def bpsd_name(track):
    m = re.search(r'Op\.\s*(\d+)\s*,?\s*(?:No\.\s*(\d+))?', track)
    return f'Beethoven_Op{int(m.group(1)):03d}' + (f'No{m.group(2)}' if m.group(2) else '') + '-01'


def dcml_name(track):
    k = re.search(r'K\.\s*(\d+)', track)
    mv = re.search(r':\s*(?:(\d)\.|(I{1,3})\.)', track)
    if not k or not mv:
        return None
    return f'K{k.group(1)}-{int(mv.group(1)) if mv.group(1) else ROMAN[mv.group(2)]}'


def bpsd_labels(path_dir, name, at_times, at_rows, grid):
    notes = pd.read_csv(path_dir / 'ann_score_note' / f'{name}.csv', sep=';')
    notes = notes[notes.grace == 0]
    ref_times, ref_rows = events(notes.start_meas.to_numpy(), notes.pitch.to_numpy())
    pos, match = align_positions(at_times, at_rows, ref_times, ref_rows, ref_times)
    meas = quarter_positions(at_times, pos, grid)
    fine = pd.read_csv(path_dir / 'ann_score_structureFine' / f'{name}.csv', sep=';')
    label = np.full(len(meas), '', object)
    for s, e, lab in fine.itertuples(index=False):
        label[(meas >= s) & (meas < e)] = lab
    return label, match


def dcml_labels(root, name, at_times, at_rows, grid):
    notes = pd.read_csv(root / 'notes' / f'{name}.notes.tsv', sep='\t')
    grace = notes['gracenote'].notna() if 'gracenote' in notes else False
    notes = notes[~grace & notes.quarterbeats.notna()]
    qb = notes.quarterbeats.map(lambda v: float(eval(str(v))) if '/' in str(v) else float(v)).to_numpy()
    ref_times, ref_rows = events(qb, notes.midi.to_numpy())
    pos, match = align_positions(at_times, at_rows, ref_times, ref_rows, ref_times)
    qpos = quarter_positions(at_times, pos, grid)
    h = pd.read_csv(root / 'harmonies' / f'{name}.harmonies.tsv', sep='\t')
    h = h[h.quarterbeats.notna()].reset_index(drop=True)
    h['qb'] = h.quarterbeats.map(lambda v: float(eval(str(v))) if '/' in str(v) else float(v))
    n = len(qpos)
    marks = {c: np.zeros(n, bool) for c in CADENCES + ('phrase_end', 'phrase_start')}
    points = [(c, h.qb[h.cadence == c].to_numpy()) for c in CADENCES]
    points += [('phrase_end', h.qb[h.phraseend.astype(str).str.contains('}')].to_numpy()),
               ('phrase_start', h.qb[h.phraseend.astype(str).str.contains('{')].to_numpy())]
    for c, at in points:
        for t in at:
            # The arrival plus the approach to it: three quarters before through one after.
            marks[c] |= (qpos >= t - 3) & (qpos < t + 2)
    return marks, match, np.isfinite(qpos)


def piece_inputs(group, keys, audit, raw_dir, args):
    row = audit.loc[keys[0]]
    notes = load_score_midi(resolve_source(row['local_score'], args.root))
    grid = group['grid']
    at_times, at_rows = events([x.onset_beats for x in notes], [x.pitch for x in notes])
    roll, onsets = base.pitch_roll(notes, grid)
    alignment = base.alignment_confidence(keys, raw_dir, grid, notes)
    raw = {c: base.divergence(base.centered(group['values'][c]) if c != 'rubato' else group['values'][c], 4)
           for c in base.CHANNELS}
    z = {c: base.normal_scores(np.log(raw[c] + 1e-6)) for c in base.CHANNELS}
    z['composite'] = base.normal_scores(np.nanmean([z[c] for c in base.STYLE_CHANNELS], axis=0))
    covariates = np.column_stack([np.log1p(onsets), alignment])
    resid = {c: base.residualize(v, covariates) for c, v in z.items()}
    _, member, _, _ = base.recurrence(roll, 16, .9)
    slow = group['common']['rubato']
    return dict(notes=notes, grid=grid, at_times=at_times, at_rows=at_rows, z=z, resid=resid,
                covariates=covariates, member=member, common_slow=slow <= np.nanquantile(slow, .1))


def effect_rows(piece, resid, labels, valid):
    rows = []
    for c, v in resid.items():
        v = np.where(valid, v, np.nan)
        for name, lab in labels.items():
            eff, p = base.circular_effect(v, lab & valid)
            rows.append(dict(piece=piece, channel=c, category=name, effect=eff, p=p, share=float(lab[valid].mean())))
    return rows


def summarize(rows, by=('channel', 'category')):
    df = pd.DataFrame(rows).dropna(subset=['effect'])
    out = []
    for key, g in df.groupby(list(by)):
        e = g.effect.to_numpy()
        out.append(dict(zip(by, key), pieces=len(g), median_effect=float(np.median(e)),
                        positive_share=float(np.mean(e > 0)),
                        wilcoxon_p=float(stats.wilcoxon(e).pvalue) if len(e) >= 6 and np.any(e) else np.nan))
    return pd.DataFrame(out)


def boundaries(label, valid, radius=4):
    change = [i for i in range(1, len(label)) if valid[i] and valid[i - 1] and label[i] != label[i - 1]]
    return base.zone(len(label), change, radius), change


def beethoven(args, groups, audit, catalog, raw_dir):
    rows, joint, overlap, quality, traces = [], [], [], [], {}
    for piece, track in catalog.items():
        if 'Beethoven' not in track[1] or not re.search(r':\s*(?:I\.|1\.)\s', track[0]):
            continue
        name = bpsd_name(track[0])
        if not (args.bpsd / 'ann_score_structureFine' / f'{name}.csv').exists() or name == 'Beethoven_Op026-01':
            continue
        group = groups[piece]
        keys = sorted(group['values']['rubato'])
        x = piece_inputs(group, keys, audit, raw_dir, args)
        label, match = bpsd_labels(args.bpsd, name, x['at_times'], x['at_rows'], x['grid'])
        valid = label != ''
        quality.append(dict(piece=piece, bpsd=name, performances=len(keys), event_match=match,
                            labelled_share=float(valid.mean())))
        if match < args.min_match or valid.mean() < .8:
            continue
        region = np.array([l.split(':')[-1] for l in label], object)
        bound, change = boundaries(label, valid)
        theme = np.isin(region, ['FirstGroup', 'SecondGroup'])
        theme_edge = np.zeros(len(label), bool)
        for i in change:
            if theme[i] != theme[i - 1] or (theme[i] and region[i] != region[i - 1]):
                theme_edge |= base.zone(len(label), [i], 4)
        labels = {f'region_{r}': (region == r) & ~bound for r in REGIONS}
        labels.update(theme_inner=theme & ~bound, theme_boundary=theme_edge, other_boundary=bound & ~theme_edge,
                      any_boundary=bound, auto_repeat=x['member'], common_slow=x['common_slow'])
        rows += effect_rows(piece, x['resid'], labels, valid)
        # Joint model: regions (Transition = reference), boundary, position, covariates.
        names = [r for r in REGIONS if r != 'Transition' and ((region == r) & valid).sum() >= 8]
        position = np.arange(len(label)) / max(len(label) - 1, 1)
        design = np.column_stack([(region == r).astype(float) for r in names] + [bound, position, x['covariates']])
        for c, v in x['z'].items():
            ok = valid & np.isfinite(v) & np.all(np.isfinite(design), axis=1)
            a = np.column_stack([np.ones(ok.sum()), design[ok]])
            beta = np.linalg.lstsq(a, v[ok], rcond=None)[0]
            joint += [dict(piece=piece, channel=c, term=t, coef=float(b))
                      for t, b in zip(names + ['boundary', 'position'], beta[1:len(names) + 3])]
        m, t = x['member'][valid], theme[valid]
        slow = x['common_slow'][valid]
        overlap.append(dict(piece=piece, bpsd=name, auto_in_theme=float(t[m].mean()) if m.any() else np.nan,
                            theme_share=float(t.mean()), theme_detected=float(m[t].mean()) if t.any() else np.nan,
                            slow_in_boundary=float(bound[valid][slow].mean()), boundary_share=float(bound[valid].mean()),
                            slow_in_theme_inner=float((theme & ~bound)[valid][slow].mean()),
                            theme_inner_share=float((theme & ~bound)[valid].mean())))
        traces[piece] = dict(z=x['z']['composite'], region=region, valid=valid, track=track[0], n=len(keys))
    return rows, joint, overlap, quality, traces


def mozart(args, groups, audit, catalog, raw_dir):
    rows, overlap, quality = [], [], []
    for piece, track in catalog.items():
        if 'Mozart' not in track[1]:
            continue
        name = dcml_name(track[0])
        if not name or not (args.dcml / 'notes' / f'{name}.notes.tsv').exists():
            continue
        group = groups[piece]
        keys = sorted(group['values']['rubato'])
        x = piece_inputs(group, keys, audit, raw_dir, args)
        marks, match, valid = dcml_labels(args.dcml, name, x['at_times'], x['at_rows'], x['grid'])
        quality.append(dict(piece=piece, dcml=name, performances=len(keys), event_match=match,
                            labelled_share=float(valid.mean())))
        if match < args.min_match or valid.mean() < .8:
            continue
        cad = np.any([marks[c] for c in CADENCES], axis=0)
        labels = dict(cadence_any=cad, **{f'cadence_{c}': marks[c] for c in CADENCES},
                      phrase_end=marks['phrase_end'], phrase_start=marks['phrase_start'] & ~cad,
                      common_slow=x['common_slow'], auto_repeat=x['member'])
        rows += effect_rows(piece, x['resid'], labels, valid)
        slow = x['common_slow'] & valid
        pend = marks['phrase_end'] | cad
        overlap.append(dict(piece=piece, dcml=name, slow_in_cadence=float(cad[slow].mean()),
                            cadence_share=float(cad[valid].mean()),
                            slow_in_phrase_end=float(pend[slow].mean()), phrase_end_share=float(pend[valid].mean()),
                            cadence_slow_share=float(slow[cad & valid].mean()) if (cad & valid).any() else np.nan))
    return rows, overlap, quality


def figures(out, b_sum, b_joint, m_sum, traces):
    setup_style()
    order = [f'region_{r}' for r in REGIONS] + ['theme_inner', 'theme_boundary', 'other_boundary', 'auto_repeat', 'common_slow']
    names = {**{f'region_{k}': v for k, v in REGIONS.items()}, 'theme_inner': '주제군 내부(경계 제외)',
             'theme_boundary': '주제군 시작·끝 ±1마디', 'other_boundary': '그 밖의 형식 경계 ±1마디',
             'auto_repeat': '(이전) 자동 반복 재료', 'common_slow': '공통 감속 지점'}
    s = b_sum[(b_sum.channel == 'composite')].set_index('category')
    order = [c for c in order if c in s.index]
    fig, ax = chart('베토벤 1악장: 형식 구간별 연주 간 차이', f'BPSD 전문가 형식 주석, 작품 {int(s.pieces.max())}개. 막대=작품 중앙값', height=6.4)
    y = np.arange(len(order))
    colors = [GRAY if c in ('auto_repeat', 'common_slow') else BLUE for c in order]
    ax.barh(y, s.loc[order, 'median_effect'], color=colors)
    for i, c in enumerate(order):
        v = s.loc[c]
        ax.text(max(v.median_effect, 0) + .01, i, f' {v.positive_share:.0%} (n={int(v.pieces)})', va='center', fontsize=9)
    ax.axvline(0, color='#333', lw=1)
    ax.set_yticks(y, [names[c] for c in order])
    ax.invert_yaxis()
    ax.set(xlabel='구간 안 − 밖 연주 간 차이 (정규점수, 밀도·정렬 보정)', ylabel='')
    save(fig, out / '01_beethoven_regions.png', '괄호=차이가 더 큰 작품 비율. 구간 범주는 경계 ±1마디를 제외한 내부.')

    cats = ['cadence_any'] + [f'cadence_{c}' for c in CADENCES] + ['phrase_end', 'phrase_start', 'common_slow', 'auto_repeat']
    labels = dict(cadence_any='종지 전체', cadence_PAC='완전정격종지(PAC)', cadence_IAC='불완전정격종지(IAC)',
                  cadence_HC='반종지(HC)', cadence_EC='회피종지(EC)', cadence_DC='위종지(DC)',
                  phrase_end='프레이즈 끝', phrase_start='프레이즈 시작', common_slow='공통 감속 지점',
                  auto_repeat='(이전) 자동 반복 재료')
    s = m_sum[m_sum.channel == 'composite'].set_index('category')
    cats = [c for c in cats if c in s.index]
    fig, ax = chart('모차르트 소나타: 종지·프레이즈 위치의 연주 간 차이', f'DCML 전문가 종지 주석, 작품 {int(s.pieces.max())}개. 도착 3박 전~1박 후', height=6.0)
    y = np.arange(len(cats))
    ax.barh(y, s.loc[cats, 'median_effect'], color=[GRAY if c in ('common_slow', 'auto_repeat') else ORANGE for c in cats])
    for i, c in enumerate(cats):
        v = s.loc[c]
        ax.text(max(v.median_effect, 0) + .01, i, f' {v.positive_share:.0%} (n={int(v.pieces)})', va='center', fontsize=9)
    ax.axvline(0, color='#333', lw=1)
    ax.set_yticks(y, [labels[c] for c in cats])
    ax.invert_yaxis()
    ax.set(xlabel='구간 안 − 밖 연주 간 차이 (정규점수, 밀도·정렬 보정)', ylabel='')
    save(fig, out / '02_mozart_cadences.png', '괄호=차이가 더 큰 작품 비율.')

    piece = max(traces, key=lambda p: traces[p]['n'])
    t = traces[piece]
    fig, ax = chart(f'예시: {t["track"][:58]}', f'연주 {t["n"]}개. 음영=BPSD 형식 구간', height=5.4)
    q = np.arange(len(t['z']))
    palette = dict(FirstGroup=GREEN, SecondGroup=BLUE, Transition='#cccccc', CadentialGroup=ORANGE,
                   Development='#b48ead', Coda='#888888', Introduction='#e6d36b')
    for r, color in palette.items():
        if (t['region'] == r).any():
            ax.fill_between(q, -3, 3, where=t['region'] == r, color=color, alpha=.18, step='post', label=REGIONS[r])
    ax.plot(q[3:-4], base.moving_mean(t['z'], 8), color='#222', lw=1.6, label='종합 차이 (8박 이동평균)')
    ax.set(xlabel='악보 위치 (4분음표)', ylabel='연주 간 차이 (정규점수)', ylim=(-3, 3))
    ax.legend(fontsize=8, ncol=4, loc='lower left')
    save(fig, out / '03_beethoven_example.png', '형식 구간은 BPSD 주석을 악보 음표 정렬로 옮긴 것.')
    return piece


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    raw_dir = args.root / 'features/dual_dtw/raw'
    audit = pd.read_csv(args.audit, dtype={'perf_id': str, 'composition_id': str}).fillna('')
    audit = audit[audit.status == 'aligned'].set_index('perf_id')
    cat = pd.read_csv(args.catalog, dtype={'key': str})
    groups = base.load_groups(args.root / 'features/dual_dtw/atepp_standardized.npz')
    catalog = {p: (g.track.iloc[0], g.composer.iloc[0]) for p, g in cat.groupby('piece')
               if p in groups and len(groups[p]['values']['rubato']) >= 5}
    b_rows, b_joint, b_overlap, b_quality, traces = beethoven(args, groups, audit, catalog, raw_dir)
    m_rows, m_overlap, m_quality = mozart(args, groups, audit, catalog, raw_dir)
    b_sum, m_sum = summarize(b_rows), summarize(m_rows)
    j = pd.DataFrame(b_joint)
    b_jsum = summarize([dict(channel=r['channel'], category=r['term'], effect=r['coef']) for r in b_joint])
    for name, df in dict(beethoven_effects=pd.DataFrame(b_rows), beethoven_summary=b_sum, beethoven_joint=j,
                         beethoven_joint_summary=b_jsum, beethoven_overlap=pd.DataFrame(b_overlap),
                         beethoven_alignment=pd.DataFrame(b_quality), mozart_effects=pd.DataFrame(m_rows),
                         mozart_summary=m_sum, mozart_overlap=pd.DataFrame(m_overlap),
                         mozart_alignment=pd.DataFrame(m_quality)).items():
        df.to_csv(args.out / f'{name}.csv', index=False, encoding='utf-8-sig')
    example = figures(args.out, b_sum, b_jsum, m_sum, traces)
    bo, mo = pd.DataFrame(b_overlap), pd.DataFrame(m_overlap)
    result = dict(beethoven_pieces=len(bo), beethoven_aligned=len(b_quality), mozart_pieces=len(mo), mozart_aligned=len(m_quality),
                  example=example,
                  beethoven_overlap=bo.drop(columns=['piece', 'bpsd']).mean().round(3).to_dict(),
                  mozart_overlap=mo.drop(columns=['piece', 'dcml']).mean().round(3).to_dict())
    (args.out / 'stats.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
