"""Report the validation sweep, reloaded full-data fits and both AXY selections."""
import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def vectors(path):
    with np.load(path, allow_pickle=False) as data:
        e = data['e'].astype(np.float64)
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        return dict(zip(data['perf_ids'].tolist(), e))


ERA_NAMES = {'baroque': '바로크', 'classical_transition': '고전+전환',
             'romantic_impressionist': '낭만+인상주의'}


def fixed_triplet(case, embedding):
    ids = [case[k].split(':')[1] for k in ('anchor', 'near', 'far')]
    a, b, c = [embedding[pid] for pid in ids]
    near, far = float(1-a@b), float(1-a@c)
    return {'old_role_B_distance': near, 'old_role_C_distance': far, 'C_minus_B': far-near}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment
    old = read(args.baseline/'final_report.json')
    new = read(root/'production/final_report.json')
    old_pack = read(root/'axy_baseline_25/audit/listening.json')
    new_pack = read(root/'axy_relaxed/audit/listening.json')
    old_e = vectors(old['embeddings'])
    new_e = vectors(new['embeddings'])
    if len(old_e) != 5082 or old_e.keys() != new_e.keys():
        raise ValueError('The two runs must have the same verified clip identities')
    lines = [
        '# G 기울기 상한 비교와 AXY 청취 실험', '',
        '공통 AE에서 시작해 기존 25% 상한을 50%와 100%로 완화했다. '
        '명목 G 가중치 .1과 3 epoch 램프, 보존항, margin 및 confidence 규칙을 유지했다.', '',
        '기존 25% 대조군과 같은 train 초기값, 데이터 분할, 배치, seed, 학습 횟수로 '
        'validation에서 비교했다. 서로 다른 배치에서 실제 G 가중치가 달라지므로 마지막 '
        'epoch의 평균과 병목 z의 평균 기울기 비율을 함께 표시한다.', '',
        '| 시대 | 상한 | G score ↓ | 거리 MSE ↓ | G 순서 정확도 ↑ | 실제 λ | z 기울기 비율 | 공통 대비 복원 변화 | 보존 검사 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    selections = {}
    for era, name in ERA_NAMES.items():
        selection = read(root/'validation'/era/'selection.json')
        selections[era] = selection
        for trial in selection['trials'].values():
            g = trial['validation']['geometry']
            label = ' (선택)' if trial['budget'] == selection['selected_budget'] else ''
            status = '통과' if trial['guard']['passed'] else '실패'
            lines.append(f"| {name} | {trial['budget']:.0%}{label} | {g['score']:.5f} | "
                         f"{g['pair_mse']:.5f} | {g['ranking_accuracy']:.2%} | "
                         f"{trial['effective_lambda']:.5f} | {trial['mean_z_gradient_ratio']:.1%} | "
                         f"{trial['reconstruction_change_pct']:+.3f}% | {status} |")
    lines += ['', '시대별로 보존 기준을 통과한 50%·100% 후보 중 마지막 validation G score가 '
              '더 낮은 설정을 청취 실험에 사용했다. 단일 seed 비교다.', '',
              '## 전체 5,082개로 재학습한 모델', '',
              '아래는 전체 자료에 다시 적합한 두 모델의 진단이다. 독립 test 성능으로 해석할 수 없다.', '',
              '| 시대 | 기존 → 새 상한 | 기존 → 새 epoch | 거리 MSE | G 순서 정확도 | 공통 대비 새 복원 변화 |',
              '|---|---:|---:|---:|---:|---:|']
    for era, name in ERA_NAMES.items():
        before, after = old['eras'][era], new['eras'][era]
        m0, m1 = (r['production_training_diagnostics']['geometry'] for r in (before, after))
        delta = 100*(after['production_training_diagnostics']['reconstruction']['loss'] /
                     after['baseline']['reconstruction']['loss']-1)
        lines.append(f"| {name} | 25% → {after['config']['gradient_budget']:.0%} | "
                     f"{before['production_epoch']} → {after['production_epoch']} | "
                     f"{m0['pair_mse']:.5f} → {m1['pair_mse']:.5f} | "
                     f"{m0['ranking_accuracy']:.2%} → {m1['ranking_accuracy']:.2%} | {delta:+.3f}% |")
    lines += ['', '## 청취', '', '- [기존 25% AXY](http://localhost:8772/)',
              '- [상한 완화 AXY](http://localhost:8771/)', '',
              '두 버전 모두 같은 시대에서 임베딩상 가장 가까운 다른 작품의 B를 고르고, '
              'B와 같은 작품에서 A와 가장 먼 C를 고른다. 거리 차이가 큰 3/3/4문항을 선택하고 '
              '동일한 YouTube 대표구간 매핑을 사용한다. 후보 선택에 G 유효성·문항 유형·'
              '연주자 조건을 추가하지 않았다. X/Y 순서는 각 버전에서 무작위다.', '']
    replacement = new_pack['selection'].get('replacement')
    if replacement:
        previous = read(root/'axy_relaxed/audit/archive'/replacement['from_pack_id']/'listening.json')
        labels = ', '.join(c['label'] for c in previous['cases'] if c['id'] in replacement['replaced_case_ids'])
        lines += [f'상한 완화판의 {labels} 문항은 기존 문항을 제외하고 나머지 문항과 '
                  '작품·영상이 겹치지 않는 후보 중 거리 차이가 가장 큰 조합으로 교체했다.', '']
    lines += ['아래 표는 같은 A/B/C에 두 모델을 적용해 거리 차이(C−B)를 비교한 값이다. '
              '음원까지 새로 고르는 두 청취 버전의 비교에는 문항 구성의 변화도 포함된다.', '',
              '| 고정 문항 | 선정에 사용한 모델 | 기존 25% 거리 차이 | 상한 완화 거리 차이 |',
              '|---|---|---:|---:|']
    fixed = []
    for source, pack in [('기존 25%',old_pack),('완화',new_pack)]:
        for case in pack['cases']:
            before, after = fixed_triplet(case, old_e), fixed_triplet(case, new_e)
            fixed.append({'selected_by': source, 'case': case['id'], 'old':before, 'new':after})
            lines.append(f"| {case['label']} | {source} | {before['C_minus_B']:.4f} | {after['C_minus_B']:.4f} |")
    shared = len({(c['anchor'],c['near'],c['far']) for c in old_pack['cases']} &
                 {(c['anchor'],c['near'],c['far']) for c in new_pack['cases']})
    lines += ['', f'두 버전 전체에서 A/B/C가 모두 같은 문항은 {shared}개였다. '
              '청각적으로 차이가 얼마나 또렷해졌는지는 실제 응답으로 확인한다.', '']
    (root/'comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    (root/'comparison.json').write_text(json.dumps({
        'validation':selections, 'baseline_pack':old_pack['id'], 'relaxed_pack':new_pack['id'],
        'shared_triplets':shared, 'fixed_triplet_distances':fixed,
    }, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
