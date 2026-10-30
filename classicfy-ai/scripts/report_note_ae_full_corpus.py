"""Publish training status or the completed full-corpus AE evaluation."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import time


def latest_epoch(path):
    if not path.exists() or not path.stat().st_size:
        return None
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 8 * 1024 * 1024))
        lines = stream.read().splitlines()
    return json.loads(lines[-1])


def publish(run, out):
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((run / 'protocol.json').read_text())
    summary_path = run / 'summary.json'
    counts = protocol['splits']
    lines = ['# 검증 통과 ATEPP 전체 데이터의 복원 AE', '',
             '검증을 통과한 254작품·2,801연주를 사용한다. 원본 ATEPP 전체가 아니다.', '',
             '| Split | 작품 | 연주 | 8마디 구간 |', '|---|---:|---:|---:|']
    for split in ('train', 'validation', 'test'):
        row = counts[split]
        lines.append(f"| {split} | {row['works']:,} | {row['recordings']:,} | {row['windows']:,} |")
    lines += ['', '작품·악보 SHA·동일 MIDI hash의 split 간 중복은 0개다. R/S scaler는 train에서만 계산했다.', '',
              '기존 ScoreEncoder/AE와 5그룹 Huber 복원 loss를 유지한다. Triplet·평균 보조 head·초기 tempo 조건은 사용하지 않는다. '
              '처음부터 학습하며, 최대 100 epoch / validation patience 15 / min_delta 0.0001이다. '
              '매 epoch 학습 51,638개 구간을 한 번씩 사용한다. 배치 수 제한은 없다. '
              'GPU 배치 128과 길이별 묶음으로 padding을 줄였다.', '']
    if not summary_path.exists():
        row = latest_epoch(run / 'training/epochs.jsonl')
        progress = run / 'training/progress.json'
        status = json.loads(progress.read_text()) if progress.exists() else {'stage': 'initial_train_evaluation'}
        lines += ['**상태: 학습 진행 중. 최종 test 성능은 아직 평가하지 않았다.**', '']
        if row:
            lines += [f"완료 epoch: {row['epoch']}. 전체 train 평가 loss: {row['full_train_evaluation']['loss']:.6f}, "
                      f"validation loss: {row['validation']['loss']:.6f}. 이 값은 최종 선택 모델의 평가 결과가 아니다.", '']
        if status.get('windows_used'):
            lines += [f"최근 학습 진행 기록: epoch {status['epoch']}, "
                      f"{status['windows_used']:,}/{status['windows_expected']:,}구간.", '']
        (out / 'progress.json').write_text(json.dumps({'completed_epoch': row['epoch'] if row else 0,
                                                    'current': status}, indent=2))
    else:
        summary = json.loads(summary_path.read_text())
        lines += [f"**완료: {summary['completed_epoch']} epoch. 선택 모델: epoch {summary['selected_epoch']}.** "
                  f"종료 사유: `{summary['stop_reason']}`. 모든 epoch의 전체 구간 사용을 확인했다.", '',
                  '| Split | Train 평균 baseline | AE | 오차 감소율 |', '|---|---:|---:|---:|']
        for split in ('train', 'validation', 'test'):
            result = summary['results'][split]
            base, value = result['train_mean']['loss'], result['model']['loss']
            lines.append(f'| {split} | {base:.6f} | {value:.6f} | {(1-value/base)*100:+.2f}% |')
        lines += ['', '동일 mask·beat당 한 번·그룹 및 구간 균등 평균으로 계산했다. '
                  'Baseline은 train 채널 평균을 고정 출력한다.', '',
                  '| Feature | Test 평균 baseline | Test AE | 오차 감소율 |', '|---|---:|---:|---:|']
        test = summary['results']['test']
        for group in ('tempo', 'onset', 'velocity', 'articulation', 'pedal'):
            base, value = [test[k]['groups'][group]['loss'] for k in ('train_mean', 'model')]
            lines.append(f'| {group} | {base:.6f} | {value:.6f} | {(1-value/base)*100:+.2f}% |')
        lines += ['', '| 실제 단위 MAE | Test 평균 baseline | Test AE |', '|---|---:|---:|']
        for label, key in [('Tempo 상대 오차 (%)', 'tempo_relative_error_percent'),
                           ('Onset 편차 (ms)', 'onset_deviation_error_ms'),
                           ('Velocity (MIDI)', 'velocity_error_MIDI'),
                           ('Articulation 상대 오차 (%)', 'articulation_relative_error_percent')]:
            base, value = [test[k]['native_MAE'][key] for k in ('train_mean', 'model')]
            lines.append(f'| {label} | {base:.4f} | {value:.4f} |')
        lines += ['', '![Train and validation learning curves](learning_curves.png)', '']
        for name in ('summary.json', 'validation_selection.json'):
            shutil.copyfile(run / name, out / name)
        history = [json.loads(line) for line in (run / 'training/epochs.jsonl').open()]
        rows = [{'epoch': r['epoch'], 'updates': r['step'], 'train_online': r['reconstruction']['loss'],
                 'train_eval': r['full_train_evaluation']['loss'], 'validation': r['validation']['loss'],
                 'complete_train_epoch': r['complete_train_epoch']} for r in history]
        with (out / 'curves.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        fig, axis = plt.subplots(figsize=(9, 4.8))
        axis.plot([r['epoch'] for r in rows], [r['train_eval'] for r in rows], label='Full train (eval)')
        axis.plot([r['epoch'] for r in rows], [r['validation'] for r in rows], label='Full validation')
        for split, color in [('train', 'C0'), ('validation', 'C1')]:
            axis.axhline(summary['results'][split]['train_mean']['loss'], color=color, linestyle=':',
                         label=f'{split.title()} mean baseline')
        axis.axvline(summary['selected_epoch'], color='gray', linestyle='--', label='Selected epoch')
        axis.set(xlabel='Complete training epochs', ylabel='Equal-group / equal-window Huber loss')
        axis.legend()
        axis.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(out / 'learning_curves.png', dpi=180)
        plt.close(fig)
    lines += ['7작품 pilot과는 split·scaler·배치 크기도 달라 데이터 증가만의 효과를 분리한 실험은 아니다. '
              '한 seed와 자동 정렬 데이터에 대한 결과다. 이전 pilot에서 일부 작품을 이미 확인했으므로 '
              '완전히 미열람 상태의 독립 benchmark로 해석하지 않는다.', '',
              'Validation 복원 loss로 체크포인트를 선택한 뒤 train·validation·test를 평가한다. '
              'Test로 설정을 변경하지 않는다. 원본 로그와 체크포인트는 '
              '`workspace/diagnostics/full_corpus_reconstruction/run01/`에 보관한다.', '']
    (out / 'README.md').write_text('\n'.join(lines))
    shutil.copyfile(run / 'protocol.json', out / 'protocol.json')
    return summary_path.exists()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        if publish(args.run, args.out) or not args.watch:
            break
        log = args.run / 'execution.log'
        if log.exists() and 'Traceback (most recent call last)' in log.read_text():
            raise RuntimeError('Training failed; inspect execution.log')
        time.sleep(30)


if __name__ == '__main__':
    main()
