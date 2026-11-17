"""Draw the implemented final protected-G pipeline for the GitHub report."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def draw(out):
    available = {font.name for font in font_manager.fontManager.ttflist}
    family = next((name for name in ('Malgun Gothic', 'Noto Sans CJK KR', 'NanumGothic')
                   if name in available), 'DejaVu Sans')
    plt.rcParams.update({'font.family': family, 'axes.unicode_minus': False,
                         'svg.fonttype': 'path'})
    fig, ax = plt.subplots(figsize=(17, 15))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    ink, muted, line = '#253448', '#526176', '#A9B5C4'

    def label(x, y, text, size=13, color=ink, weight='normal', **kwargs):
        ax.text(x, y, text, fontsize=size, color=color, fontweight=weight,
                ha='center', va='center', linespacing=1.55, **kwargs)

    def box(x, y, w, h, text='', fill='#F3F6FA', edge='#D6DEE8', size=13,
            color=ink, weight='normal'):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
            boxstyle='round,pad=0.006,rounding_size=0.012',
            facecolor=fill, edgecolor=edge, linewidth=1.25))
        if text:
            label(x+w/2, y+h/2, text, size, color, weight)

    def arrow(a, b, color=line):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=16,
                                    linewidth=1.5, color=color))

    label(.5, .969, '공통 Note AE → 시대별 보호 G 학습', 25, weight='bold')
    label(.5, .938, '최종 배포 모델 구조  |  대표구간 전체 사용 · 시대별 독립 공간 · 64차원 임베딩', 13, muted)

    box(.035, .72, .93, .184, fill='#F7F9FC')
    label(.066, .882, '01', 14, '#246186', 'bold')
    label(.36, .882, '전체 5,082개 대표구간으로 공통 AE 복원 사전학습', 15, weight='bold')
    box(.055, .787, .255, .065,
        'Performance MIDI + MusicXML\n음표별 R11 · 관측 mask · 악보 조건', '#FFFFFF', size=12)
    box(.354, .787, .324, .065,
        '조건부 BiGRU AE\nhidden 64 · 2 layers · window z 64D', '#EAF2F8', '#A9C3D8', 13, weight='bold')
    box(.722, .787, .222, .065,
        '음표별 R11 복원\n관측 mask + 영역 균형 Huber', '#FFFFFF', size=12)
    arrow((.316, .82), (.346, .82))
    arrow((.685, .82), (.714, .82))
    label(.5, .759, '대표구간 전체 → 최대 128음표 윈도 + tail 유지 → encoder → z → 조건부 decoder', 12, muted)
    label(.5, .735, '평가용 train / validation / test로 설정 선택 후, 전체 5,082개 공통 모델 재학습', 11, muted)

    box(.37, .644, .26, .041, '공통 AE 가중치를 세 번 복사', '#EAF2F8', '#A9C3D8', 14, weight='bold')
    arrow((.5, .714), (.5, .691))
    ax.plot([.185, .815], [.617, .617], color=line, lw=1.5)
    ax.plot([.5, .5], [.638, .617], color=line, lw=1.5)

    eras = [
        (.055, '바로크', '103작품·악장  /  558구간', '2', '437', '121', '#EAF4F0', '#95BBAA', '#27694D'),
        (.37, '고전 + 고전–낭만 전환', '132작품·악장  /  2,303구간', '3', '2,101', '202', '#EEF0FA', '#A4AED0', '#45558D'),
        (.685, '낭만·후기 낭만 + 인상주의', '244작품·악장  /  2,221구간', '4', '1,362', '859', '#FBF2E7', '#D1B38D', '#8C5F2F'),
    ]
    for x, name, count, epoch, direct, anchor, fill, edge, accent in eras:
        center = x+.13
        arrow((center, .617), (center, .589))
        box(x, .28, .26, .302, fill='#FFFFFF', edge=edge)
        box(x+.007, .529, .246, .045, name, fill, edge, 13, accent, 'bold')
        label(center, .508, count, 11, muted)
        box(x+.014, .413, .232, .067,
            '시대별 performance encoder + head\n윈도 z 가중평균 → z_ae (64D)\nresidual head → z_style → L2 → e', fill, edge, 11.5)
        arrow((center, .499), (center, .487), edge)
        arrow((center, .406), (center, .384), edge)
        box(x+.014, .316, .232, .061,
            '같은 시대 · 서로 다른 작품의 G 비교\n보호 G loss + z anchor + head 억제', '#F8FAFC', '#D6DEE8', 11)
        label(center, .297, f'최종 {epoch} epoch  |  직접 G {direct} · anchor만 {anchor}', 10, muted)

    box(.055, .079, .89, .162, fill='#F3F6FA')
    label(.5, .217, '최종 추가 학습: 직접 복원 loss 없이 G + 보존항', 15, weight='bold')
    label(.5, .184, 'L = λ_effective × (0.5 L_pair + 0.5 L_rank) + 0.05 L_z_anchor + 0.01 L_head_shift', 14)
    label(.5, .153, 'G 거리 차이 · 관측 신뢰도 반영  |  윈도 z에서 G gradient ≤ 가중 전 복원 gradient의 25%', 11.5, muted)
    label(.5, .127, '악보 encoder와 decoder 전용 가중치 유지  |  head 마지막 층 0 초기화 · residual 조정 폭 제한', 11.5, muted)
    label(.5, .101, '복원·채널 오차 · z 변화 · 임베딩 분산·effective rank 검사 후 체크포인트 선택', 11.5, muted)

    label(.5, .048, '전체 5,082개: encoder·anchor·임베딩 생성  |  직접 G 3,900개 · anchor만 1,182개', 12, weight='bold')
    label(.5, .025, '검색: 같은 시대의 e 사이 코사인 거리만 사용  ·  전체 자료 최종 모델의 진단은 독립 test가 아님', 11, muted)
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'svg'):
        fig.savefig(out/f'model_architecture.{suffix}', dpi=150, facecolor='white')
    svg = out/'model_architecture.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',
                   encoding='utf-8')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1]/'analysis'/'era_g_loss_report')
    draw(parser.parse_args().out)
