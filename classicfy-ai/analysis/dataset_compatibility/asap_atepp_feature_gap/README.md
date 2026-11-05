# ATEPP–ASAP 특징 차이와 4분음표 시간축 통일

2026-10-28 KST. 현재 보유 자료를 이용한 진단과 별도 전처리 결과다. 기존 학습 cache·checkpoint·저장소 코드는 변경하지 않았다.

**시간 단위는 통일할 수 있고, ASAP의 정렬된 연주 1,035개를 실제로 변환했다. 같은 단위로 비교해도 데이터셋 특징 차이는 남았다.** 다만 두 데이터셋의 서로 다른 연주자·악보·정렬·녹음/전사 조건이 함께 다르므로 차이를 전사 오류량으로 해석하지 않는다.

## 비교 범위와 단위

- 이전 작품 대응표의 `same_unit`만 사용했다. 악장을 합치거나 나눈 두 사례는 제외했다. 슈만 Toccata 반복판과 ATEPP의 같은 작품에 대한 중복 ID는 같은 작품으로 묶었다.
- ATEPP는 로컬에 원본 MIDI·DualDTW 정렬 cache가 없다. `atepp_normalization/feature_normalization/performance_summary.csv`의 저장된 품질 승인 결과와 `performance_catalog.csv`를 사용했다. 이 결과는 이미 4분음표 grid에서 계산된 것이다.
- ATEPP 결과가 있는 공통 134작품에서 ASAP 496연주를 원본 MIDI에서 재추출하고 ATEPP 1,510연주의 요약과 대응시켰다. 이 숫자는 전체 데이터셋이 아니라 이번 비교에 사용할 수 있는 분석 모집단이다.
- 특징별 분포 비교는 양쪽 데이터셋에 연주가 2개 이상 있는 작품만 사용한다. 특징별로 80–91작품이다. 작품당 연주 수가 많은 베토벤 등이 결과를 지배하지 않도록 작품을 같은 가중치로 집계했다.
- ASAP은 정렬된 연주를 사용하고 Articulation은 nASAP robust=True인 note 정렬만 사용했다. ATEPP는 기존 DualDTW 품질 승인 정책을 따른다. 동일 MIDI hash 중복은 제거했다.
- CC64 이벤트가 없는 연주는 이번 진단에서 양쪽 모두 페달 결측으로 처리했다. 기존 ASAP 기본 extractor의 0 처리를 그대로 섞지 않았다.

## 같은 단위에서 관측된 원시 특징 차이

각 연주의 유효 beat 평균 → 작품 내 연주 중앙값 → 작품 간 중앙값이다. 마지막 열은 작품마다 계산한 ATEPP−ASAP 중앙값 차이를 다시 중앙값으로 요약한 것으로, 앞 두 대표값을 뺀 값과 같을 필요가 없다.

| 특징 | 비교 작품 | ASAP 대표값 | ATEPP 대표값 | 작품 내 차이의 중앙값 |
|---|---:|---:|---:|---:|
| Tempo (log₂, 120-BPM quarter reference) | 91 | -0.2153 | -0.2504 | -0.0575 |
| Rubato (log₂, 자기 Tempo 중앙값 제거) | 91 | -0.0253 | -0.0306 | -0.0001 |
| Dynamics (velocity/127) | 91 | 0.4905 | 0.5082 | +0.0146 |
| Articulation (log₂ 길이 비율) | 80 | -0.6521 | -0.4503 | +0.1453 |
| 페달 깊이 (0–1) | 90 | 0.3745 | 0.4784 | +0.0635 |
| 페달 on 비율 (0–1) | 90 | 0.3410 | 0.5395 | +0.1048 |
| 페달 on/off 전환 / 4분음표 | 90 | 0.6216 | 1.6117 | +0.9365 |

특히 페달 전환 수와 깊이, Articulation 분포 차이가 컸다. 전환 수 대표값은 같은 4분음표 기준에서도 약 0.62 대 1.61이었다. 전체 평균의 단순 비교가 아니라 작품을 맞춰 본 값이다.

관측된 CC64 값 종류의 연주별 중앙값은 ASAP 110종, ATEPP 67종이다(CC64가 있는 연주 기준). 이번 비교 모집단에서는 ASAP 12/496연주에 CC64가 없고 ATEPP 1,510연주에는 모두 CC64가 있었다. 따라서 이번 결과를 'ATEPP에는 페달이 없어서 생긴 차이'라고 설명할 수는 없다. 값 종류 차이는 관측 통계이며 센서 정밀도나 전사 오차를 직접 측정한 값이 아니다.

## 공통 패턴 제거·표준화 후에도 구분되는가

진단용 ASAP quarter 특징은 같은 악보/정렬 폴더의 정확한 quarter 위치 교집합에서 reference를 만들었다. support>=2인 위치에서 common 중앙값을 제거하고 Tempo/Rubato/Dynamics/Articulation은 MAD, 세 Pedaling 채널은 SD로 나눴다. ATEPP는 같은 채널별 정책의 저장 결과를 사용했다. 원시값·표준화값의 평균과 p95−p5를 사용한 간단한 L2 logistic classifier로 데이터 출처를 예측했다.

- 작품을 5개 fold로 나누어 학습에 없는 작품에서 출처 구분을 평가했다.
- 각 작품·데이터셋의 학습 가중치를 같게 하고, classifier의 수치 표준화는 학습 fold에서만 fit했다. 같은 작품의 모든 연주는 같은 fold에 있다.
- 같은 특징 묶음의 원시/표준화 비교는 정확히 같은 연주들로 수행했다.
- AUC는 작품별로 계산한 후 같은 가중치로 평균했다. 0.5는 무작위 수준이고, 1은 완전 구분이다. 아래 수치는 정확도(accuracy)가 아니다.

| 사용 특징 | 전처리 | 작품 수 | ASAP / ATEPP 연주 | 출처 구분 AUC | 작품 bootstrap 95% 구간 |
|---|---|---:|---:|---:|---:|
| Dynamics+Articulation | 원시 특징 | 80 | 367 / 843 | 0.795 | 0.745–0.843 |
| Dynamics+Articulation | 공통 제거+표준화 | 80 | 367 / 843 | 0.762 | 0.704–0.817 |
| 7채널 전체 | 원시 특징 | 79 | 355 / 839 | 0.938 | 0.897–0.972 |
| 7채널 전체 | 공통 제거+표준화 | 79 | 355 / 839 | 0.769 | 0.716–0.824 |

7채널 전체의 고정 라벨 섞기 대조 AUC는 원시 0.490, 표준화 후 0.523이었다. 실제 출처 구분 결과는 단순히 classifier가 아무 라벨이나 구분해 보이는 현상으로 설명되지 않는다.

**해석:** 정규화로 차이가 줄지만 남는다. Dynamics+Articulation만 사용해도 표준화 후 AUC가 약 0.762이므로 페달만 제거하면 두 데이터가 같아진다고 볼 근거도 없다.

**해석의 한계:** AUC는 오토인코더 성능·추천 성능·전사 오류율이 아니다. 연주자 집단, 녹음·전사 장비, 악보 판본, 자동/수동 정렬, 비교 구간, reference 집단 크기 차이가 함께 반영된다. 특히 표준화 reference는 현재 파이프라인처럼 데이터셋별 해당 작품 cohort를 사용하며, heldout 작품의 reference도 그 작품의 연주들로 계산한다. classifier 가중치의 작품 분리는 지켰지만, reference 없는 새 연주 하나에 대한 독립 추론을 검증한 것은 아니다. ATEPP 원본이 없어 같은 cohort 크기·같은 구간으로 양쪽 정규화를 다시 fit하는 통제도 수행하지 못했다. Tempo/Rubato의 ASAP 진단 cache는 통일한 quarter 기준의 일반적인 common 제거이며, 기존 native-beat 모델의 suspicious-status 정책과 훈련 모집단을 완전히 재현한 cache는 아니다.

전사 자체의 영향을 분리하려면 동일한 ASAP 녹음을 ATEPP와 같은 전사기로 변환한 MIDI와 원래 센서 MIDI를 같은 악보·정렬·구간에서 비교하는 대조가 필요하다.

## 시간축 변환을 실제로 수행한 결과

전체 ASAP 1,067경로 중 정렬 annotation이 있는 1,036개에서 완전 동일 MIDI 복제 1개를 제외한 **1,035개 모두 변환 성공**했다. 정렬 annotation이 없는 31개는 이번 작업으로 변환하지 않았다.

| 항목 | 결과 |
|---|---:|
| 기존 native beat 구간 | 553,924 |
| 4분음표 1개와 같은 구간 | 380,059 (68.6%) |
| 더 짧은 구간 | 57,724 (10.4%) |
| 더 긴 구간 | 116,141 (21.0%) |
| 새 4분음표 구간 | 611,258 |
| 반복·특수 처리로 mask한 quarter 구간 | 4,581 |
| 같은 악보/grid의 reference로 정규화한 연주 | 968 |
| 현재 WindowDataset에서 유효한 64-quarter 구간이 있는 연주 | 767 |
| 위 조건에서 생성 가능한 64-quarter windows | 12,889 |

변환 절차:

1. 악보 MIDI의 ticks-per-quarter와 tempo-change map으로 annotation의 악보 초를 quarter-note 좌표로 역변환했다. 120 BPM을 모든 악보에 일괄 가정하지 않았다. 초 단위 반올림 때문에 정수 quarter에서 2e-5 이내인 좌표만 정수로 보정했다.
2. annotation이 덮는 구간 안에서 정수 quarter 경계를 만들었다. 그 위치의 실제 연주 시각은 기존 악보 위치–연주 시각 anchor 사이에서 piecewise-linear하게 보간했다. 양 끝 범위를 넘겨 외삽하지 않았다.
3. Dynamics·Pedaling·Articulation을 새 경계에 맞춰 원본 MIDI 이벤트와 note 정렬에서 다시 계산했다. 기존 feature 곡선의 값을 보간하지 않았다. Articulation의 기대 음 길이는 기존 note tempo-map 정의를 유지했다.
4. Tempo는 quarter당 실제 시간으로 계산하며 ATEPP와 같은 120 BPM 기준을 사용했다. 원래 연주의 초 단위 타이밍은 유지했다.
5. bR 특수 anchor에 닿는 source 구간과 겹치는 quarter bin을 결측 처리했다. 떨어진 구간을 삭제해서 이웃으로 붙이지 않았다. 비증가 anchor는 거부하는 정책이며 이번 전체 변환에서는 거부 사례가 없었다.

새 quarter 경계 612,293개 중 222,693개(36.4%)는 기존 annotation 사이에서 추정한 시각이다. 이는 데이터에 없던 실제 quarter-onset 정답을 복원했다는 뜻이 아니다. 긴 annotation beat 내부의 rubato를 정확히 알 수는 없으며, 더 정밀한 변환은 note alignment의 onset map과 별도 대조해야 한다. 단위 통일은 악보 판본·반복 구조·정렬 오류·전사 편향을 해결하지 않는다.

## 산출물과 적용 범위

- `all_asap_quarter_raw.npz`: 정렬된 전체 1,035연주의 quarter grid, 연주 경계 시각, 7채널 raw 값과 mask, source hash.
- `all_asap_quarter_normalized.npz`: reference를 만들 수 있는 968연주의 `[7,T]` 값과 mask. mask=False는 값 0이며 별도 boolean으로 구분된다. q는 원래 quarter 좌표를 보존한다.
- `asap_quarter_raw.npz`, `asap_quarter_normalized.npz`: 비교에 사용한 공통 작품 subset.
- `asap_quarter_summary.csv`, `matched_atepp_summary.csv`: 특징별 원시/표준화 요약.
- `per_work_differences.csv`, `feature_gap_summary.csv`: 작품별/특징별 분포 차이.
- `domain_separability.csv`, `domain_predictions.csv`: 출처 구분 결과와 heldout 예측.
- `domain_separability_label_shuffle.csv`: 고정 라벨 섞기 대조.
- `all_asap_time_axis_audit.csv`: 전체 연주의 원래 beat 길이, 추정 quarter 경계 수, 특수 mask, MIDI/악보 hash.
- `verification.json`: 독립 검산 및 기존 WindowDataset 입력 검증.
- `feature_gap_summary.png/.svg/.pdf`, `time_axis_conversion.png/.svg/.pdf`: 공유 가능한 요약 그림.

기존 학습 CLI에는 이 cache를 자동 연결하지 않았다. 이 자료는 단위가 달라진 새 입력이므로 기존 native-beat checkpoint를 검증 없이 그대로 적용하지 않는다. ATEPP 학습용 시계열을 실제로 준비해 같은 quarter 규격의 학습·ASAP 평가를 수행하는 단계는 남아 있다.

## 검증과 재현

- synthetic MIDI의 tempo-change 역변환, 2-quarter subdivision, bR mask, 범위 외삽 금지, 비증가 anchor 거부를 검사했다.
- 전체 1,035개 raw 입력에서 q 간격 1, 시각 증가, mask/값 일관성, source hash 중복 없음과 Tempo 수식을 검사했다.
- 서로 다른 단위를 포함한 3연주·36구간의 Dynamics와 Pedal depth를 원본 note/CC64 이벤트에서 독립 계산했다. 최대 오차는 Dynamics 0, Pedal depth 약 8.66e-15였다.
- 정규화 cache의 모든 값을 기존 `WindowDataset`으로 실제 읽었다. 유효 window 12,889개와 767개 연주를 확인했다.
- 출처 AUC는 저장 예측의 모든 ASAP–ATEPP 연주 쌍을 직접 비교해 독립 재계산했다. 작품 fold 분리와 원시/표준화의 동일 모집단을 확인했다.

```bash
.venv/bin/python classicfy-ai/analysis/dataset_compatibility/asap_atepp_feature_gap/verify_quarter_grid.py
.venv/bin/python classicfy-ai/analysis/dataset_compatibility/asap_atepp_feature_gap/analyze.py --workers 4
.venv/bin/python classicfy-ai/analysis/dataset_compatibility/asap_atepp_feature_gap/convert_all_asap.py
.venv/bin/python classicfy-ai/analysis/dataset_compatibility/asap_atepp_feature_gap/verify_results.py
MPLCONFIGDIR=/tmp/classicfy-matplotlib .venv/bin/python classicfy-ai/analysis/dataset_compatibility/asap_atepp_feature_gap/plot_results.py
```

`analyze.py --reuse-extraction`은 첫 MIDI 재추출을 재사용한다. 원본 출처·원본 파일 hash는 summary와 개별 time-axis audit에 있다. 비교는 원본 전체 데이터셋이 아닌 현재 분석 가능한 subset이라는 점을 유지한다.
