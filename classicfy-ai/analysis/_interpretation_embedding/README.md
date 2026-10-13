# 같은 작품의 연주 차이와 다른 작품의 유사 특징 검색

현재 다섯 feature를 요약한 임베딩으로 **연주별 차이를 표현하고 다른 작품에서 특징상 가까운 연주를 검색할 수 있는지** 실제 사례를 확인한다.
기존 [원본 연주 복구 실험](../_embedding_validation/README.md)과는 다른 질문이다.
여기서는 기존 14차원 요약 벡터와 거리 함수를 그대로 사용한다. 학습된 신경망 임베딩이나 청취 평가 결과는 아니다.
PNG 파일 하나에 그래프 하나씩 담았다.

## 먼저 볼 그림 세 장

1. [같은 작품의 3연주 profile](same_work/01_profile.png): 악보가 같아도 연주별 성향이 다르다.
2. [다른 작품의 A·B·C profile](cross_work/01_profile.png): 자기 작품 안의 상대적 경향을 비교한다.
3. [B와 같은 작품의 후보 거리](cross_work/04_target_candidates.png): 검색이 실제로 어떤 후보를 골랐는지 확인한다.

## ① 같은 작품에서도 연주별 특징이 다르다

작품은 **Bach · Fugue · BWV 848**다. 해당 작품/grid의 유효 후보 전체로 공통값과 scale을 계산했다.
그중 기준 A와 Tempo 평균 순위 25%·75% 근처의 연주를 골랐다. 중복되면 중앙/끝 순위를 사용한다.
선택한 파일은 MiyashitaM01M, Lou01M, SunY01M다.
이 세 연주만으로 공통값을 다시 계산한 것이 아니다.

profile은 전체 유효 공통 beat의 요약이며, 곡선은 중앙 93~124구간이다.
차이가 가장 큰 구간을 찾아 자르지 않았다. 가로축은 같은 score beat **구간 번호**이며 마디 번호가 아니다.
각 곡선의 점선은 작품 내 위치별 공통 중앙값이다. 원래 단위로 표시해 수치를 이해하기 쉽게 했다.
검색에는 공통값을 제거하고 scale로 나눈 값의 요약을 사용한다.

| 그림 | 쉽게 읽는 방법 |
|---|---|
| [Tempo](same_work/02_tempo_curve.png) | 같은 위치에서 위에 있는 연주가 빠르다. 100%는 score MIDI 기준 속도다. |
| [Rubato](same_work/03_rubato_curve.png) | 각 연주 자신의 전체 빠르기를 제거했다. 0 위는 자기 평소보다 빠르고, 아래는 느리다. |
| [Dynamics](same_work/04_dynamics_curve.png) | 같은 위치에서 velocity가 크면 건반을 더 강하게 친 기록이다. |
| [Articulation](same_work/05_articulation_curve.png) | 해당 위치 tempo로 예상한 음 길이에 비해 건반을 유지한 비율이다. |
| [페달 깊이](same_work/06_pedal_depth_curve.png) | 높은 값은 CC64를 깊게 유지한 시간이 많다는 뜻이다. |
| [페달 on 비율](same_work/07_pedal_down_ratio_curve.png) | CC64 ≥ 64인 시간이 beat 안에서 차지하는 비율이다. |
| [페달 전환](same_work/08_pedal_changes_curve.png) | beat 안에서 on/off 경계를 넘은 횟수다. |

같은 beat의 곡선이 다르고, 공통값을 빼고 요약한 profile도 서로 다르다.
따라서 이 사례에서는 **“추출한 특징으로 연주별 차이를 표현할 수 있다”**고 설명할 수 있다.
작품 내 중앙 패턴은 음악적 정답이 아니라 이 데이터의 연주들로 만든 기준이다.

동일한 세 연주의 **메타데이터·14D 요약·CNN/BiLSTM 전곡 거리**를 추가로 비교한
[발표용 통합 그림](../metadata_same_work/same_work_metadata_overview.png)과
[계산 조건·결과](../metadata_same_work/README.md)가 있다.
메타데이터 거리는 모두 0이고, 연주 표현의 거리는 서로 다르다.

## MIDI 원자료: 이 수치는 어디서 나왔나?

곡선 안의 중앙 107~110구간을 실제 MIDI에서 가져왔다.
세 연주의 pitch 범위와 초 단위 가로축 범위를 같게 했다. beat의 시간 간격이 서로 다른 것도 보인다.

| 연주 | note-on/off·velocity | 페달 신호 |
|---|---|---|
| MiyashitaM01M | [음표 원자료](source_midi/01_MiyashitaM01M_notes.png) | [CC64 원자료](source_midi/01_MiyashitaM01M_cc64.png) |
| Lou01M | [음표 원자료](source_midi/02_Lou01M_notes.png) | [CC64 원자료](source_midi/02_Lou01M_cc64.png) |
| SunY01M | [음표 원자료](source_midi/03_SunY01M_notes.png) | [CC64 원자료](source_midi/03_SunY01M_cc64.png) |

음표 그림에서 막대 시작은 note-on, 막대 길이는 note-off까지의 실제 건반 유지 시간, 색은 velocity다.
점선은 score beat가 연주에서 시작하는 실제 시각이다. note-on 간격은 속도에, velocity는 Dynamics에,
실제 길이/tempo 기반 기대 길이는 Articulation에 연결된다. CC64 그림은 페달 지표의 원신호다.
한 beat에서 시작한 여러 음의 velocity는 평균, 유효 정렬 음의 Articulation은 중앙값으로 요약한다.
소리의 데시벨이나 페달로 울리는 길이를 직접 측정한 것은 아니다.

독립 재계산한 12개 beat에서 velocity 평균·Articulation 중앙값·CC64 적분/전환값이 캐시와 일치했다.
유효 정렬 음 85개의 note-on/off를 MIDI와 대조한 최대 시간 차이는
0.521 ms다(기존 정렬 검증 허용값 2 ms).
원본 timestamp를 그림에서 수정하지 않았다.

## ② 다른 작품에서도 비슷한 연주 경향을 찾을 수 있다

| 역할 | 작품 | 실제 연주 파일 | A와의 거리 |
|---|---|---|---:|
| A · 기준 | Bach · Fugue · BWV 848 | `Lou01M.mid` | — |
| B · 가까운 후보 | Bach · Prelude · BWV 857 | `ToA01M.mid` | 0.544 |
| C · 대조 후보 | Bach · Prelude · BWV 857 | `Lan01M.mid` | 1.363 |

A는 고정한 기준 작품에서 **페달 깊이 평균이 중앙값 이상인 연주 중 Rubato 절댓값 중앙값이 가장 큰 연주**다.
B는 A의 작품을 제외한 561개 후보를 모두 비교해 거리 최소로 찾았다.
C는 B와 같은 작품의 후보 중 A와의 거리가 가장 큰 연주다. C를 멀리 있는 대조로 선택한 조건을 숨기지 않는다.
비교가 잘 나오는 A를 찾으려고 전체 작품의 기준 연주를 순회하며 최적화하지 않았다.
전체 570개 연주의 다른 작품 최근접 결과도 CSV로 저장했다.

쉬운 [A·B·C profile](cross_work/01_profile.png)은 각 요약값을 **자기 작품 후보 안에서의 백분위**로 표시한다.
50 부근은 중간 순위이며, 위로 갈수록 해당 특징의 요약값이 크다. 50이 residual=0이라는 뜻은 아니다.
선은 다섯 범주를 연결한 안내선으로, 범주 사이의 중간 값에는 의미가 없다.

| 연주 | 빠르기 | 루바토 폭 | 강약 성향 | 음 유지 성향 | 페달 깊이 |
|---|---:|---:|---:|---:|---:|
| A · Lou01M | 50.0 | 94.4 | 5.6 | 83.3 | 72.2 |
| B · ToA01M | 25.0 | 91.7 | 41.7 | 58.3 | 91.7 |
| C · Lan01M | 58.3 | 25.0 | 8.3 | 91.7 | 8.3 |

이번 결과에서 A와 B는 자기 작품에서 **루바토 변화 폭이 상위이고 페달 깊이도 큰 편**이다.
C는 페달 깊이가 작은 편이고, 음 유지 성향이 B보다 높다. 이런 차이가 여러 축의 거리에 반영된다.
빠르기와 Dynamics 등 일부 축에서는 C가 A에 더 가깝다. B가 모든 축에서 더 닮은 것은 아니다.

**루바토 폭과 대표 변화량을 구분해야 한다.** 쉬운 profile의 Rubato는 `p95-p5` 변화 폭이다.
검색에 함께 들어가는 `median(abs(relative/scale))`는 B가 A보다 작으며, B의 작품 내 최상위도 아니다.
따라서 “A와 B는 모두 루바토 양이 큰 연주”라고 넓게 설명하면 정확하지 않다.

| 추가 그림 | 확인할 내용 |
|---|---|
| [성향 좌표 7개](cross_work/02_embedding_tendencies.png) | 실제 검색값의 평균과 Rubato 절댓값 중앙값. |
| [변화 폭 좌표 7개](cross_work/03_embedding_widths.png) | `p95-p5`이며, Rubato 폭의 A·B 유사성과 Articulation 폭의 C 차이. |
| [B 작품의 모든 후보 거리](cross_work/04_target_candidates.png) | 같은 곡 후보끼리도 A와의 거리가 다르다. |
| [feature별 거리 성분](cross_work/05_feature_distance.png) | B는 특히 Articulation·Pedaling과 Rubato를 함께 보았을 때 C보다 가깝다. |

백분위는 이해를 돕는 **표시용**이다. 검색에는 아래 14차원 벡터를 쓰므로 profile의 눈으로 보이는 선 간격이 검색 거리와 같지는 않다.

## 정규화와 검색 방법

1. 동일 작품/grid에서 유효 beat의 위치별 중앙값을 common으로 만들고 D/A/P는 `relative = raw - common`을 계산한다.
2. Tempo는 기존 `individual_tempo`, Rubato는 기존 `relative_rubato`를 그대로 쓴다. 공통값 제거를 중복하지 않는다.
3. 각 작품/grid의 모든 유효 residual을 모아 채널별 scale을 계산하고 `standardized = relative / scale`을 사용한다.
4. Tempo·Rubato·Dynamics·Articulation은 MAD, 페달의 depth/down_ratio/changes는 각각 독립 SD다.
   MAD는 `1.4826 × median(|r - median(r)|)`이고, 0이면 IQR/1.349 → SD → unit 1 순으로 fallback한다.
   residual 자체를 다시 중앙값으로 이동하거나 clipping하지 않는다. 최종 선택 scale은 [scales.csv](scales.csv)에 기록했다.
5. 같은 작품/grid의 모든 후보와 7채널에서 동시에 유효한 beat만 요약한다. mask가 검색 단서가 되지 않게 한다.
6. 채널당 성향 1개 + `p95-p5` 폭 1개로 총 14좌표다. 성향은 평균이고 Rubato만 절댓값 중앙값이다.
7. Tempo/Rubato/Dynamics/Articulation/Pedaling을 다섯 묶음으로 비교한다.
   각 묶음의 좌표별 제곱 차이를 평균한 뒤, 다섯 묶음을 같은 가중치로 평균하고 제곱근을 취한다.
   Pedaling은 6좌표를 한 묶음으로 평균한다. 표시용 5축에서는 depth만 대표로 보여준다.

전체 캐시 1,036개 정렬 연주에서 robust note alignment와 후보 수 ≥ 5, 공통 유효 beat ≥ 32,
유효 비율 ≥ 50% 조건을 통과한 55개 작품/grid의 570연주를 검색 대상으로 사용했다.
채널 간 scale은 각 작품의 연주 변동을 단위로 만든다. 백분위는 동점에 중간 순위를 준다.

## 이 결과로 말할 수 있는 범위

이 사례는 **“서로 다른 작품에서도 현재 특징상 유사한 연주를 검색할 수 있다”**를 보여준다.
음악적으로 실제로 유사하다는 청취 판정, 추천 성능 향상, 사용자 취향 일치는 아직 측정하지 않았다.
현재 벡터는 연주 전체를 요약하므로 구절 순서를 보존하지 않고, Dynamics/Articulation 폭에는 작품 정보가 남을 수 있다.
페달 전환 횟수는 beat 길이와 CC64 장치 보정에도 영향을 받는다.
작품 공통값·scale에 해당 후보들이 포함되어 있어, 처음 보는 작품이나 독립 연주에 대한 성능 평가도 아니다.

변화 폭을 뺀 7차원 성향 벡터의 최근접 후보는 `Beethoven/Piano_Sonatas/21-1_no_repeat/Sladek02M.mid`로 달라진다.
이 비교는 [embedding_variants.csv](embedding_variants.csv)에 남겼다. 차원 수가 다른 두 거리의 숫자는 직접 비교하지 않는다.
검색 표현의 선택이 결과에 영향을 준다는 뜻이며, 예쁜 그림을 위해 이번 14차원 가중치를 다시 맞추지 않았다.
다음 성능 검증에는 청취자가 A에 대해 B와 다른 후보 중 어느 쪽을 더 비슷하게 느끼는지 독립 비교하고,
연주 축을 추가하기 전후의 추천 성능을 비교하는 절차가 필요하다.

## 원자료와 재현

- [profiles.csv](profiles.csv): 14좌표 원값·표시 백분위·후보 수.
- [same_work/beat_features.csv](same_work/beat_features.csv): 전체 구간의 raw/common/relative/standardized·mask/support/scale/status.
- [source_midi/note_events.csv](source_midi/note_events.csv): 선택 구간과 겹치는 원본 음표 이벤트.
- [source_midi/cc64_events.csv](source_midi/cc64_events.csv): 선택한 세 연주의 전체 원본 CC64 이벤트(구간 시작 전 상태 포함).
- [source_midi/note_ratios.csv](source_midi/note_ratios.csv), [beat_checks.csv](source_midi/beat_checks.csv): 기대/실제 음 길이·MIDI 대조 및 beat 검산.
- [retrieval_candidates.csv](retrieval_candidates.csv): A에 대한 다른 작품 전체 후보 거리와 순위.
- [all_query_neighbors.csv](all_query_neighbors.csv): 전체 연주의 다른 작품 최근접과 같은 후보 작품의 대조 결과.
- [stats.json](stats.json): 선정 규칙·데이터 revision·cache/source hash·검증 결과.

저장소 루트에서 실행한다. 기본 데이터 경로는 저장소 밖 `../datasets/ASAP`, `../datasets/nASAP`과 raw cache다.

```bash
.venv/bin/python classicfy-ai/scripts/validate_interpretation_examples.py
```

`--anchor`, `--out`, `--asap-root`, `--nasap-root`, `--cache`로 다른 경로/기준 작품을 지정할 수 있다.
그림과 CSV·JSON·이 문서를 함께 재생성한다. 모든 이전 분석 자료와 원본 feature 구현은 보존한다.
