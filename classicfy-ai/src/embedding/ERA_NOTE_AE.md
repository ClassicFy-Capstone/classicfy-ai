# 공통 Note AE → 시대별 AE + head

`feat/#20-g-loss-update`, `scripts/train_era_note_ae.py`.
5개 스타일 요약을 입력하는 TemporalAE가 아니라 MusicXML 조건 + 음표별 R11을
사용하는 `NotePerformanceAutoencoder`의 기존 체크포인트를 이어서 학습한다.

## 사용자 지정 분류

| 모델 키 | 시대 | 작품·악장 | 대표구간 |
|---|---|---:|---:|
| baroque | 바로크: 바흐 | 103 | 558 |
| classical_transition | 고전 + 고전–낭만 전환: 하이든·모차르트·베토벤 | 132 | 2,303 |
| romantic_impressionist | 낭만(후기 낭만 포함) + 인상주의 | 244 | 2,221 |

현대 작품은 없다. 시대가 다른 벡터의 거리는 비교하지 않는다. 분류는
`embedding.era_note_style.ERA_COMPOSERS`에 명시되어 있다. 미등록 작곡가는 실패시킨다.

## 구조와 데이터 범위

공통 가중치를 세 번 복사한다. 각 모델의 performance encoder와 decoder는
자기 시대 자료로 독립적으로 업데이트한다. 악보 조건 encoder는 사전학습 가중치로
고정하며, 정확히 같은 윈도별 조건을 캐시한다. 공통 체크포인트는 수정하지 않는다.

최대 128음표 윈도와 모든 tail을 유지하고, 하나의 대표구간에 속한 모든 윈도를
같은 학습 스텝에 포함한다. `z_ae`는 관측 음표 수로 가중평균한 윈도 z다.
head는 대표구간 `z_ae`에 적용한다. 학습과 추론에 동일한 집계 함수를 쓴다.

```text
z_style = z_ae + 0.5 * sigma_z * tanh(MLP(LayerNorm(z_ae)))
e = L2_normalize(z_style)
```

head의 마지막 층을 0으로 초기화해 출발 시 공통 벡터를 그대로 보존한다.
`sigma_z`는 train 구간의 공통 z 차원별 표준편차(최소 .05)다.
head의 조정 폭을 제한하지만 의미 보존을 보장하는 수학적 증명은 아니다.
`z_ae`, `z_style`, `e`를 모두 내보내며 검색은 같은 시대의 `e`로 한다.

## 대표구간 G 계약

`selected-clip-relative-g-v2-absolute-beat-cells`은 기존 8마디/전곡 G와 별도 계약이다.
같은 23개 이름(채널별 bias/spread + tempo roughness)을 대표구간 전체에서 계산한다.
선택 데이터에 이미 저장된 이전 정규화 벡터를 G 정답으로 재사용하지 않는다.

- 같은 split·작품·원본 XML·반복 경로의 악보 occurrence별 MIDI 관측 중앙값을 기준으로 한다.
- 원본 MIDI 해시 중복을 제거하고 cell당 최소 3개의 기준 연주를 요구한다. 기존 계약처럼 대상 연주도 포함될 수 있다.
- 각 채널은 80% 관측률, 8개 이상의 beat/onset이 있어야 한다.
- tempo는 원본 alignment와 MIDI→XML bridge로 복구한 실제 XML [시작 quarter, 끝 quarter) 구간이 같을 때만 연주 간 기준을 공유한다. 로컬 beat ID는 같은 연주 내 중복 제거에만 사용한다. 길이 가중치를 적용하며, 음표가 없는 모든 박자의 값까지 있는 것으로 간주하지 않는다.
- 나머지 채널은 화음의 음표 수가 가중치를 늘리지 않도록 onset 단위 균형을 맞춘다.
- bias는 중앙값, spread는 가중 P90−P10, roughness는 실제로 맞닿은 유효 XML 박자 구간 간 변화량을 quarter 간격으로 나눈 값이다.
- 품질은 coverage² × 기준 관측 수의 n/(n+3) 평균이다. 청취 정답 확률이 아니다.
- G 결측 구간도 모두 복원 학습에 포함한다. 결측 G를 0 정답으로 학습하지 않는다.

G 정규화는 시대별 train만으로 중앙값/IQR을 fit하고, IQR이 0이면 std를 사용한다.
관측이 20개 미만이거나 상수인 차원은 비활성화한다. 정규화 값은 ±5로 제한한다.
작품 분할 때문에 validation/test에는 해당 split 내부의 동일 작품 기준을 따로
구해 평가 정답만 정의한다. 이 값은 train 정규화·거리 calibration·학습에 섞지 않는다.
따라서 이 평가는 여러 비교 연주가 있는 catalogue의 G 근사 평가이며, 단일 새 곡에서
G 기준을 즉시 얻는 능력이나 인간 청취 판단을 검증한 것이 아니다.

## G loss와 보존 장치

동일 시대·동일 split에서 **서로 다른 작품**끼리만 비교한다. anchor의 유효 차원
signature를 고정하고 이를 모두 관측한 후보만 사용한다. tempo, timing, dynamics,
articulation 네 영역은 필요하며, pedaling은 관측 가능할 때 포함한다.
각 영역 내부 평균 후 영역 간 평균을 내므로 페달의 많은 차원이 거리를 지배하지 않는다.

```text
Dg(a,b) = mean_blocks(mean_features((g_a-g_b)^2))
w(a,b) = mean_blocks(min(quality_a_block, quality_b_block))
target(a,b) = Dg(a,b) / (Dg(a,b) + train_median_distance)
Dz(a,b) = 1 - dot(e_a, e_b)
L_pair = mean_valid_pairs(w * (Dz-target)^2)

margin = clip(0.2 * (Dg_an-Dg_ap) / train_distance_IQR, .05, .4)
L_rank = mean_valid_triplets(min(w_ap,w_an) * relu(Dz_ap-Dz_an+margin))
L_G = .5 L_pair + .5 L_rank
L = L_reconstruction + lambda_effective L_G + .05 L_z_anchor + .01 L_head_shift
```

rank는 배치 안에서 anchor마다 가장 가까운/먼 유효 후보를 쓰며,
G gap이 train IQR의 .1보다 작은 경우 생략한다. 낮은 신뢰도는 평균 분모에서
상쇄하지 않는다. 근접도 순서뿐 아니라 거리 차이도 학습한다.

복원 loss는 기존 equal-clip/equal-five-group Huber를 유지한다. 따라서 전체 구간의
관측된 모든 음표·채널을 사용한다. anchor loss는 공통 모델의 윈도 z와의 차이를
sigma_z로 나눈 MSE다. head shift 역시 sigma_z 단위로 정규화한다.

lambda .1/.3을 비교하고 3 epoch 동안 점진적으로 적용한다. 추가로 **현재 배치의
윈도 z에 대한 G 기울기 norm이 복원 기울기의 25%를 넘지 않도록 lambda를 낮춘다.**
이는 bottleneck에서의 제한이며 모든 파라미터 gradient의 비율을 보장하지 않는다.
cap이 활성화된 배치에서는 모든 신뢰도를 같은 비율로 낮춰도 lambda가 커지며
그 공통 배율이 상쇄될 수 있다. 배치 내부 쌍들의 상대 신뢰도는 유지되지만,
배치 전체의 낮은 신뢰도가 항상 최종 gradient 크기를 줄인다고 해석하면 안 된다.
AE 학습률은 5e-5, head는 1e-3이며 전체 gradient norm은 1로 제한한다.

## 선택 및 최종 학습

평가용 시작점은 `evaluated_best.pt`다. 작품·원본 MIDI·XML·score MIDI 해시로 연결된
기존 train/validation/test 분할을 유지한다. 각 시대별 2개 lambda를 최대 6 epoch씩 비교한다.
다음 validation 보존 조건을 모두 통과한 epoch 중 G score가 가장 작은 모델을 고른다.
공통 모델(epoch 0)도 후보이므로 개선이 없으면 성공한 미세조정으로 포장하지 않는다.

- 전체 복원 loss: 공통 모델의 105% 이하
- 각 복원 그룹 loss: 110% 이하
- 각 채널 MSE: 120% 이하
- 공통 z 대비 정규화 MSE: .1 이하
- 정규화 embedding의 차원별 평균 표준편차와 effective rank: 각각 기존의 50% 이상

모든 시대의 설정을 고정한 뒤 test를 평가하며 test 결과로 설정을 고르지 않는다.
공통 AE의 test 결과는 이전에 살펴본 적이 있으므로 완전히 새로운 외부 검증이라고
주장하지 않는다. near/far 순서 정확도와 G 거리 Spearman은 청취 성능과 별개다.

최종 배포 학습은 `final_all_5082.pt`에서 다시 시작해 각 시대의 **모든** 대표구간을
선택된 설정/epoch 수로 학습한다. 이 단계에는 독립 test가 없다. 전체 학습 자료의
복원·붕괴 조건을 통과한 epoch 중 G 목적이 가장 좋은 것을 저장한다.
`final.pt`는 encoder/decoder/score encoder/head/scaler를 포함하는 독립 체크포인트다.

```powershell
python scripts/train_era_note_ae.py --common-run /path/to/common/training_run01 `
  --selected-csv /path/to/selected_segments.csv --out /path/to/era_run --stage all
python scripts/verify_era_note_ae.py --common-run /path/to/common/training_run01 --run /path/to/era_run
```

추론 로더는 `embedding.era_note_style.load_era_checkpoint`다. 체크포인트에 저장된
scaler로 입력을 준비하고, `aggregate_clip_latents`로 대표구간을 집계한 다음 head를
적용한다. G 프로필은 추론 입력이 아니다. 전체 구간을 읽는 데 score와 MIDI는 필요하다.

## 실행 및 무효화 기록

현재 유효한 재실행 경로는 `outputs/era_note_ae_20261113/run03`이다.
`run01`은 분류 변경 전 준비 결과다. **`run02`는 사용 금지**다. 로컬 beat ID를
다른 연주와 직접 대응한 G 기준 오류가 발견되어 모든 G 개선 수치와 시대별
체크포인트를 무효화했다. 원본 공통 AE와 복원 입력은 변경되지 않았다.
`run02/INVALIDATED.json`과 `beat_identity_audit.json`에 근거를 보존한다.
v2는 원본 전처리 signature와 alignment/bridge 해시 및 실제 XML 좌표를 검증한다.

## Loss 기여도 대조 실험

공통 동결 / 복원만 / G만 / 복원+G 네 조건을 동일한 초기화, seed, train,
학습률, 배치 순서, joint에서 이미 선택한 epoch 수로 비교한다.
복원만/G만 조건도 동일한 z anchor/head 보존항을 유지한다. G만 조건은
직접 복원 gradient를 제거하지만, 복원 gradient에 기반한 G cap은 유지한다.
따라서 이는 안전장치 안에서 각 직접 loss 항의 추가 기여도 분석이다.

정해진 마지막 epoch를 평가하며 대조 실험의 test 결과로 모델을 다시 고르지 않는다.
고정 G near/far 정확도, 거리 MSE, Spearman, 복원 loss, z drift/붕괴 지표를 기록한다.
작품 단위 paired bootstrap 5,000회로 95% 및 9개 비교 보정 구간을 계산한다.
순서 정확도·거리 MSE·복원 loss 각각에 9개 비교 보정을 적용하며, 모든 지표를
한꺼번에 묶은 보정은 아니다. 거리 MSE 구간은 anchor별 평균 후 작품 균등 평균이므로,
전체 pair를 동일 가중한 점추정과 숫자가 다를 수 있다.
이는 고정 후보 pool과 한 학습 seed에 조건부인 사후 분석이다. 청취 정확도나
새로운 외부 검증을 뜻하지 않는다. 신뢰구간이 0을 포함하면 개선 확정으로 해석하지 않는다.

```powershell
python scripts/analyze_era_loss_ablation.py --run /path/to/era_run --common-run /path/to/common/training_run01
python scripts/report_era_loss_ablation.py --run /path/to/era_run
```

## 수정된 run03의 검증 결과

평가용 선택 epoch는 바로크 2, 고전+전환 3, 낭만+인상주의 5다.
두 lambda 상한의 결과는 같았다. 바로크·고전은 모든 스텝에서 상한보다 작은
gradient cap이 작동한 것을 확인하여 동일 경로를 재사용했고, 낭만은 두 후보를
각각 6 epoch 실행했다. test를 보기 전에 모두 선택을 고정했다.

| 시대 | Test 순서 정확도: 공통 → 복원+G | Test 복원 loss 변화 | Test G 목적값 변화 |
|---|---:|---:|---:|
| 바로크 | 60.48% → 60.89% | −0.311% | −4.965% |
| 고전+전환 | 64.36% → 67.89% | +0.108% | −36.138% |
| 낭만+인상주의 | 59.31% → 63.19% | −0.025% | −23.201% |

정확도는 고정한 G near/far triplet의 순서 일치율이다. 이 표는 point estimate이며,
loss별 기여도나 통계적 확실성을 단독으로 증명하지 않는다. 모든 test 보존 조건은
통과했다. 표는 train-only 공통 AE에서 시작한 평가 모델의 결과이며, 전체 자료로
재학습한 최종 체크포인트의 독립 test가 아니다.

최종 전체 자료 학습에서는 보존 조건을 통과한 epoch 2 / 3 / 4를 각각 저장했다.
5,082개 구간 모두 복원 학습과 임베딩 생성에 포함했다. 최종 checkpoint까지 실제
G 쌍에 참여한 고유 구간은 437 / 2,101 / 1,362개, 총 3,900개다. 나머지는
충분한 reference·공통 유효 차원·배치 상대가 없어 G에 직접 참여하지 않았으며,
AE 입력에서 제외한 것은 아니다.

`verification.json`은 원본 공통 체크포인트·packed 입력 해시 불변, split 독립성,
전체 G 프로필 재생성 일치, 5,082개 단위 벡터, 저장 모델 재로딩을 확인한다.
악보 실제 구간과 시간에서 템포를 다시 계산한 1,118,669개 관측값의 최대 오차는
4.802e-7 log2 QPM, 재로딩 임베딩의 최대 오차는 1.491e-7이었다.

주요 산출물은 `run03/<era>/final.pt`, `all_5082_era_embeddings.npz`,
`final_report.json`, `verification.json`, `production_G_coverage.json`,
`validation_curves.png`, `ablation/comparison.json`, `ablation/loss_contribution.png`다.

## Loss 기여도: 최종 해석

아래는 **복원만 대비 복원+G**, 즉 G 항을 추가한 효과다. 작품을 동일 가중하고,
9개 비교 보정을 적용한 약 95% 구간을 괄호에 적었다. 위의 triplet 균등 정확도
표와는 가중 방식이 다르므로 변화량도 다르다.

| 시대 | 순서 정확도 변화 (%p) | 거리 MSE 변화 (%) |
|---|---:|---:|
| 바로크 | −0.35 (−5.33, +3.05) | −13.82 (−20.98, −6.72) |
| 고전+전환 | +3.05 (+1.41, +4.80) | −47.00 (−52.19, −41.99) |
| 낭만+인상주의 | +3.21 (−0.63, +7.01) | −39.67 (−44.08, −35.32) |

G는 세 시대 모두 거리 오차 감소를 뒷받침한다. 순서 정확도는 고전+전환에서만
보정 구간이 0을 넘으며, 나머지 시대는 개선을 확정할 수 없다.

반대로 **G만 대비 복원+G**에서 추가 복원 항의 순서 정확도 및 복원 오차 개선은
세 시대 모두 보정 구간이 0을 포함했다. 고전+전환의 거리 MSE는 +12.25%
(+10.09, +14.57), 낭만+인상주의는 +5.06% (+4.14, +5.99)로 오히려 높았다.
따라서 두 loss를 함께 쓰면 모든 지표가 더 좋다거나, 보존이 추가 복원 항 덕분에만
가능했다고 주장하지 않는다. 모든 조건에 이미 공통 복원 사전학습과 보존항이 있고,
G만 조건에도 복원 gradient 기반 cap이 남아 있다는 범위 안에서 해석해야 한다.

최종 복원+G 모델은 모두 보존 기준을 통과했다. 다만 test Spearman은 약 .16–.26이며,
한 학습 seed·고정 후보 pool·G 대리 정답에 대한 사후 분석이다. 인간의 청취 유사도나
임의의 새 작품·새 학습 seed에서 같은 개선이 나는지는 검증하지 않았다.
이 대조 결과로 이미 고정한 최종 모델의 선택을 바꾸지 않았다.

## 추가 복원 항을 약하게 둔 validation 검색

`scripts/validate_era_reconstruction_weight.py`는 보호 장치를 둔 G만을 기준으로
`alpha * Rec + lambda_effective * G + .05 * z_anchor + .01 * head_shift`를 비교한다.
alpha=0은 기존 G만의 validation 결과, alpha=1은 기존 복원+G 결과를 재사용한다.
새 alpha .01/.1/.3은 동일 공통 체크포인트·seed·배치 순서·학습률에서 각각 시작하고,
시대별 2/3/5 epoch의 마지막 체크포인트를 비교한다.

G cap의 기준은 alpha를 곱하기 전 복원 gradient로 고정한다. alpha와 함께 cap까지
낮추면 G 학습도 약해져 비교가 달라지기 때문이다. 따라서 G gradient가
`alpha * Rec` gradient의 25% 이하라는 뜻은 아니다. 또한 AdamW를 사용하므로
복원 계수 .01이 파라미터 업데이트 크기를 정확히 100분의 1로 만든다는 뜻도 아니다.

실행 전에 정한 실무 판정선은 공통 보존 기준 통과, G만 대비 작품 균등 복원 오차
최소 .1% 감소 및 보정 구간 상한<0, 전체 쌍 거리 MSE 증가 1% 이내,
전체 triplet 정확도 하락 1%p 이내다. 이 선은 validation 탐색용 판단 기준이며
통계적 비열등성 증명이나 사용자 요구에서 정해진 절대 한계가 아니다.
통과한 약한 복원 후보가 없으면 alpha0을 권장하며, 모든 후보의 원시 지표와
tradeoff를 함께 저장한다. validation으로 선택한 뒤의 구간은 확인적 증거가 아니다.

새 test 평가 및 전체 자료 재학습은 수행하지 않는다. 기존 test 관찰이 이 검색의
동기가 되었으므로, 결과는 한 seed의 validation 탐색으로 해석한다.

```powershell
python scripts/validate_era_reconstruction_weight.py --run /path/to/run03 --common-run /path/to/common/training_run01
python scripts/report_era_reconstruction_weight.py --run /path/to/run03
```

산출물은 `run03/weak_reconstruction_validation/`의 protocol, 후보별 checkpoint와
validation 진단 배열, report.json, comparison.json, validation_tradeoff.png다.

### Validation 검색 결과

새 후보 9개 모두 학습과 checkpoint/진단 검증을 마쳤고, 공통 보존 기준을 통과했다.
시험한 alpha 0/.01/.1/.3/1 중 세 시대 모두 alpha0의 종합 G score가 가장 낮았다.
약한 복원 후보 중 미리 정한 유용성 조건을 모두 통과한 후보는 없었다.

가장 약한 양의 계수 .01의 G만 대비 전체 validation 평균 변화는 다음과 같다.

| 시대 | 거리 MSE 변화 | 복원 loss 변화 | 순서 정확도 변화 |
|---|---:|---:|---:|
| 바로크 | +1.41% | −0.393% | +0.214%p |
| 고전+전환 | +9.32% | −0.350% | −0.382%p |
| 낭만+인상주의 | +3.98% | −0.018% | −0.691%p |

작품 균등 복원 오차 변화와 보정 구간은 바로크 −0.300% (−0.568, −0.061),
고전+전환 −0.410% (−0.860, +0.075), 낭만+인상주의 +0.013%
(−0.367, +0.401)였다. 바로크에는 작은 복원 이득이 확인되지만 거리 손해도 있다.
따라서 복원 항이 아무 효과 없다고 해석하지 않고, 현재의 거리 우선 목적과 판정선에서
보호 장치를 둔 G만을 권장한다. 계수를 낮추면 기존 alpha1보다 거리 손해는 작아졌으나,
이 범위·한 seed·고정 학습량의 validation 비교에서 G만을 이기는 균형은 찾지 못했다.
기존 전체 자료 최종 모델은 변경하지 않았으며, 이 추천은 validation 선택 결과다.

## 전체 자료 보호 G 최종 모델

2026년 11월 15일 `outputs/era_note_ae_g_only_20261115/run01`에 전체 자료 재학습을
완료했다. 공통 `final_all_5082.pt`에서 각각 시작했고, 직접 복원 항 없이
G와 z anchor 및 head 이동 억제항을 유지했다. 선택된 epoch는 2 / 3 / 4다.
5,082개 모두 encoder와 anchor에 사용하고 64차원 z_ae z_style e를 저장했다.
최종 checkpoint까지 직접 G에 참여한 구간은 3,900개, anchor만 적용한 구간은
1,182개다. decoder 전용 가중치와 악보 encoder는 공통 가중치를 유지했다.
모든 보존 조건과 원본 해시 및 재로딩 검증을 통과했다. 전체 학습 자료에서의
복원 loss 증가는 약 .043% / .092% / .087%다. 이는 독립 test 결과가 아니다.

```powershell
python scripts/train_protected_g_era_final.py --source-run /path/to/run03 --common-run /path/to/common/training_run01 --out /path/to/protected_g_final
python scripts/verify_era_note_ae.py --run /path/to/protected_g_final --common-run /path/to/common/training_run01
python scripts/audit_era_g_coverage.py --run /path/to/protected_g_final
python scripts/package_protected_g_era_final.py --run /path/to/protected_g_final
```

시대별 `final.pt`와 `embeddings.npz`, 통합 `all_5082_era_embeddings.npz`,
검증과 사용 설명을 포함한 `protected_g_era_models_5082.zip`을 제공한다.

## 시대별 G 기울기 상한 비교

검증 분할에서 기존 25%를 대조군으로 50%·100%를 비교했다. 같은 시대의 마지막
validation epoch에서 전체·그룹·채널 복원과 임베딩 보존 조건을 모두 통과한
완화 후보 중 G score가 가장 낮은 것을 선택했다. 바로크 100%, 고전+전환 50%,
낭만+인상주의 100%를 선택하고 전체 5,082개로 다시 적합했다. 수치와 조건은
[비교 보고서](../../analysis/era_g_budget/comparison.md)에 있다.

```powershell
python scripts/validate_era_g_budget.py --source-run /path/to/run03 --common-run /path/to/common/training_run01 --out /path/to/budget_validation/baroque --era baroque
python scripts/train_protected_g_era_final.py --source-run /path/to/run03 --common-run /path/to/common/training_run01 --out /path/to/g_budget_production --budget-selection /path/to/budget_validation
```

validation 명령의 시대와 출력 폴더를 각각 `classical_transition`,
`romantic_impressionist`로 바꾸어 나머지 두 결과를 만든다. 상한 완화 모델에서
임베딩 거리로 고른 문항은 [별도 AXY 도구](../../tools/ae_selected_era_listening_relaxed/README.md)에 있다.

## 새 대표구간 기반 AXY 청취

2026-11-16 현재 문항은 `g-verified-v2` 정책으로 갱신했다. A의 고정 유효 G
차원을 양쪽 후보가 모두 관측하고, G와 임베딩 순서 및 차이 기준을 모두 통과한다.
X/Y는 서로 다른 연주자이며, 제목상 장르·악장 유형과 악보 onset 밀도·화음 밀도를
맞춘다. 아래의 기존 전역 최근접 B / 작품 전체 최원거리 C 규칙은 **이전 pack의
기록**이다. 현재 정책은 조건을 만족하는 후보 안에서 선정하며 전역 최근접을
보장하지 않는다. 현재 선정 수치·한계는 `analysis/ae_selected_era_listening/selection_report.md`,
실행·재현 방법은 청취 도구의 `README.txt`에 정리한다. UI 엔진은 유지한다.

`tools/ae_selected_era_listening`은 #17의 UI와 재생 및 응답 엔진을 복사하고
문항 데이터와 입력 범위 설명만 변경한 프로토타입이다. source commit은
`d5311e6247b3d8fff43f5c314b6c817c6ad42a64`다. 같은 시대에서 A의 최근접
다른 작품 B를 고른 뒤 B와 같은 작품에서 A로부터 가장 먼 다른 연주 C를 고른다.
영상 상태로 B나 C를 차순위로 바꾸지 않고 부적합한 문항 전체를 제외한다.
새 임베딩 전체를 시대별로 검색하고, 거리 gap 내림차순과 작품 중복 배제 조건으로
3 / 3 / 4문항을 선정했다. 유튜브 시간은 제공된 youtube_matches.csv를 그대로 쓴다.

`analysis/ae_selected_era_listening`에 선정 근거, 원본 UI 비교 해시와 데이터 검증을
보관한다. v2 영상은 30개 중 8개에서 브라우저 재생을 확인했다. 초기판 검증 기록은
`archive`에 있다.
UI의 원래 블라인드·무작위 X/Y·선택 후 공개·수정 잠금·JSON 내보내기를 유지했다.
실행 명령은 `python tools/ae_selected_era_listening/serve.py --port 8770`이며
접속 주소는 `http://localhost:8770/`이다. QA 응답은 `?qa=1`로 구분한다.
