# 8마디 g 거리·신뢰도 기반 Triplet loss

`NoteAETrainingConfig.style_objective`로 다음 네 모드를 선택한다.

| 모드 | Margin | Weight |
|---|---|---|
| `legacy` (기본) | 기존 `margin` (기본 .2) | 1 |
| `confidence` | 기존 `margin` | 관측 품질 |
| `adaptive` | g gap 기반 | 1 |
| `adaptive_confidence` | g gap 기반 | 관측 품질 |

선정 규칙·g 거리·복원 목적은 유지한다. 이 기능은 기존 `--windows`와 고정
Triplet 학습 경로를 사용한다. 전곡 g는 별도다. Packed 입력은 원본 R fingerprint를
추가로 보존한 manifest에서 동일한 g/Triplet 경로를 지원한다.
기존의 배치당 style 배치 1개, loader 소진 시 재순회 정책을 유지하며 노출 수를 기록한다.

## Margin

각 Triplet t의 기존 `D_an-D_ap`를 train calibration에서 해당 `J_a` signature의
`distance_scale`로 나눈다. `distance_scale`은 거리 IQR이며 기존 std fallback을 따른다.

```text
r_t = (D_an - D_ap) / distance_scale[J_a]
m_t = clip(margin_scale * r_t, margin_min, margin_max)
```

초기 설정은 `margin_scale=.2`, `margin_min=.05`, `margin_max=.4`다.
Scale이 유한하지 않거나 기존 StyleConfig epsilon 이하이면 adaptive 모드는 거부한다.
배치별 정규화나 validation/test calibration fit을 하지 않는다.

## 품질

새로 추출한 profile은 별도 `quality_schema=style-quality-v1` 계약을 기록한다.
기존 artifact schema는 유지하며 추가 필드도 profile/artifact content hash에 포함된다.
`g_raw`, `g_mask`, 표준화 식과 g 선정 기준은 바뀌지 않는다.

채널 k의 유효 위치 v_i, 원래 가중치 omega_i, 독립 reference 수 n_i에 대해:

```text
c_k = sum(omega_i * v_i) / sum(omega_i)
u_k = sum(omega_i * v_i * n_i/(n_i+3)) / sum(omega_i * v_i)
q_k = c_k**2 * u_k
```

관측이 없으면 u=0이다. 기존 g 유효성 조건에 탈락한 차원의 품질은 0으로 둔다.
Bias/spread는 같은 채널 품질을 공유한다. Tempo roughness는 실제 인접 valid beat
쌍만 사용하며, 쌍의 quarter 간격으로 가중하고 양 끝 reference 수의 최솟값을 쓴다.
결측을 건너뛰어 연결하거나 화음 내 결측 가중치를 재분배하지 않는다.

Profile에는 `g_quality[23]`, 채널별 `reference_support[11]`,
`roughness_reference_support`와 계산 계약을 저장한다. 이는 데이터 품질 가중치이며
청취자 판단의 정답 확률이 아니다. Reference에는 기존 정책대로 대상 연주가 포함될 수 있다.

모든 역할에 anchor의 동일한 J_a를 사용한다. 영역 b별 선택 차원의 품질 평균 Q_xb를 구해:

```text
w_t = mean_over_present_blocks(min(Q_ab, Q_pb, Q_nb))
```

페달 14차원이 영역 비중을 증가시키지 않는다. Triplet에 품질 사본을 저장하고,
DataLoader에서 원래 profile과의 일치·계약·범위를 검증한다.

## Loss와 gradient

```text
d_ap = 1 - cosine(z_a, z_p)
d_an = 1 - cosine(z_a, z_n)
hinge_t = relu(d_ap - d_an + m_t)
L_style = sum(w_t * hinge_t) / batch_size
L_total = L_rec + lambda_style * L_style [+ 기존 mean-tempo 보조항]
```

분모는 weight 합이 아니다. 배치 전체의 신뢰도가 낮거나 크기가 1이어도 신뢰도에
비례해 gradient가 줄어든다. 모든 weight가 0이면 유한한 0 loss/gradient를 낸다.
Margin·weight는 detach하고 모델에만 gradient를 전달한다.
큰 margin은 더 넓은 간격을 요구하며, 활성 hinge의 gradient를 직접 배수로 키우지는 않는다.

## 실행과 artifact 이행

`STYLE_TRIPLETS.md`의 기존 CLI로 새 profile을 추출한 다음 scaler → scaled profile →
calibration → triplet을 순서대로 새 출력 경로에 생성한다. 새 품질 필드 때문에 profile ID가
달라지므로 기존 downstream artifact를 섞어 쓰지 않는다. 원본 R을 다시 만들 필요는 없다.
이 구현은 품질 추출의 kappa=3과 coverage power=2를 버전 계약으로 고정한다.
다른 값은 계약을 갱신하고 artifact를 재생성하는 별도 실험으로 다룬다.

`classicfy-ai`에서:

```sh
python scripts/train_note_autoencoder.py train \
  --windows run/windows.json \
  --triplets run/triplets.json --g-profiles run/profiles.json \
  --g-scaler run/scaler.json --g-calibration run/calibration.json \
  --config configs/note_ae_g_loss.json --out run/training
```

이 config는 `adaptive_confidence`, lambda=.1을 선택하며 나머지 학습·모델 기본값을 쓴다.
같은 입력·seed·예산에서 style_objective만 바꿔 네 모드를 비교한다.
이 수치들은 초기 실험값이며 성능 향상을 검증한 결과가 아니다.

이전 품질 없는 artifact는 legacy/adaptive 모드에서 읽을 수 있다. Confidence 모드는
품질 있는 profile/triplet 재생성을 요구하며 기본 weight=1로 대체하지 않는다.
양수 lambda로 새 모드를 요청했는데 Triplet이 없으면 학습 시작 전에 오류를 낸다.

Checkpoint/training.json에 별도 `g-triplet-objective-v1` 계약과 설정을 저장한다.
이전 목적함수 계약 없는 checkpoint는 legacy로만 해석한다. 목적함수 설정·품질 계약·
g artifact가 다른 epoch resume는 거부한다. 모델의 추론용 schema는 바뀌지 않는다.

## 진단과 검증

`history[].style_diagnostics`와 epochs.jsonl은 노출 수 가중 평균으로 다음을 기록한다:
가중 전 hinge, 적용 confidence/margin 평균, margin 상·하한 clipping 비율,
고정 .2 margin 만족률. 기존 스타일 loss는 confidence 적용 후 값이고
`weighted_style_loss`는 여기에 lambda_style을 곱한 값이다.
기존 `train_triplet_margin_satisfied`는 적용된 margin 기준이다.
Checkpoint 선택은 기존 validation 복원 loss를 유지한다.

```sh
PYTHONPATH=src:tests python -m unittest \
  unit.embedding.test_g_style_loss integration.test_style_triplets \
  integration.test_note_ae_pipeline -q
```

테스트는 legacy loss/gradient 재현, singleton·zero weight, adaptive clipping,
품질의 관측·reference·화음·영역 균형, 새 감독값 전달, 네 모드의 학습/정확한 epoch
resume, 이전 artifact/checkpoint 호환 및 잘못된 계약 거부를 검증한다.
합성 테스트 통과는 실제 검색·청취 성능 향상을 의미하지 않는다.

## 로컬 대표구간 원본으로 전체 학습

`scripts/run_local_atepp_ae.py`는 대표구간 목록의 모든 연주를 조사해 로컬 MusicXML,
원본 MIDI, 승인된 정렬이 있는 자료를 준비한다. 파일명이 같다는 이유로 악보를 같다고
보지 않는다. Score MIDI의 음높이와 quarter 위치가 XML의 유일한 타건 occurrence에
1e-4 quarter 안에서 대응하는지 확인한다. Score aliases와 기존 정렬의 verified 행이
각각 80% 이상이고 최소 20행인 경우에만 원본 MIDI onset/offset과 재대조한다.
같은 음높이·시각의 모호한 중복이나 검증되지 않은 행은 임의로 대응시키지 않는다.

```sh
python scripts/run_local_atepp_ae.py \
  --dataset /path/to/atepp_candidate_style_selected_recovered_final \
  --out /path/to/new-training-run --workers 3
```

대표 MIDI clip 대신 대응하는 원본 전곡으로 모든 full 8마디를 학습한다. 원본·선정 구간은
수정하지 않는다. 승인된 대안 중 XML과 직접 대응하는 원본 score MIDI를 우선하며
선정 구간과 다른 정렬을 학습에 쓴 경우도 inventory에 보존한다. 표본을 임의 축소하지 않으며
XML 부재·악보 좌표 불일치·특징 추출 실패는 모두 기록한다. 품질 gate를 통과한 전체 자료를
쓴다는 뜻이지 원본 ZIP의 모든 연주가 학습 가능하다는 뜻은 아니다.

Work ID, 동일 XML SHA, 동일 MIDI SHA를 union한 뒤 고정 SHA256(seed=2026) 70/15/15
split을 부여한다. 이는 로컬 실행의 새 protocol이며 다른 실험의 split을 덮어쓰지 않는다.
R/S scaler와 g reference/scaler/calibration은 train에서만 fit한다. R scaler는 녹음별
population moments를 병합하고, g reference는 cohort별로 처리해 메모리를 제한한다.
g 요약에는 compact 구간 식별자를 사용하며 전체 음표 소속은 해시된 window 파일로 보존한다.
Packed 입력에서는 구간 ID·작품·녹음·악보 SHA·q 범위·N과 원본 R fingerprint를 대조한다.
모든 적격 복원 구간을 사용하고 g 선정에 실패한 구간도 복원 학습에서 제거하지 않는다.

초기 로컬 설정은 CPU/4 threads, batch=8, adaptive_confidence/lambda=.1,
최대 100 epoch, validation patience=10/min_delta=.0001, 길이별 배치다.
배치 수 제한은 없다. Validation 최저 best.pt를 선택하고 test는 마지막에 한 번 평가한다.
`status.json`, `extraction.jsonl`, `inventory.json`, packed/style artifacts,
`training_config.json`, `training/epochs.jsonl`, best.pt/last.pt와 test.json을 남긴다.
`--prepare-only`는 전처리까지, `--train-only`는 준비된 입력으로 학습만 실행한다.
동일 --out 재실행은 완료된 추출을 재사용하고 last.pt가 있으면 새 training_resumeNN
폴더로 epoch 재개한다. 기존 입력과 설정이 달라지면 새 --out을 사용한다.

Packed 학습을 따로 실행할 때는 아래 CLI도 사용할 수 있다.

```sh
python scripts/train_indexed_note_autoencoder.py train \
  --corpus /run/packed --config configs/note_ae_g_loss.json --out /run/new-training \
  --triplets /run/style/triplets.json --g-profiles /run/style/profiles.json \
  --g-scaler /run/style/scaler.json --g-calibration /run/style/calibration.json
```

학습 catalogue를 만들 때 사용한 대표구간 선정 자체가 모델과 무관한 기존 feature에 의존하므로,
선정된 catalogue의 test 수치를 별도의 청취 실험이나 무작위 모집단 성능으로 해석하지 않는다.
