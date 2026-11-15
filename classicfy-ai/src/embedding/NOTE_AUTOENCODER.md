# 8마디 조건부 연주 시퀀스 AE

검증을 통과한 ATEPP 전체 입력의 학습·평가 상태는
[전체 데이터 실행 보고서](../../analysis/note_ae_full_corpus_reconstruction/README.md)에 기록한다.
기존 7작품 실험의 전체 epoch 순회와 전체 코퍼스 학습은 구분한다.

대규모 per-record window index는 한꺼번에 전부 적재하지 않고 텐서 캐시로 준비한다.
S/R 값·관측 mask·구간 대응을 확인하고 train 전용 scaler를 한 번 적용한다.
ScoreEncoder의 C는 학습 중 계산한다. `bucket_by_length`는 padding을 줄이며
모든 full train window를 매 epoch 한 번씩 사용한다.

```sh
.venv/bin/python scripts/train_indexed_note_autoencoder.py prepare \
  --source /path/to/validated-pipeline --out runs/full-corpus-inputs
.venv/bin/python scripts/train_indexed_note_autoencoder.py train \
  --corpus runs/full-corpus-inputs --config configs/note_ae_full_corpus_reconstruction.json \
  --out runs/full-corpus-training
```

이 설정은 MPS GPU가 필요하다. 초기 tempo 조건·평균 보조 head·Triplet은 사용하지 않는다.
기존과 동일한 beat 평균·복원 목적을 일괄 연산하며, 기존 구현 대비 출력과 gradient를 대조했다.
`epochs.jsonl`은 완료 epoch별 지표와 구간별 실제 노출 횟수,
`progress.json`은 최근 배치 진행 상태다. 체크포인트는 완료 epoch 경계에서 저장한다.

기존 g의 full 8마디 manifest에서 원래 S/R을 읽는다. ΔR과 g_scaled는 AE 입력이나
복원 목표가 아니다. g는 고정 Triplet 관계의 출처로만 사용하며 학습 중 갱신하지 않는다.

## 모델·데이터 계약

- 기존 `NoteScoreEncoder`가 8마디의 S에서 C를 즉시 계산한다. 구조는 변경하지 않았다.
  기본 `score_gradient=joint`에서는 복원·Triplet loss가 ScoreEncoder까지 전달된다.
  `frozen`은 그 가중치를 고정한다. 사전 계산 C 캐시를 사용하는 구현은 아니다.
  새 모델을 frozen으로 시작하면 초기 ScoreEncoder 가중치도 고정된다.
- `C[B,N,d_c]`, 표준화된 `R[B,N,11]`, 관측 `M[B,N,11]`, 실제 음표 `P[B,N]`,
  `lengths[B]`, beat IDs를 사용한다. N은 가변이며 배치에서만 오른쪽 padding한다.
  큰 window를 고정 음표 수로 자르지 않는다. 필요하면 batch size를 명시적으로 줄인다.
- 원본 R 캐시에만 train R scaler를 한 번 적용한다. 기존 R scaler를 제공할 수 있다.
  기본 fit은 입력 manifest의 train 전곡 R을 사용한다(기존 R 정책대로 tempo는 beat당 한 번).
  S 수치 scaler도 train 악보에서만 fit한다. g scaler와 별개이며 둘 다 checkpoint에 저장한다.
- tail·빈 window·복원 관측이 없는 window는 사유와 함께 제외한다. g 선정 실패만으로
  복원 window를 제외하지 않는다. 원본 R 행·마스크·파일은 변경하지 않는다.
- source SHA, 판본, 반복 occurrence, performance/window ID, 음표 키와 q 구간을 검증한다.
  별도 validation manifest는 validation 작품만 받아 기존 train/g manifest를 유지할 수 있다.

기본 모델(`configs/note_ae.json`):

```text
c = Linear(C)                                            64
x = MLP([c, R_filled, M])                               128
h = packed BiGRU(x)                              2 × 64/방향
z = MLP(masked_mean(h, P))                               64
추천 표현 e = normalize(z)
d = packed BiGRU([c, unnormalized z])             2 × 64/방향
R_hat = linear heads(d)
```

두 GRU는 2층·층간 dropout .1이다. decoder에는 R·g·encoder의 음표별 h를 전달하지 않는다.
음표별 onset/velocity/articulation 각 1열과 pedal 7열을 예측한다. tempo는 같은 beat에
대응하는 실제 음표의 decoder state를 평균하여 한 번 예측하고 같은 행에 복사한다.
음표가 없는 beat는 예측·복원 대상에서 제외한다. 임의 음표를 생성하지 않는다.
출력 순서는 기존 R의 11열과 같다.

## 목적함수

g 거리 차이와 관측 품질을 반영하는 선택적 목적함수는
[G loss 설정·계산 계약](G_LOSS.md)에 설명한다. 기존 설정의 기본 모드는 `legacy`다.

표준화된 R의 Huber(delta=1)를 사용한다. 구간마다:

- onset/velocity/articulation은 채널별 관측된 실제 음표의 평균.
- tempo는 관측 target이 일관적인 beat당 한 번 평균.
- pedal은 각 채널의 관측 음표 평균 후, 관측된 채널끼리 평균.
- 위 다섯 묶음 중 관측된 묶음만 균등 평균.

이후 유효 window끼리 평균한다. padding/결측은 분모에 포함하지 않으며,
관측 없는 묶음의 0을 평균에 넣지 않는다. 관측 없는 배치는 skip 사유를 제공한다.
g의 onset 균형 가중치는 복원 loss에 적용하지 않는다.

고정 Triplet manifest의 원래 세 C/R window에 동일 인코더를 적용한다.
`d(a,b)=1-dot(normalize(z_a), normalize(z_b))`,
`L_style=mean(relu(d_ap-d_an+margin))`,
`L_total=L_rec+lambda_style*L_style`다.
margin=.2, lambda=.1은 실행용 초기 설정이며 g의 delta_g와 별개다.
Triplet이 없으면 복원만 실행하고 거리 학습이 없었음을 기록한다.

복원 배치당 style 배치 하나를 결합하며, style loader 재순회·실제 노출 수와 window/Triplet ID를
기록한다. style의 세 window를 복원 배치에 추가하지 않는다. 기본 batch size는 각각 4,
AdamW lr=.001·weight decay=.0001·gradient clip=1·seed=2026이다.

## 실행

전체 순회 복원 전용 설정은 `configs/note_ae_full_reconstruction.json`이다.
`max_steps_per_epoch=0`이면 마지막 짧은 배치까지 모든 train 구간을 정확히 한 번 사용했는지 검사한다.
최대 epoch와 validation early stopping은 별개이며, 실제 최소 validation 복원 loss의 모델을 선택한다.
`evaluate_train_each_epoch=true`는 학습 중 minibatch loss와 구분되는 전체 train의 eval loss를 기록한다.

`configs/note_ae_plateau_reconstruction.json`은 같은 조건에 `lr_scheduler=plateau`만 추가한다.
Validation 이후 `ReduceLROnPlateau(mode=min, threshold_mode=abs)`를 한 번 호출하며,
`learning_rate`는 해당 epoch에서 사용한 값, `next_learning_rate`는 다음 epoch에 적용할 값이다.
기본 감소 계수는 .5, patience는 5, 절대 개선 threshold는 .0001, 최소 LR은 .00001이다.
LR 감소는 early stopping의 대기 횟수를 초기화하지 않는다. Scheduler 상태는 optimizer와 함께
checkpoint에 저장하고 resume 시 복원한다. 기본값 `lr_scheduler=none`은 기존 고정 학습률 동작을 유지한다.

`classicfy-ai`에서 실행한다. 아래 경로는 사용자가 준비한 각 단계의 산출물이다.
출력 디렉터리는 새 경로여야 한다.

```sh
# 복원만 짧게 점검(2 epoch × 최대 2 update). config 생략 시 본문 기본 설정.
.venv/bin/python scripts/train_note_autoencoder.py train --windows g/windows.json --config configs/note_ae_smoke.json --out runs/note-ae-rec

# 고정 g 관계를 함께 학습. 별도 validation은 선택 사항.
.venv/bin/python scripts/train_note_autoencoder.py train --windows g/windows.json --validation-windows validation/windows.json --triplets g/triplets.json --g-profiles g/profiles.json --g-scaler g/scaler.json --g-calibration g/calibration.json --config configs/note_ae.json --out runs/note-ae-style

# 같은 설정·입력으로 완료 epoch 다음부터 재개. config의 epochs는 총 목표 epoch 수.
.venv/bin/python scripts/train_note_autoencoder.py train --windows g/windows.json --config resume-config.json --resume runs/note-ae-rec/last.pt --out runs/note-ae-resumed

# checkpoint의 R/S scaler를 그대로 사용하여 구간별 z/e와 원본 키를 저장.
.venv/bin/python scripts/train_note_autoencoder.py export --windows g/windows.json --checkpoint runs/note-ae-rec/last.pt --split test --out test-embeddings.npz

# 같은 악보 구간 두 연주의 z 교환/zero 진단.
.venv/bin/python scripts/train_note_autoencoder.py diagnose --windows g/windows.json --checkpoint runs/note-ae-rec/last.pt --split train --out latent-ablation.json
```

외부 R scaler는 train의 `--r-scaler scaler.json`으로 지정한다.
재개 시 원래 validation·g 인자가 있었다면 동일하게 제공해야 한다.
새 작품 추론은 해당 작품의 새 window manifest와 checkpoint의 저장된 scaler를 사용한다.
새 작품의 g/reference를 fit할 필요가 없다. 추론 모델은 배치의 scaler 식별자도 검사한다.

Checkpoint에는 모델·optimizer·두 scaler·config·완료 epoch/step·Python/NumPy/Torch RNG·loader
RNG·작품 split·원본 window 및 고정 g dependency를 저장한다. epoch 경계 재개를 지원한다.
다른 schema/scaler/입력 split/Triplet로의 재개를 거부한다. `weights_only=True`로 로드한다.
validation 복원 오차가 개선되면 best.pt, 매 epoch last.pt를 저장한다. validation이 없으면
best를 선정하지 않고 last만 저장한다. test는 학습·선정에 사용하지 않는다.

## 점검

```sh
PYTHONPATH=src:tests .venv/bin/python -m unittest integration.test_note_ae_pipeline -q
```

shape·열 순서·padding 불변성·결측의 loss 불변성·Huber 손계산·tempo 중복 제외·pedal 비중·
전부 결측·Triplet 손계산·유한 gradient·ScoreEncoder 학습 정책·checkpoint 출력 재현·
epoch 경계 재개의 동일 결과·작은 synthetic 복원 오차 감소·style loader 재순회를 확인한다.

짧은 실행은 config에서 `epochs`, `max_steps_per_epoch`, `max_validation_batches`를 명시적으로
제한한다. 두 max 값이 0이면 제한하지 않는다. 제한 실행의 train/validation 통계는 전체
데이터 평가가 아니다. `training.json`에 묶음별 loss·유효 수·실제 노출·z norm/분산·train
Triplet 거리/충족률·제외 사유를 저장한다. Triplet 충족률은 train 진단이며 일반화 성능이 아니다.

g 일치나 작은 실행만으로 청취 스타일 유사성·작품 정보 제거를 주장하지 않는다.
ATEPP은 검증된 XML 대응과 원본 R이 준비되어야 동일 경로를 사용할 수 있다.


실제 실행 점검(ASAP, 기존 작품 split 유지): train 후보 80구간·validation 2구간,
고정 Triplet 10개를 연결했다. 4 update에서 복원 8개 구간과 Triplet 6개(총 8회 노출)를
사용했다. 제한된 고정 train 표본의 복원 오차는 .47827→.46604였다.
마지막 train 배치 묶음 평균 L_rec=.40305, L_style=.19948,
가중 style=.01995, L_total=.42300이었다. validation은 epoch 1의 .72019에서
2의 .76008로 증가했다. z 교환 시 출력 변화 MSE는 약 9.59e-6로 작았다.
이 결과는 연결·수치 안정성 점검이며 스타일 표현 학습 완료를 뜻하지 않는다.
실행 산출물은 workspace의 `diagnostics/note_ae_validation/run01/`에 있다.
위 ASAP 소규모 점검 당시에는 ATEPP 대규모 학습을 실행하지 않았다.
이후 선정 대표구간 전체를 사용한 실행 결과는 아래에 기록한다.

## 선정 대표구간 5,082개 학습 경로

`scripts/prepare_selected_note_ae.py`와 `scripts/train_selected_note_ae.py`는 위
`NotePerformanceAutoencoder`에 원본 MusicXML의 S와 원본 연주 MIDI의 음표별 R11을
공급한다. 7채널 `TemporalAutoencoder`와는 다른 실행 경로이다.

입력 준비는 `--stage bridges`, `--stage clips` 순서로 실행한다. 두 단계 모두
`--workspace <데이터셋을 보관한 workspace> --out <전처리 출력 폴더>`를 받는다.
원본 score MIDI와 MusicXML 사이의 음고·악보 시간·반복 경로를 검증하고, 선정 CSV의
시작부터 끝까지 사용한다. 원본 MIDI 해시와 기존 정렬 캐시를 검증하며, 정확히 연결되지
않는 음표는 관측 마스크를 끈다. MIDI를 XML로 변환해 악상 지시를 만들어내지 않는다.
페달 carry-in은 원본 이벤트에서 읽고, 구간 밖 key release/gap과 경계 밖 박자 target은
loss에서 제외한다. 입력별 JSON에 대응률, 관측량, 복구·경계 마스킹 내역을 남긴다.

학습 실행:

```sh
python scripts/train_selected_note_ae.py --prepared <전처리 출력 폴더> --out <학습 출력 폴더>
```

모든 5,082개 입력이 준비되지 않으면 학습을 거부한다. 메모리 맵 배열을 사용하고,
기본 128음표 길이로 나누되 화음·박자를 쪼개거나 마지막 조각을 버리지 않는다.
작품 ID, 원본 MIDI 해시, XML 해시, score MIDI 해시로 연결된 자료는 같은 split에
배치한다. 평가 실험의 정규화 통계는 train만 사용한다.

4개 설정(은닉 크기 32/64, 1/2층, 페달 가중치 1/1.5)을 먼저 8 epoch 비교하고,
상위 2개를 최대 30 epoch까지 이어 학습한다(patience 7). 선택 기준은 구간별 동일
비중 및 5개 복원 그룹 동일 비중의 validation Huber loss이다. 한 epoch에서는 모든
학습 구간의 모든 조각을 사용한다. G loss는 이 공통 AE 학습에서 사용하지 않는다.
test는 설정 선택 후 한 번 평가하고 평균값·zero-z·shuffled-z와 함께 보고한다.

`evaluated_best.pt`는 분리 평가용 모델이다. 별도로 선택된 설정과 epoch 수를 사용해
5,082개 전체로 처음부터 재학습한 모델을 `final_all_5082.pt`에 저장한다. 후자의
정규화 통계는 전체 자료에서 다시 추정하므로 독립 test 성능을 주장하지 않는다.
두 체크포인트에는 모델 설정, 입력 스키마, 정규화 통계, corpus 식별자와 학습 이력이
들어 있다. 이 경로의 신뢰할 수 있는 로컬 체크포인트는 실행 스크립트의 `load_model`로
복원한다. `all_5082_embeddings.npz`는 관측 음표 수로 가중한 조각별 z의 평균과
그 평균을 L2 정규화한 e를 저장한다. 복원 성능만으로 청취상 스타일 유사성을 검증한
것으로 해석하지 않는다. 실제 실행 완료 여부와 수치는 출력 `final_report.json`을 따른다.

### 2026-11-13 전체 대표구간 실행 결과

5,082개 구간, 479개 작품·악장 전체가 입력 준비를 통과했다. 기존 정렬 음표의
97.94%를 원본 MusicXML에 연결했으며, 미대응·미관측 정답은 마스킹했다.
총 1,355,869개 악보 음표 행과 13,340개 조각을 사용했고 마지막 조각도 보존했다.
평가 분할은 train 3,432 / validation 853 / test 797구간이다.

| 설정 | 은닉 크기 | AE 층 수 | 페달 복원 가중치 | 최선 epoch | Validation loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| compact | 32 | 1 | 1.0 | 8 | 0.154886 |
| standard | 64 | 1 | 1.0 | 6 | 0.154659 |
| deep (선택) | 64 | 2 | 1.0 | 16 | 0.153213 |
| pedal15 | 64 | 1 | 1.5 | 12 | 0.153848 |

모든 설정의 z는 64차원이다. 초기 8 epoch 비교 후 deep/pedal15를 연장했고,
각각 23/19 epoch에서 early stopping했다. 선택된 deep의 16 epoch 가중치로
고정 test 평가를 한 번 수행했다. Test Huber loss는 0.167938이며 train 평균값
기준선 0.407931보다 58.83% 낮다. z를 0으로 바꾸면 0.333592, 다른 구간의 z로
바꾸면 0.449176이다. 이는 z의 복원 기여를 보여주며 청취 스타일 거리 검증은 아니다.

Test의 음표 수 가중 R²는 템포 0.936, velocity 0.658, articulation 0.423,
음표 시작/끝의 sustain 값 0.409/0.454, soft 값 0.534였다. 음표 시작 시점 편차는
0.008, sustain 최소값까지의 hold/gap 지연은 0.122/0.060으로 여전히 약하다.
시험한 네 설정 중 validation 기준 최선이며 전역 최적해를 주장하지 않는다.

선택 설정으로 5,082개 전체를 16 epoch 재학습했다. 학습 loss는 0.231291에서
0.132173으로 낮아졌다. `final_all_5082.pt`와 `all_5082_embeddings.npz`를 생성했고,
후자에는 모든 구간의 z와 정규화 e가 들어 있다. 전체 재학습 모델에는 독립 test가
없으며 위 test 수치는 `evaluated_best.pt`에 해당한다. G loss는 사용하지 않았다.

산출물은 workspace의 `outputs/note_ae_selected_20261113/training_run01/`에 있다.
`final_report.json`, `test_evaluation.json`, `input_audit.json`, `loss_curves.png`와
`final_artifact_verification.json`에서 결과와 근거를 확인할 수 있다.

```sh
python scripts/verify_selected_note_ae.py --run <학습 출력 폴더> --selected-csv <원본 selected_segments.csv>
```

검증은 원본 ID 5,082개, 매 epoch 전체 구간 사용, 평가 split 간 identity 비중복,
파일 해시, 파라미터·임베딩의 유한성, e의 단위 norm을 확인했다. 재로딩한 모델로
6개 구간의 임베딩을 재계산한 최대 절대 차이는 1.79e-7이었다. 검증 과정에서
test 추론을 반복하거나 test 결과로 설정을 변경하지 않았다.
