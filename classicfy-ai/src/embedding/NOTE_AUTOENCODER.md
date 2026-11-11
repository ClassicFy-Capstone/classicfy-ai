# 8마디 조건부 연주 시퀀스 AE

## 원본 캐시 인덱스 기반 실행


원본 캐시 입력은 `note_ae_source_index.py`를 사용하며, 기존 packed tensor 입력의
`note_ae_indexed_data.py` 및 `train_indexed_note_autoencoder.py`와 별도로 유지한다.
`--index`는 `ae-training-index` JSON을 받으며 packed corpus 디렉터리를 받지 않는다.
이 경로의 record-block 순서와 `bucket_by_length`를 동시에 지정하면 오류로 처리한다.

`prepare_note_ae_index.py`는 완료된 ATEPP 정렬 로그에서 모든 통과 연주의 full 8마디를
인덱스로 만들고, train 원본 R과 train 악보에서만 R/S scaler를 계산한다. tail·빈 구간·
복원 관측 없는 구간은 사유와 함께 제외한다. g·Triplet 선정으로 복원 구간을 줄이지 않는다.

```sh
python scripts/prepare_note_ae_index.py --preprocessing /run/preprocessing --out /run/ae_inputs
python scripts/validate_note_ae_index.py --index /run/ae_inputs/index.json --out /run/ae_inputs/validation.json
python scripts/train_note_autoencoder.py train --index /run/ae_inputs/index.json --config configs/note_ae_full_reconstruction.json --out /run/training
```

기존 collator·모델·loss를 그대로 사용하고, 전체 R을 동시에 보관하는 대신 기본 8개 연주의
검증된 S/R 캐시만 유지한다. 연주 순서를 seed로 섞고, 캐시 크기만큼의 연주 블록 안에서
구간 순서를 섞는다. 모든 적격 train 구간은 epoch마다 정확히 한 번 사용한다. 기존 전 구간
무작위 순서와 배치 순서는 다르며 인덱스 계약과 checkpoint에 이 설정을 기록한다.
이 실행 경로는 현재 복원 전용이며 `--index`와 Triplet 입력의 결합은 거부한다.

`progress.json`에는 현재 단계와 update를, `epochs.jsonl`에는 완료 epoch별 기록을 즉시 남긴다.
validation 최저 checkpoint는 `best.pt`, 완료 epoch 재개용은 `last.pt`다. 재개는 같은 인덱스·
scaler·캐시 크기·설정과 `--resume .../last.pt --out 새폴더`를 사용한다.
학습 후 선택된 모델의 test 평가는 한 번만 따로 실행할 수 있다.

```sh
python scripts/train_note_autoencoder.py evaluate --index /run/ae_inputs/index.json --checkpoint /run/training/best.pt --split test --out /run/training/test.json
```

`run_atepp_note_training.py --run /run`은 준비된 원본 경로 view·scan 결과·runtime·training_config.json에서
자동 정렬→큰 정렬 재시도→전체 입력 생성·검증→학습→선택 checkpoint의 test 평가를 이어 실행한다.
단계별 로그와 `status.json`을 남기며 실패를 숨기거나 이전 모델로 대체하지 않는다.


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
ATEPP 실데이터와 대규모 학습은 이번에 실행하지 않았다.
