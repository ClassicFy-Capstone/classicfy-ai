# 시계열 연주 성향 임베딩

작품이 달라도 유사한 연주 성향을 검색하기 위한 첫 1D CNN 구현이다.
기존 feature 추출·공통 패턴 제거·scale 정책은 그대로 재사용한다.
학습에 연주 성향 유사도 라벨은 쓰지 않는다. 숨긴 beat의 feature 값을 복원하면서
구간 표현을 학습하고, 다른 작품의 검색 결과는 이후 청취로 확인한다.

## 코드 구성

| 파일 | 역할 |
|---|---|
| `data.py` | 원래 beat 위치 복원, 작품별 split, 고정 길이 구간·마스크 |
| `model.py` | 앞뒤 문맥을 읽는 CNN, 64차원 bottleneck, 복원기, feature별 오차 |
| `training.py` | 학습·검증, checkpoint 저장/로드, 구간·전곡 벡터 추출, 검색 |
| `../../scripts/train_temporal_embedding.py` | 기존 ASAP 입력 연결, 실행 CLI, CSV·그림 저장 |

## 입력과 정규화 기준

입력 순서는 `tempo`, `rubato`, `dynamics`, `articulation`, `pedal_depth`,
`pedal_down_ratio`, `pedal_changes`의 7개 채널이다.
기존 `validate_embedding.collect_groups`가 사용하는 robust note alignment, 동일
score grid, 작품/grid당 최소 5연주, 공통 유효 beat 최소 32개·50% 조건을 재사용한다.
기본적으로 55작품/grid·570연주다. 원본 MIDI가 아니라 provenance를 검사한 기존
`datasets/feature_normalization_raw.npz`에서 시작하며 Tempo/Rubato는 annotation으로 계산한다.

Tempo는 기존 `individual_tempo_sequence`, Rubato는 `relative_rubato_sequence`에
작품별 MAD scale을 적용한다. Dynamics/Articulation은 beat별 작품 중앙값 제거 후 MAD,
Pedaling 세 채널은 중앙값 제거 후 각각 SD를 사용한다. 기존 계산을 중복 적용하지 않는다.

각 작품/grid의 전체 입력 cohort를 공통값·scale reference로 사용한다.
검증/테스트 작품의 reference도 해당 작품의 연주들로 따로 계산하며, 다른 작품과 scale을
공유하거나 모델 가중치 학습에 사용하지 않는다. **후보 연주를 포함한 reference가 있는
상황**의 구현이며, reference 없이 새 MIDI 하나만 넣는 추론은 아직 지원하지 않는다.
계산 기준·cohort·scale·cache hash는 `dataset.json`에 저장한다.

기존 통계 분석의 압축된 `shared_indices`는 원래 위치에 되돌려 놓는다.
결측 beat를 삭제해서 떨어진 구간이 이웃으로 바뀌지 않게 한다. 모든 채널·후보의 공통
유효 위치만 사용하는 기존 정책을 유지하므로, 개별 연주에서만 유효한 위치도 이번 입력에서는
결측일 수 있다. 결측은 값 0과 별도 validity mask로 전달하며 복원 정답으로 쓰지 않는다.

## 작품 split과 구간

- 정렬 폴더 변형·grid가 달라도 같은 원본 score 작품은 같은 split에 넣는다.
- seed로 작품을 섞고 대략 train 70% / validation 15% / test 15%로 나눈다.
- 각 split 안에서 64 beat 구간을 stride 32로 만든다. 같은 구간의 일부가 다른 split에 들어가지 않는다.
- 마지막 구간은 끝을 포함하도록 배치한다. 64 beat 미만이면 오른쪽을 padding한다.
- padding을 포함한 64위치 중 실제 유효 beat가 50% 이상인 구간만 사용한다.
- 사용할 구간이 없는 연주는 `dataset.json`의 `skipped_keys`에 기록한다.

기본 seed 20261006에서는 train 39작품·395연주·5803구간,
validation 8작품·93연주·935구간, test 8작품·82연주·1670구간이다.

## 모델과 학습

```text
원본 구간 [batch, 7, 64]
    ↓ 약 20%의 유효 beat를 짧은 연속 구간(최대 4beat)으로 숨김
값 + validity + hidden mask [batch, 21, 64]
    ↓ 1×1 입력 projection
    ↓ 양방향 dilated CNN residual block 4개 (dilation 1, 2, 4, 8)
    ↓ 구간 순서를 유지한 flatten + Linear
구간 임베딩 [batch, 64]
    ↓ Linear 복원기 (임베딩을 우회하는 skip connection 없음)
예측 [batch, 7, 64]
```

값이 0인 유효 beat, 결측·padding, 학습용으로 숨긴 위치를 구분한다.
모델 안에서 hidden 위치를 다시 가리므로 정답 tensor를 전달해도 인코더에 정답이 노출되지 않는다.
복원 오차는 hidden AND valid 위치에서만 계산한다. 각 채널의 MSE를 구한 뒤
Tempo/Rubato/Dynamics/Articulation/Pedaling의 5개 블록에 같은 비중을 둔다.
Pedaling은 세 채널 MSE의 평균을 한 블록으로 사용한다.

AdamW로 인코더와 복원기를 함께 수정한다. 학습 마스크는 매번 달라지고 검증 마스크는
고정한다. 초기 모델(epoch 0)도 기록하며 검증 오차가 가장 작은 epoch의 `best.pt`를 저장한다.
5 epoch 동안 개선이 없으면 종료한다. test는 checkpoint 선택에 쓰지 않고 best 선택 후 한 번 평가한다.
첫 버전은 CPU로 실행한다. default batch size 32, learning rate 0.001, dropout 0.1이다.

## 실행

아래 명령은 저장소 루트에서 실행한다. Python 환경에 dependencies를 설치한다.

```bash
.venv/bin/python -m pip install -r requirements.txt
```

먼저 짧게 학습·저장·그림·벡터 추출을 확인한다.

```bash
.venv/bin/python classicfy-ai/scripts/train_temporal_embedding.py \
  --epochs 2 --out ../datasets/temporal_embedding_smoke
```

기본 설정으로 학습하려면 다른 출력 폴더를 사용한다. 기존 run을 덮어쓰지 않는다.

```bash
.venv/bin/python classicfy-ai/scripts/train_temporal_embedding.py \
  --epochs 20 --out ../datasets/temporal_embedding_run01
```

저장 모델을 다시 로드해 같은 reference cohort의 임베딩을 추출할 수 있다.
checkpoint와 함께 같은 폴더의 `dataset.json`이 필요하다.

```bash
.venv/bin/python classicfy-ai/scripts/train_temporal_embedding.py \
  --mode encode --checkpoint ../datasets/temporal_embedding_run01/best.pt \
  --out ../datasets/temporal_embedding_encoded
```

출력 기본 경로는 저장소 밖 `Classicfy/datasets/temporal_embedding`이다.
checkpoint·NPZ·원본 dataset을 커밋하지 않는다.
`--help`에서 경로·구간·모델·학습 설정을 확인할 수 있다.

## 출력과 검색

| 파일 | 내용 |
|---|---|
| `dataset.json` | 작품 split, cohort/reference, scale, 제외 기록, provenance |
| `initial.pt`, `best.pt` | 초기/검증 최선 모델 및 모델·학습 설정 |
| `history.csv`, `training.json` | 오차·설정·seed·선택 epoch·검증 zero/직선 보간 기준 |
| `loss.png` | 학습·검증 오차와 검증 직선 보간 오차 |
| `reconstruction.npz`, `reconstruction_*.png` | 첫 검증 구간의 실제·초기·best 예측 (7채널 각각 한 그림) |
| `test_reconstruction.json` | best 모델의 최종 test 복원 오차 |
| `embeddings.csv`, `embeddings.npz` | 전곡 128차원 및 NPZ의 구간 64차원 벡터 |
| `windows.csv` | 구간 벡터의 연주·원래 시작 beat 대응 |
| `neighbors.csv` | 각 연주의 다른 작품 Top 5, 코사인 유사도 및 query/candidate split |

추출 시에는 특징을 숨기지 않고 dropout을 끈 인코더를 사용한다. 각 연주의 구간 벡터
평균과 population 표준편차를 합쳐 전곡 벡터를 만들고, 검색 때 L2 정규화하여 코사인
유사도를 계산한다. 같은 작품의 모든 연주는 후보에서 제외한다. 이 목록은 전체 corpus의
검색 후보이며 학습/검증/test 연주를 모두 포함한다. `split` 열로 구분할 수 있다.

구간마다 표현이 달라도 된다. 구간 벡터를 같게 만들도록 강제하지 않으며, 전곡 요약은
구간들의 평균 성향과 다양성을 표현한다. 전곡의 구간 배치 순서는 보존하지 않는다.

## 첫 실행 결과와 범위

실제 570연주의 기본 모델로 2 epoch를 실행했다. 고정 검증 mask의 5블록 MSE는
초기 **1.575197 → 1.385568**으로 감소했다. 같은 위치의 zero 예측은 1.569728,
직선 보간은 1.746897이었다. 이는 짧은 학습이 작동하는지 확인한 결과다.
특정 feature·구간의 변화가 모두 정확히 복원되었다는 뜻은 아니다.

추가로 동일 조건에서 20 epoch를 실행했다. 최선은 18 epoch의 검증 MSE **1.302937**이며,
순서 섞기·구간 내 모양·변화량 비교 결과는 [20 epoch 흐름 복원 보고서](../../analysis/temporal_embedding/README.md)에 있다.
시간 순서를 일부 활용하지만, 급격한 beat별 변화는 아직 충분히 복원하지 못했다.
현재 모델의 [다른 작품 검색 관찰](../../analysis/temporal_neighbors/README.md)에서는
test 8작품·82연주의 최근접 후보가 대표 성향을 중심으로 가까웠으며,
변화 폭과 인접 beat 변화 크기의 일치는 약했다. feature 기반 대조이며 청취 정답 평가가 아니다.
사람의 연주 성향 비교는 아직 수행하지 않았다.
복원 오차와 검색 후보가 생성된 사실만으로 사람이 느끼는 성향 유사도를 입증하지 않는다.
이후 후보 연주를 청취해 목표에 맞게 검색되는지 확인한다.

## 검증

`classicfy-ai`에서 실행한다.

```bash
PYTHONPATH=src ../.venv/bin/python -m unittest discover -s tests/unit/embedding -v
```

구간의 결측·padding·tail, 작품 split, hidden 정답 누출 방지, 5feature 오차 비중,
가중치 학습, checkpoint 재로드 및 다른 작품 후보 제외를 검증한다.

## 구조 조건의 해석 가능한 비교

`structure_retrieval.py`는 기존 정규화 잔차의 프레이즈 가설/긴장 수준 반응과 겹치지 않는 원래 beat 구간의 평가를 제공한다. 신경망 학습 없이 train-only 거리 scale과 동일 가중 block 조합으로 비교한다. [결과와 재현](../../analysis/structure_response_comparison/README.md)에 공통 cohort·전체 축·대조 및 지원/해석 한계를 기록했다. 현재 고정 구간보다 구조 조건의 추가 이득은 확인하지 못했다.
