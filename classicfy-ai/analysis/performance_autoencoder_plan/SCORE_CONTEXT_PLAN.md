# 조건부 AE의 악보 맥락 입력과 encoder 구현 계획

2026-10-29 KST. `feat/#16-performance-autoencoder`, 이슈 #34의 기본 모델 명세 기준.
사용자 확인: 여기서 악보 임베딩은 **AE 복원기에 조건으로 주는 악보 맥락 C**다.
이 문서는 처음 정한 구현 계획이다. 같은 날 S 추출·cache adapter·ScoreEncoder를 구현했고,
현재 API와 검증은 [구현 안내](README.md)에 기록했다. 전체 AE 구현과 학습은 후속이다.

## 목적과 출력

악보에서 결정적으로 계산한 입력 S를 ScoreEncoder가 위치별 맥락 C로 변환한다.
S는 전처리 산출물이고, C는 학습되는 encoder의 출력이다.
구간을 한 벡터로 미리 합치지 않고 연주 R의 각 시점과 대응하는 C를 유지한다.

```text
악보 S[B,K,T,Ds] → shared ScoreEncoder(BiGRU) → C[B,K,T,128]
concat(C, R_valid, M_valid) → performance BiGRU → masked mean → z[B,K,32]
concat(C, broadcast(z)) → reconstruction GRU → R_hat[B,K,T,Dr]
```

C의 128은 기본 방향별 hidden64의 연결이며 설정을 바꾸면 2*score_hidden이다.
S→C→R_hat와 S→C→연주 encoder→z→R_hat 양쪽에 복원 gradient가 흐른다.
별도 악보 검색 임베딩·score-only AE 선학습은 첫 구현에 추가하지 않는다.
최종 연주 검색용 embedding은 유효 구간 z의 평균·모집단 표준편차를 연결하고 정규화한다.

## 1. ATEPP의 같은 악보 좌표에서 S/R 만들기

- DualDTW 전처리가 실제 사용한 score MIDI와 SHA를 악보 기준으로 삼는다.
  연주를 전사한 performance MIDI를 악보로 대체하지 않는다.
- score MIDI의 tick/PPQ로 onset/duration의 quarter 좌표를 읽는다.
  velocity, CC64, 실제 연주 초는 S에 넣지 않는다.
- 첫 데이터 adapter는 기존 R의 q 경계 `[q[i],q[i+1])`에서 S를 계산한다.
  이 adapter의 첫 실험은 1-quarter bin이며 모델 자체는 임의의 T/Ds/Dr를 받는다.
- MIDI의 반복 전개/판본과 R의 alignment reference가 같은 SHA인지 확인한다.
  XML을 쓰는 경우 먼저 이 MIDI 좌표와 연결해야 한다. XML의 쓰인 위치를 그대로 붙이지 않는다.
- 기존 결측·정렬 비유효 구간과 원래 q를 유지한다. 떨어진 관측을 이어 붙이지 않는다.
  score가 불명확한 시점은 미지원으로 남기고 의미상 쉼이나 padding으로 바꾸지 않는다.

ATEPP 1.2의 공식 설명은 일부 트랙에 MusicXML을 제공한다고 기록한다.
출처: https://github.com/tangjjbetsy/ATEPP/blob/master/README.md
기존 ATEPP 전처리는 score MIDI를 사용하고, 일부 누락 MIDI를 XML에서 만들었다.
첫 구현은 두 데이터셋에 공통으로 준비 가능한 score MIDI 정보를 사용한다.

## 2. S의 첫 feature 제안: 30D

학습 성능을 보기 전 확정할 초기안이며 의미·순서·단위·정규화를 schema로 저장한다.
모델의 score_dim을 30으로 고정하지 않는다.

| 필드 | 차원 | 제안 정의 |
|---|---:|---|
| onset pitch-class counts | 12 | bin에서 시작한 악보 음의 pitch class별 개수 |
| duration-overlap pitch-class density | 12 | pitch class별 양의 표기 길이 overlap 합 / bin quarter 폭 |
| onset count | 1 | bin의 악보 note onset 수 |
| register | 3 | bin에 양의 overlap이 있는 음의 duration-weighted 평균 pitch, 최저/최고 MIDI pitch |
| onset-note written duration | 1 | bin에서 시작한 음의 원래 표기 길이 평균(quarter) |
| notated-silence flag | 1 | 양의 overlap을 가진 악보 음이 없는지 |

onset가 없으면 onset-note duration=0, pitched overlap이 없으면 register=0으로 두고
silence flag로 뜻을 구분한다. 시작 음이 없지만 유지음이 있으면 silence가 아니다.
여러 성부의 같은 음 중복 처리와 grace/tie는 원본 score MIDI의 기존 loader 계약을 먼저 감사한다.
count/density/길이 계열의 log1p 및 연속 feature의 scale은 train 악보에서만 정하고 저장한다.
동일 악보를 연주 수만큼 중복 계산해 scale 가중치를 늘리지 않는다.
조성 긴장도·프레이즈·국소 조성 추정은 첫 S에 필수로 넣지 않는다.

## 3. ScoreEncoder

- 입력 `score[B,K,T,Ds]`, `time_mask[B,K,T]`, `segment_mask[B,K]`.
- 유효 구간만 펼치고 실제 길이를 반영한 1-layer BiGRU(hidden64 each direction)를 사용한다.
- 위치별 양방향 출력 C를 `[B,K,T,128]`로 복원하며 padding 출력은 0으로 둔다.
- 실재 시점의 악보 feature는 finite여야 한다. padding의 NaN/잡음은 GRU 전에 정리한다.
- 시간 padding은 오른쪽 연속 padding이며 mask의 구멍을 sequence 길이로 오해하지 않는다.
- 같은 C를 연주 encoder와 복원기에 전달한다. 복원기에 원본 R 우회 입력은 없다.
- 첫 설정은 명세의 BiGRU이며 모델 종류/크기를 검색 결과로 교체하지 않는다.

## 4. 학습과 검증

초기 무작위 ScoreEncoder에서 나온 C는 학습된 음악 표현이 아니다.
첫 단위에서는 S 캐시와 encoder shape/mask를 검증하고, 후속 전체 AE의 유효 관측
Huber 복원 손실로 ScoreEncoder와 연주 encoder/decoder를 함께 학습한다.
triplet weight는 기본0을 유지한다. 새로운 악보 자체 손실을 임의 추가하지 않는다.

구현 검증 계획:

1. 악보 C/D/E의 짧은 합성 예제로 pitch class·지속 overlap·화음·쉼·경계와 q 정렬을 손계산한다.
2. score MIDI의 velocity/tempo 메타데이터만 바꿔도 같은 symbolic S가 나오는지 확인한다.
3. 같은 악보와 q의 다른 연주는 동일 S를 공유하고, R의 velocity/timing 변화가 S에 유입되지 않는지 확인한다.
4. 오른쪽 padding과 구간 padding을 늘려도 유효 C가 같고 출력/gradient가 finite인지 확인한다.
5. 전체 AE 연결 후 L_rec의 gradient가 ScoreEncoder에도 도달하는지 확인한다.
6. 같은 악보의 다른 연주로 z를 바꾸거나 z를 제거했을 때 복원 반응을 확인해, decoder가 C만 사용하고 z를 무시하는지 진단한다.

복원 오차 감소는 연주 임베딩의 유사성 품질과 별도다. 최종 ASAP 전이 검증에서는
같은 최종 encoder·같은 S/R 의미·같은 전처리 계약을 사용하고, 공통 작품/새 작품을 구분한다.

## 구현 단위와 데이터 인계

1. `src/features/score_context.py`: deterministic S 추출·schema와 합성 테스트.
2. `scripts/export_atepp_score_context.py`: score SHA + q-grid + schema version별 캐시,
   `recording→score_ref` 및 R/mask/segment 연결 manifest·소수 실데이터 점검.
3. `src/embedding/score_encoder.py`: 가변 shape/mask를 받는 ScoreEncoder와 padding 테스트.
4. 후속 `PerformanceAutoencoder`: 이슈 #34의 연주 encoder·decoder·loss와 연결, ATEPP 학습.

현재 이 Mac에는 ATEPP 원본 score/performance MIDI와 DualDTW 시계열 cache가 없다.
ATEPP 자료가 있는 컴퓨터에서 실제 score MIDI/동일 q의 R/mask·연결 manifest를 가져오거나,
위 exporter로 만든 S와 R을 source SHA·schema와 함께 받아야 한다.
기존 인계된 NPZ 4개는 ASAP 자료이며 ATEPP 학습 입력을 대신하지 않는다.
