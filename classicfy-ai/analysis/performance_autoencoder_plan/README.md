# 조건부 AE의 악보 맥락 구현

2026-10-29, `feat/#16-performance-autoencoder`. 이슈 #34의 첫 구현 단위다.
**악보 입력 S 추출·ATEPP 캐시 adapter·위치별 ScoreEncoder까지 구현했다.**
전체 PerformanceAutoencoder와 실제 ATEPP 학습은 후속이다.

후속 [MusicXML 음표별 입력·ScoreEncoder 계획](NOTE_SCORE_CONTEXT_PLAN.md)은
음표별 특징 명세, 지시 전파, 반복·붙임줄·정렬 연결 및 onset별 문맥 인코딩을 정리한다.
아래 MIDI/quarter 구현과 구분되는 구현 전 설계안이다.

```text
score MIDI + 동일 quarter 경계 → S[T,30] → batch[B,K,T,30]
                                      ↓
                             ScoreEncoder(BiGRU)
                                      ↓
                              C[B,K,T,128]
```

S는 악보에서 계산한 입력이고 C는 학습되는 가중치의 출력이다.
초기 무작위 가중치에서 나온 C를 학습된 음악 임베딩으로 저장하거나 보고하지 않는다.
C를 연주 encoder와 복원기에 함께 연결하고, 복원 손실로 공동 학습하는 설계다.
[초기 계획과 전체 AE 연결](SCORE_CONTEXT_PLAN.md)을 함께 읽는다.

## 악보 특징

[score_context.py](../../src/features/score_context.py)의 30개 채널은 아래 순서다.

| 인덱스 | 의미 |
|---|---|
| 0–11 | C부터 B까지 pitch-class별 onset 개수 |
| 12–23 | pitch-class별 양의 표기 길이 overlap 합 / interval quarter 폭 |
| 24 | 전체 onset 개수 |
| 25–27 | overlap 길이로 가중한 평균 MIDI pitch, 최저, 최고 pitch |
| 28 | interval에서 시작한 음의 전체 표기 길이 평균, onset가 없으면 0 |
| 29 | 양의 pitched overlap이 없으면 1인 silence flag |

경계는 `[q[i],q[i+1])`다. 유지음이 있으면 새 onset가 없어도 silence가 아니다.
동시음 때문에 overlap density는 1보다 클 수 있다. count/density/길이는 log1p,
pitch 세 채널은 /127, silence는 0/1을 유지한다. 별도 clipping은 하지 않는다.

MIDI tick/PPQ만 사용하므로 연주의 velocity·CC64·실제 연주 초가 S에 들어가지 않는다.
encoded end-of-track까지의 빈 구간은 쉼으로 표현하고, 그 범위 밖은 score mask=False다.
unclosed note는 실패로 알린다. 원래 ATEPP 로더의 FIFO pairing과 완전히 같은
pitch/onset/offset 중복 제거를 유지하며 zero-length note는 진단 수와 함께 제외한다.
MIDI에서 XML의 tie·grace·성부·슬러 의미를 복원했다고 해석하지 않는다.
1-quarter 집계는 bin 안 음들의 세부 순서까지 보존하지 않는다.

기존 `preprocessing.atepp_alignment.load_score_midi`는 같은 공통 로더를 재노출한다.
새 경로 `preprocessing.score_midi`는 numba 정렬 engine을 import하지 않는다.

## ATEPP 데이터 export

현재 두 기존 형식을 받는다.

1. `validate_atepp_normalization.py`가 만든 `atepp_standardized.npz`:
   `manifest.records`의 key/channel, 각 `i/score_beats`, `i/standardized_values`,
   `i/standardized_mask`를 7채널 R로 연결한다. `ATEPP-1.2`·`dual_dtw` provenance와
   정렬에 사용한 score SHA를 대조한다. 중복 source key/채널과 grid 불일치는 거부한다.
2. portable quarter NPZ: manifest list의 key/channels와 `i/q`, `i/values[Dr,T]`,
   `i/valid[Dr,T]`를 읽는다. Dr은 가변이다. 이 형식의 데이터셋 이름은 호출자가 명시하며
   원래 score fingerprint가 없는 경우 연결 CSV의 식별·해시가 기준이다.

기존 유효 R을 float32로 보존하고 mask=False 위치만 0으로 정리한다.
R의 작품 공통값이나 scale을 다시 fit하지 않는다. 원본 파일은 변경하지 않는다.
유효한 실제 0은 mask=True로 남는다. valid 위치의 NaN/Inf는 오류다.

입력 CSV `score_map.csv`:

```csv
key,work,score_path,score_sha256,split
00025,canonical-work-7,scores/work7.mid,<실제 64자리 SHA256>,unassigned
```

- `key`는 원래 연주 ID이며 앞자리 0을 유지한다. 모든 cache key가 정확히 한 번 있어야 한다.
- `score_path`는 `--score-root` 기준 상대 경로 또는 이 컴퓨터의 절대 경로다.
- `score_sha256`은 **전처리가 정렬에 사용한 score MIDI의 해시**다.
- `work`는 상위 단계에서 판본/반복판 대응까지 결정한 음악 작품 식별자다.
  파일명이나 작곡가 이름으로 자동 추정하지 않는다.
- `split`은 train/validation/test/unassigned다. 미지정은 unassigned이며 비율을 임의 배정하지 않는다.
  work-disjoint 실험의 동일 work/동일 score SHA가 여러 split을 가로지르면 오류다.

ATEPP 원본이 있는 컴퓨터에서는 기존 정규화 cache의 `provenance.source_files`에서
key→score_sha256을 얻고, DualDTW audit의 `perf_id`→`local_score`로 경로를 연결할 수 있다.
canonical work와 split은 해당 실험의 명시적 대응표로 채운다.

```bash
.venv/bin/python classicfy-ai/scripts/export_atepp_score_context.py \
  --performance-cache /data/ATEPP_dataset/features/dual_dtw/atepp_standardized.npz \
  --score-map /data/score_map.csv --score-root /data/ATEPP_dataset \
  --dataset ATEPP --out /data/score_context_run01
```

기본은 고정 transform만 적용한다. `--fit-score-scale`을 주면 명시된 train score만으로
score scale을 fit한다. 같은 악보/q가 여러 연주·grid에 반복돼도 가중치를 늘리지 않으며,
악보 균등→그 악보의 서로 다른 interval 균등으로 평균/SD를 계산한다.
silence flag는 scale하지 않고, 상수 연속 채널의 분모는 1이다.
후속 입력은 `--score-scale /data/train-export/score_scale.json`으로 같은 scale을 사용한다.
평가 score가 scale의 학습 score 목록에 있으면 work-disjoint 평가로 받지 않는다.
학습에 본 작품의 새 연주를 추론할 때는 unassigned로 export하고, 별도 known-work 평가로 구분한다.

score scale을 train-only로 만들었다고 기존 R의 corpus-fitted 정규화가 독립적으로 바뀌지는 않는다.
새 작품/새 연주 평가의 performance reference 정책은 전체 AE 학습 단계에서 따로 고정해야 한다.

## 저장 규격과 모델 연결

출력은 빈 폴더에만 쓰며 `dataset.npz`, `manifest.json`, `integrity.json`과 선택적
`score_scale.json`을 만든다. schema와 channel 의미·순서, source cache/map 해시,
score SHA/진단, original interval 수와 유효/지원 수를 보존한다.
`score SHA + q-grid + schema version`당 S를 한 번만 저장하고 record는 그 ref를 참조한다.
R의 원래 valid 값 hash와 float32 export 값 hash를 각각 저장한다.

```python
from embedding.score_context_data import read_score_context_segments, collate_score_context
from embedding.score_encoder import ScoreEncoder, ScoreEncoderConfig

manifest, recordings = read_score_context_segments(
    '/data/score_context_run01', max_segment_length=64)
eligible = [segments for segments in recordings if segments]
batch = collate_score_context(eligible[:4])
encoder = ScoreEncoder(ScoreEncoderConfig(score_dim=30, hidden_size=64))
context = encoder(batch['score'], batch['time_mask'], batch['segment_mask'])
# [B,K,T,128], weights are not trained yet
```

batch는 `score[B,K,T,Ds]`, `performance[B,K,T,Dr]`, bool `observation_mask`,
bool `time_mask[B,K,T]`, bool `segment_mask[B,K]`와 key/q 메타데이터를 반환한다.
최대 길이 64는 예시 선택이며 모델의 고정 계약이나 음악적 프레이즈 분할이 아니다.
score 미지원 구간은 끊고 R 결측 시점은 q를 유지한다. R 관측이 하나도 없는 chunk는
비유효로 제외하되 기록의 원래 q/coverage는 manifest와 배열에서 남긴다.
전체 record에 관측 구간이 없으면 빈 리스트로 반환하므로 caller가 명시적으로 제외한다.

ScoreEncoder는 실제 prefix 길이를 반영한 packed 1-layer BiGRU다.
기본 64+64 방향별 출력을 원래 연주·구간 소속에 되돌리고 모든 padding C를 0으로 둔다.
오른쪽 time padding을 늘려도 유효 C가 바뀌지 않으며 inactive segment의 NaN/Inf는 GRU 전에 제거한다.
유효 시점의 NaN/Inf, time-mask 구멍, 빈 유효 구간/연주, dtype/device 불일치는 오류다.
[PyTorch packed sequence](https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.rnn.pack_padded_sequence.html)와
[BiGRU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html)를 사용한다.

## 검증 범위와 다음 단계

[ASAP 실데이터 검증](../score_context_validation/README.md): 968연주·170악보,
584,868개 원래 구간의 R/mask/q 보존, 7,680개 raw MIDI/Fraction 산술 대조,
`[4,20,64,30]→[4,20,64,128]` 연결·padding/개별구간 일치 확인.
ATEPP 표준화 cache 형식은 합성 NPZ로 export CLI부터 검증했다.
실제 ATEPP 원본/정렬/정규화 cache는 이 기기에 없으므로 실제 ATEPP export/학습은 미실행이다.

전체 AE의 연주 encoder·decoder·손실·triplet·학습 CLI는 아직 구현하지 않았다.
test-only 작은 decoder에서 Huber gradient가 ScoreEncoder에 전달되는 것은 확인했다.
실제 AE 연결 후에는 복원기의 z 무시 여부와 실제 복원/임베딩 품질을 따로 확인해야 한다.
