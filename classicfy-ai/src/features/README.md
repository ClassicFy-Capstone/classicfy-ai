# Classicfy AI features — Dynamics · Pedaling · Articulation

`features`는 `preprocessing`이 읽은 **연주 MIDI**와 ASAP의 **beat 정렬 정보**로, beat 단위 연주 해석 특징을 계산한다. 이 문서는 Dynamics(음량 표현), Pedaling(페달 표현), Articulation(음을 끊고 잇는 방식) 세 특징을 **무엇을 기준으로, 어떤 방법으로** 계산했는지 처음부터 설명한다.

## 0. 입력이 무엇인지부터

Dynamics와 Pedaling의 입력은 두 가지뿐이다. Articulation은 여기에 (n)ASAP의 note 단위 정렬을 더 쓴다(4절).

1. **연주 MIDI** — `load_midi(performance_path)`가 돌려주는 `MidiData`. 드럼을 제외한 모든 파트의 음표(`notes`)와 서스테인 페달 이벤트(`pedals`, CC64)를 담고 있다.
2. **연주 beat 시각** — ASAP annotation의 `performance_beats`. 사람이 직접(또는 정교한 정렬 도구로) 매긴, "이 시각이 몇 번째 박자다"라는 초 단위 시각 목록이다.

**Dynamics와 Pedaling은 악보를 쓰지 않는다.** ASAP의 악보 MIDI는 음표 velocity가 사실상 상수(예: 전부 64)라서 세기 정보가 없고, 원본 ASAP에는 음 하나하나를 연주와 대응시키는 note 단위 정렬도 없다. 그래서 Dynamics와 Pedaling은 **연주 MIDI만으로** 계산한다. "악보 대비 얼마나 세게/많이 밟았는가"가 아니라 "이 연주 자체가 시간에 따라 얼마나 세게, 얼마나 페달을 썼는가"를 재는 특징이다.

**서스테인 페달(CC64)만 본다.** `load_midi`가 애초에 CC64 이벤트만 `PedalEvent`로 읽어 오기 때문에, 소스테누토(CC66)나 소프트 페달(CC67)은 이 특징에 반영되지 않는다.

## 1. 시간축: 왜 "beat 구간"인가

연주마다 곡 길이가 다르고 빠르기도 다르므로, 초 단위 그대로는 연주끼리 비교할 수 없다. 그래서 시간축을 **beat 번호**로 바꾼다.

beat 시각이 `[b0, b1, b2, ..., b_{B-1}]`처럼 B개 있으면, 그 **사이 간격**을 구간(window)으로 쓴다. 구간은 `T = B - 1`개이고, i번째 구간은

```
[b_i, b_{i+1})   ← b_i 포함, b_{i+1}은 포함하지 않음
```

이다. 첫 beat 이전과 마지막 beat 이후에 일어난 일(도입부의 짧은 도입, 마지막 잔향 등)은 어느 구간에도 들어가지 않는다 — 계산에서 그냥 제외된다.

예를 들어 beat가 `[10.0, 10.5, 11.2, 12.0]`이면 구간은 3개다.

| 구간 번호 | 범위 |
|---|---|
| 0 | `[10.0, 10.5)` |
| 1 | `[10.5, 11.2)` |
| 2 | `[11.2, 12.0)` |

이 배정은 `assign_windows(times, beats)` 한 함수가 전담한다(`np.searchsorted` 기반이라 음표·페달 이벤트가 몇 만 개여도 빠르다). 시각이 범위 밖이면 -1을 주고, Dynamics·Pedaling 모두 이 함수로 "몇 번째 구간 것인지"를 정한다.

**같은 작품, 다른 연주는 구간 개수가 다를 수 있다** — 연주마다 beat를 센 결과가 다르기 때문이다. 다만 ASAP에서 `aligned=True`로 표시된 연주들은 악보 beat와 연주 beat가 1:1로 대응하도록 만들어져 있어서, 실제로는 **같은 작품의 aligned 연주들은 항상 같은 T를 가진다**(1,036개 연주 전체에서 확인함). 그래서 인덱스로 바로 비교할 수 있다.

### 공통 beat-level 결과 형식

Tempo·Rubato·Dynamics·Pedaling·Articulation의 beat-level 값은 모두 `BeatSequence`로 제공한다.

```python
sequence.values  # (T,) NumPy 배열
sequence.mask    # (T,) 같은 위치의 값이 유효한지를 나타내는 boolean 배열
```

두 배열은 항상 길이가 같고 생성 후에는 수정할 수 없다. 결측 위치는 숫자 배열에서
`NaN`, mask에서 `False`로 표현한다. 단, 페달 이벤트가 없는 연주의 0은 결측이 아니라
"페달을 사용하지 않음"이므로 값은 0이고 mask는 `True`다.

## 2. Dynamics — "이 구간을 얼마나 세게 쳤는가"

### 계산 방법

1. 이 구간(`[b_i, b_{i+1})`)에서 **시작(onset)한 음표**를 모두 찾는다. 이미 울리고 있다가 이 구간까지 이어지는 음이 아니라, 이 구간 **안에서 새로 친** 음만 센다.
2. 그 음표들의 MIDI velocity(0~127, 세게 칠수록 큰 값)를 평균한다.
3. 127로 나눠 0~1 범위로 맞춘다.

```
dynamics[i] = mean(velocity of notes with b_i <= onset < b_{i+1}) / 127
```

**구간에 시작한 음이 하나도 없으면** (긴 음이 이어지기만 하거나, 쉼표 구간) 값은 `NaN`이고 `mask[i] = False`로 표시한다. 이런 구간은 이후 평균·통계에서 자동으로 빠진다.

### 예시

beat 구간이 `[10.0, 11.0)`이고, 그 안에서 음이 세 번 시작했다고 하자.

| onset 시각 | velocity |
|---|---|
| 10.2 | 60 |
| 10.6 | 80 |
| 10.9 | 100 |

```
dynamics = (60 + 80 + 100) / 3 / 127 = 80 / 127 ≈ 0.630
```

이 구간에서 화음을 쳤다면(같은 시각에 여러 음), 그 음들도 모두 평균에 들어간다 — 화음의 세기는 각 음 velocity의 평균으로 표현된다.

### 왜 이렇게 했는가

- **평균을 쓴 이유**: 최댓값을 쓰면 한 음만 세게 쳐도 구간 전체가 "포르테"로 잡힌다. 평균은 화음·꾸밈음까지 합쳐 그 순간의 전반적인 세기를 반영한다.
- **onset만 세는 이유**: 페달처럼 "지금 울리고 있는 소리의 세기"를 적분하려면 음 길이까지 고려해야 하는데, 피아노는 한 번 친 음이 감쇠할 뿐 세기가 바뀌지 않는다. 그래서 "언제 세게 쳤는가"는 onset 시점의 velocity로 충분히 표현된다.
- **127로 나눈 이유**: MIDI velocity의 정의역이 0~127이라, 그대로 나누면 다른 0~1 특징(페달 등)과 같은 눈금에서 비교·모델 입력에 넣기 쉽다.

## 3. Pedaling — "이 구간에서 페달을 얼마나 썼는가"

서스테인 페달은 Dynamics보다 한 단계 더 손이 간다. **CC64 이벤트는 "페달이 이렇게 바뀌었다"는 순간만 기록**하고, 그 값이 다음 이벤트가 올 때까지 유지된다는 점이 다르기 때문이다. 그래서 이벤트 목록을 먼저 **계단 신호(다음 이벤트까지 값이 그대로 유지되는 함수)**로 복원한 다음, 그 신호를 각 beat 구간 안에서 요약한다.

### 3-1. 계단 신호 복원

CC64 이벤트가 시각 순서로 `(t0, v0), (t1, v1), (t2, v2), ...`라면, 신호는

```
signal(t) = v_k   (t_k <= t < t_{k+1}인 가장 마지막 이벤트 k의 값)
signal(t) = 0     (t < t0, 즉 첫 이벤트 이전 — 페달을 아직 밟지 않은 것으로 본다)
```

### 3-2. 세 가지 지표

beat 구간 `[b_i, b_{i+1})` 안에서 이 신호를 **시간 가중**으로 요약해 세 값을 만든다.

| 지표 | 정의 | 의미 |
|---|---|---|
| `depth[i]` | `signal(t)/127`의 구간 내 시간 평균 | 하프 페달까지 반영한 평균 밟은 깊이 |
| `down_ratio[i]` | `signal(t) >= 64`인 시간의 비율 | 구간 중 "페달이 켜져 있었다"고 볼 수 있는 시간 비중 |
| `changes[i]` | 구간 안에서 값이 64를 넘나든(on↔off) 횟수 | 페달을 밟았다 뗀 횟수 (페달링 빈도) |

64라는 문턱값은 MIDI 규격에서 컨트롤 값 64 이상을 "on"으로 보는 관례를 따른다.

**시간 가중이 필요한 이유**: 값이 바뀐 시점이 beat 경계와 맞지 않는 경우가 대부분이다. 예를 들어 구간 `[10.0, 11.0)` 안에서 페달 값이 다음처럼 변했다고 하자.

```
10.0초 ─────────── 10.4초 ─────────── 10.8초 ─────────── 11.0초
  값 100 (깊게)        값 40 (살짝만)        값 120 (깊게)
  (0.4초 동안)          (0.4초 동안)          (0.2초 동안)
```

`depth`는 단순 평균 `(100+40+120)/3 ≈ 87`이 아니라, **각 값이 유지된 시간만큼 가중**해서

```
depth = (100/127 × 0.4 + 40/127 × 0.4 + 120/127 × 0.2) / 1.0
      ≈ (0.315 + 0.126 + 0.189)
      ≈ 0.630
```

으로 계산한다. `down_ratio`도 같은 방식으로, "값이 64 이상이었던 시간"만 더한다(위 예시라면 100과 120인 구간을 더해 0.4 + 0.2 = 0.6, 즉 60%). 이 시간 가중 적분은 `_step_integral`이 누적합으로 한 번에 계산한다(구간마다 반복문을 돌지 않는다).

`changes`는 시간이 아니라 **횟수**를 센다. 위 예시라면 100→40(off로 전환)과 40→120(on으로 전환)이 이 구간 안에서 일어났으니 2회다. 이때 "구간 진입 직전의 상태"까지 비교 대상에 넣기 때문에, 구간 경계 바로 앞에서 넘어온 전환도 올바른 구간에 정확히 배정된다.

### 3-3. 페달을 아예 쓰지 않은 연주

CC64 이벤트가 하나도 없는 연주는 (Bach 푸가 등 일부 곡에서 실제로 나타난다) 모든 구간의 `depth`, `down_ratio`, `changes`를 **0으로** 채운다. "측정 불가"가 아니라 "페달을 밟지 않았다"는 사실 자체이므로 `mask`는 `True`로 둔다(구간 폭이 0보다 크기만 하면 유효한 값으로 취급).

## 4. Articulation — "악보 음가보다 건반을 얼마나 길게/짧게 눌렀는가"

Articulation은 음과 음 사이를 **끊어서(스타카토) 치는지, 이어서(레가토) 치는지**를 나타낸다. 연주 MIDI만 보면 "이 음을 0.3초 눌렀다"는 것만 알 수 있다. 그 0.3초가 짧은 것인지는 **악보에 그 음이 몇 박으로 적혀 있는지, 그 순간 연주가 얼마나 빨랐는지**를 알아야 판단할 수 있다. 그래서 Articulation은 다섯 특징 중 유일하게 **음 하나하나를 악보 음과 짝지은 note 단위 정렬**을 입력으로 쓴다.

### 4-1. 입력: (n)ASAP의 note 단위 정렬

원본 ASAP에는 beat 단위 정렬만 있다. note 단위 정렬은 확장판인 **(n)ASAP**(`github.com/CPJKU/asap-dataset`, Peter et al., TISMIR 2023)이 연주마다 `{연주}.match` 파일로 제공한다. 불러오는 방법은 `src/preprocessing/README.md`에 정리되어 있다.

```python
loader = ASAPLoader(asap_root, note_alignment_root=nasap_root)
sample = loader.get_sample("Bach/Fugue/bwv_846/Shi05M.mid")
alignment = load_match(sample.note_alignment_path)   # NoteAlignment
feature = extract_articulation(alignment, sample.performance_beats)
```

`NoteAlignment`은 세 목록으로 이루어진다.

| 목록 | 뜻 | Articulation에서 |
|---|---|---|
| `matches` | 악보 음(`ScoreNote`)과 연주 음(`PerformedNote`)의 짝 | **이것만 쓴다** |
| `deletions` | 연주에서 빠진 악보 음 | 연주 음이 없으므로 쓰지 않는다 |
| `insertions` | 악보에 없는 연주 음(실수, 장식음의 나머지 음 등) | 악보 음가가 없으므로 쓰지 않는다 |

한 쌍에서 쓰는 값은 다음과 같다.

- 악보 음: `onset_beats`, `offset_beats`(악보 beat 단위 시작·끝 위치), `attributes`(`staccato`, `grace`, `trillmark` 등 악보 기호)
- 연주 음: `onset`, `offset`(초). `offset`은 **건반을 뗀 시각(MIDI note-off)**이다. match 5.0 형식에는 페달로 늘어난 소리의 끝(`AdjOffset`)도 있지만 쓰지 않는다. 페달 효과는 Pedaling 특징이 따로 담당하므로, Articulation은 **손가락이 건반을 누르고 있던 시간**만 잰다.

### 4-2. 정의

음 하나의 articulation 값은 다음과 같다.

```
articulation = log2( 실제로 누른 시간 / 기대 길이 )

실제로 누른 시간 = performed.offset − performed.onset                 (초)
기대 길이       = T(score.offset_beats) − T(score.onset_beats)       (초)
```

`T`는 악보 위치(beat)를 그 연주의 실제 시각(초)으로 바꾸는 함수(tempo map)이고 4-3에서 만든다. 기대 길이는 "이 연주자가 실제로 연주한 빠르기에서, 악보에 적힌 음가가 차지하는 시간"이다.

| 값 | 비율 | 뜻 |
|---|---|---|
| `+1` | 2배 | 음가의 두 배 동안 누름. 다음 음과 겹침(레가토, 손가락 페달) |
| `0` | 1배 | 악보 음가만큼 정확히 누름 |
| `−1` | 0.5배 | 음가의 절반만 누름 |
| `−2` | 0.25배 | 음가의 1/4만 누름. 뚜렷한 스타카토 |

**log2를 쓰는 이유**: 비율 그대로 쓰면 "두 배 길게"(2.0)와 "절반으로 짧게"(0.5)가 1.0에서 떨어진 거리가 다르다(+1.0과 −0.5). log2를 씌우면 +1과 −1로 대칭이 되어 평균·중앙값·거리 계산이 치우치지 않는다. Tempo 특징도 같은 이유로 log2 비율을 쓴다.

### 4-3. 악보 위치 → 연주 시각: tempo map

기대 길이를 구하려면 악보 beat를 초로 바꿔야 한다. 곡 전체의 평균 빠르기 하나로 바꾸면 rubato가 그대로 섞인다. 예를 들어 연주자가 ritardando로 그 박을 늘렸는데 손가락은 평소처럼 뗐다면, 평균 빠르기 기준으로는 "길게 눌렀다"로 잘못 잡힌다. 그래서 **그 연주 자체의 onset으로 음 위치마다 시각을 정한 tempo map**을 만든다(`build_tempo_map`).

1. **꾸밈음을 뺀다.** 꾸밈음(`grace` 속성이거나 악보 길이 0)은 본음보다 앞당겨 치므로 그 위치의 시각을 대표하지 못한다.
2. **같은 악보 위치의 음을 하나로 묶는다.** 화음은 손가락마다 수십 ms씩 어긋나므로(asynchrony), 그 위치 연주 onset들의 **중앙값**을 그 위치의 시각으로 쓴다. 예를 들어 화음 세 음을 10.00초, 10.02초, 10.04초에 쳤다면 그 위치는 10.02초다.
3. **거꾸로 가는 점을 없앤다.** 악보 위치 순서로 늘어놓으면 시각도 계속 커져야 한다. 정렬 오류로 뒤 위치가 앞 위치보다 이른 시각에 붙은 점이 있으면, **시각이 순증가하는 가장 긴 부분열(LIS)**만 남기고 나머지를 버린다. 한 점만 튀면 그 점만 빠진다. ASAP 전체에서 버려진 위치는 1.4%다(robust 정렬 1.0%, non-robust 2.6%).
4. **점 사이는 선형 보간한다.** `T(s) = np.interp(s, positions, times)`.

예를 들어 tempo map이 다음과 같고, 연주자가 beat 1→2 사이를 두 배로 늘려 쳤다고 하자.

| 악보 위치(beat) | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 연주 시각(초) | 0.0 | 0.5 | 1.5 | 2.0 |

beat 1에서 시작하는 한 박짜리 음을 1.0초 눌렀다면 기대 길이는 `T(2) − T(1) = 1.0초`이므로 articulation은 `log2(1.0/1.0) = 0`이다. 곡 평균 빠르기(0.667초/beat)를 썼다면 `log2(1.0/0.667) ≈ +0.58`, 즉 "길게 눌렀다"로 잘못 나온다. 늘어난 시간은 Tempo·Rubato가 이미 잡고 있으므로, Articulation에는 **음 사이를 얼마나 끊었는가**만 남기는 것이 목적이다.

**ASAP beat annotation을 쓰지 않은 이유**: match 파일의 악보 위치(beat)와 ASAP의 악보 beat 시각(악보 MIDI의 초)은 단위가 달라 변환 규칙이 따로 필요하다. 또 onset 기반 tempo map은 음이 있는 모든 위치(16분음표 단위까지)에 점이 있어서 beat 안에서의 빠르기 변화까지 반영한다. tempo map은 악보 위치를 직접 초로 옮기므로 match의 beat 단위(4분음표인지 8분음표인지)에 영향을 받지 않는다.

### 4-4. 계산에서 빼는 음

| 제외 사유 | 이유 | ASAP 전체 비율 |
|---|---|---|
| 꾸밈음 (`grace` 또는 악보 길이 0) | 악보 음가가 0이라 비율을 정의할 수 없다 | 0.55% |
| 장식음 기호 (`trillmark`, `tremolo`, `mordent`, `invertedmordent`, `turn`) | 연주에서는 여러 음으로 쪼개지고 첫 음 하나만 악보 음에 정렬되어, 그 음의 길이가 음가를 대표하지 못한다 | 0.03% |
| tempo map 범위 밖 (`outside_tempo_map`) | 곡 마지막 화음처럼 악보상 끝 위치가 마지막 onset 위치보다 뒤라서 외삽해야 하는 음 | 0.16% |
| 길이가 0 이하 (`non_positive`) | 기대 길이나 누른 시간이 0 이하 | 0.01% |

정렬된 음의 **99.25%**가 계산에 쓰인다. 음마다의 결과와 제외 개수는 `extract_note_articulation(alignment)`가 돌려주는 `NoteArticulation`(`onsets`, `values`, `match_indices`, `excluded_counts`)에서 확인할 수 있다. `match_indices`로 원래 악보 음을 다시 찾을 수 있어서 기호별·성부별 분석에도 쓸 수 있다.

### 4-5. beat 구간으로 모으기

다른 특징과 시간축을 맞추기 위해 음 단위 값을 연주 beat 구간으로 모은다(`extract_articulation`).

1. 각 음을 **연주 onset**으로 구간에 배정한다. Dynamics와 같은 `assign_windows(onsets, performance_beats)`를 쓴다.
2. 구간 값은 그 구간에서 시작한 음들 값의 **중앙값**이다. 계산에 쓰인 음 개수는 `note_counts`에 남긴다.
3. 시작한 음이 없는 구간은 `NaN + mask=False`다.

```
articulation[i] = median(articulation of notes with b_i <= performed.onset < b_{i+1})
```

**평균이 아니라 중앙값을 쓰는 이유**: 음 단위 값은 꼬리가 길다. 아주 짧게 뗀 음은 −3 아래까지 내려가고(전체 음의 3.6%), 페달을 밟은 채 일찍 뗀 긴 음도 큰 음수가 된다. 화음 중 한 음만 일찍 떼었을 때 구간 전체 값이 끌려가지 않도록 중앙값을 쓴다.

예를 들어 한 구간에서 네 음이 시작했고 값이 `[0.0, −1.0, 1.0, 0.0]`이면 구간 값은 0.0이다. 첫 두 음만 있는 구간이라면 −0.5다.

### 4-6. 해석할 때 알아둘 점

ASAP 전체에서 확인한 값이다. 자세한 수치는 `analysis/articulation/README.md`에 있다.

- **값은 대부분 음수다.** 음 단위 중앙값은 −0.76(음가의 약 59%)이다. 피아니스트는 보통 다음 음을 치기 전에 건반을 뗀다.
- **악보 기호와 순서가 맞다.** 중앙값이 staccatissimo −2.37 < staccato −1.95 < 기호 없음 −0.72 순서다. 정의가 의도한 것을 재고 있다는 근거다.
- **페달이 섞인다.** 페달을 밟고 있으면 건반을 일찍 떼도 소리가 이어지므로, 긴 음에서 특히 값이 작아진다. 기대 길이 0.5초 이상인 음에서 떼는 순간 페달이 on이면 중앙값이 −0.96, off면 −0.25다(424,916음). 반대로 0.15~0.4초 음은 페달 off일 때 더 짧게 끊는다(−1.76 vs −1.29, 페달 없이 치는 스타카토 악구). 이 특징은 **손가락 articulation**이고 들리는 소리의 길이가 아니다. 해석할 때는 Pedaling과 함께 봐야 한다.
- **작품이 크게 좌우한다.** 연주별 `articulation_mean` 분산의 87%를 작품이 설명한다(라벨을 섞었을 때 22%). 기대 길이가 긴 음일수록 값이 작아지는 경향(상관 −0.35)과 악보 기호가 작품마다 달라서다. 다른 특징과 마찬가지로 작품 기준 정규화가 필요하다.

## 5. 곡 전체 요약 특징

beat별 시퀀스 말고, 연주 하나를 대표하는 숫자 몇 개도 함께 만든다. Feature-only 베이스라인이나 이상치 점검에 쓴다.

| 이름 | 계산식 | 왜 이렇게 재는가 |
|---|---|---|
| `overall_score_relative_tempo` | 유효한 score-relative tempo의 중앙값 | 악보 MIDI 속도에 대한 연주 전체의 빠르기 |
| `overall_individual_tempo` | 유효한 individual tempo의 중앙값 | 같은 작품의 공통 해석에 대한 연주 전체의 속도 편차 |
| `absolute_rubato_amount` | 유효한 absolute rubato 절댓값의 중앙값 | 연주 내부의 국소적인 빠르기 변화량 |
| `relative_rubato_amount` | 유효한 relative rubato 절댓값의 중앙값 | 작품의 공통 Rubato를 제외한 연주자 고유 변화량 |
| `dynamics_mean` | 유효한 구간의 Dynamics 평균 | 곡 전체의 평균 세기 |
| `dynamics_range` | 유효한 구간의 5~95 백분위 차이 | 최댓값-최솟값 대신 백분위를 써서, 단 한 번의 실수나 이상치 음표에 흔들리지 않게 함 |
| `pedal_depth_mean` | 유효한 구간의 `depth` 평균 | 곡 전체에서 페달을 평균적으로 얼마나 깊게 밟았는가 |
| `pedal_usage` | 유효한 구간의 `down_ratio` 평균 | 곡 전체 시간 중 페달이 켜져 있던 비중 |
| `pedal_change_rate` | 유효한 구간의 `changes` 평균 (구간당 평균 횟수) | 페달을 얼마나 자주 갈아 밟았는가 |
| `articulation_mean` | 유효한 구간의 Articulation 평균 | 곡 전체에서 음가 대비 평균적으로 얼마나 길게/짧게 눌렀는가 |
| `articulation_range` | 유효한 구간의 5~95 백분위 차이 | 레가토와 스타카토를 얼마나 대비시켰는가. 극단값에 흔들리지 않게 백분위를 쓴다 |

다섯 feature 모두 `summarize_<feature>(feature)` 함수가 `dict[str, float]`를 반환한다.
요약값을 계산할 유효 구간이 없으면 해당 값은 `NaN`이다. 계산식은 feature의 음악적
의미에 따라 다르며, 인터페이스가 같다는 이유로 동일한 평균 방식을 강제하지 않는다.

**`pedal_change_rate`를 작품 사이에서 그대로 비교하면 안 된다.** 이 값은 "beat당 평균 전환 횟수"인데, beat 하나의 길이(초)는 곡 빠르기에 따라 몇 배씩 차이가 난다. 느린 곡은 beat 하나가 몇 초씩이라 그 안에 전환이 몰릴 기회가 많아지고, 실제로 ASAP 전체에서 이 값과 beat 길이의 순위상관은 0.54로 뚜렷하다. 반면 **같은 작품 안에서 연주끼리 비교**할 때는 beat 길이가 거의 같으므로 문제가 없다(같은 작품 안에서의 순위상관은 0.12로 약함). 자세한 근거는 `analysis/dynamics_pedaling/README.md`에 있다.

## 6. 결측·경계 상황 정리

| 상황 | 처리 |
|---|---|
| 구간에 시작한 음이 없음 (Dynamics, Articulation) | `NaN` + `mask=False`. 요약·비교에서 제외 |
| note 정렬 파일이 없거나 비어 있는 연주 (Articulation) | 파일이 없으면 `note_alignment_path`가 `None`이라 계산하지 않는다. 정렬된 음이 없으면 모든 구간이 `NaN + mask=False`다 |
| 꾸밈음, 장식음 기호, tempo map 밖의 음 (Articulation) | 음 단위에서 제외하고 사유별 개수를 `excluded_counts`에 남긴다 (4-4) |
| 페달 이벤트가 하나도 없는 연주 | 모든 구간 0 + `mask=True`. "안 씀"으로 취급 |
| 첫 페달 이벤트 이전 시각 | 값 0 (페달을 밟기 전) |
| beat 간격이 0인 구간 (같은 시각이 중복 기록된 경우) | `mask=False`로 제외 (0으로 나누기 방지) |
| beat 범위 밖의 음·페달 이벤트 | 계산에서 제외 (`assign_windows`가 -1을 줌) |
| beat가 2개 미만이거나 감소하는 경우 | `ValueError` (애초에 계산 불가능한 입력이므로 조용히 넘기지 않는다) |

## 7. 공개 인터페이스

| 함수 | 결과 |
|---|---|
| `extract_dynamics(performance, beats)` | `DynamicsFeature`: `sequence`(`values`는 평균 velocity / 127), `onset_counts` |
| `extract_pedaling(performance, beats)` | `PedalingFeature`: `depth`, `down_ratio`, `changes`가 각각 `BeatSequence` |
| `extract_piece_tempo_features(performances)` | `TempoFeature`: 공통·개별 tempo `BeatSequence`, interval별 상태와 원시값 |
| `extract_piece_rubato_features(tempo_features)` | `RubatoFeature`: 절대·공통·상대 rubato `BeatSequence` |
| `TempoInput.from_asap_sample(sample)` | 정렬된 `AsapSample`의 beat 정보를 Tempo 입력으로 변환 |
| `summarize_tempo(feature)` | `overall_score_relative_tempo`, `overall_individual_tempo` |
| `summarize_rubato(feature)` | `absolute_rubato_amount`, `relative_rubato_amount` |
| `summarize_dynamics(feature)` | `dynamics_mean`, `dynamics_range` |
| `summarize_pedaling(feature)` | `pedal_depth_mean`, `pedal_usage`, `pedal_change_rate` |
| `extract_articulation(alignment, beats)` | `ArticulationFeature`: `sequence`(`values`는 구간 음들의 log2 비율 중앙값), `note_counts` |
| `extract_note_articulation(alignment)` | `NoteArticulation`: 음 단위 `onsets`, `values`, `match_indices`, `excluded_counts` |
| `build_tempo_map(alignment)` | 악보 위치(beat)와 연주 시각(초)의 단조 증가 점 배열 두 개 |
| `summarize_articulation(feature)` | `articulation_mean`, `articulation_range` |
| `build_beat_grid(beats)` | 공통 beat 경계, 구간 길이, 0폭 구간을 제외하는 mask |
| `assign_windows(times, beats)` | 각 시각이 속한 구간 번호. 밖이면 -1 |
| `standardize_piece_feature(features, feature_name=..., method=None)` | 이미 분리한 작품 연주들에 공통 scale을 적용한 `NormalizedFeature` dict |
| `fit_residual_scale(sequences, method="mad")` | 유효 residual들에 fit한 `ResidualScale`, MAD/IQR/SD와 fallback 추적 |
| `standardize_sequence(sequence, scale)` | 기존 상대 sequence를 다시 공통 제거하지 않고 scale만 나눈 `BeatSequence` |

`performance`는 `load_midi`가 돌려준 `MidiData`이고 `beats`는 `AsapSample.performance_beats`다.
`alignment`는 `load_match(sample.note_alignment_path)`가 돌려준 `NoteAlignment`다.
Tempo도 같은 beat grid와 mask를 사용하며, `bR`, suspicious, 0폭 구간은 상태와 이유를
보존한 채 일반 통계에서 제외한다.

## 8. 작품 공통 패턴과 연주별 편차 분리

Dynamics, Pedaling의 `depth`·`down_ratio`·`changes`, Articulation에
`separate_piece_feature`를 동일하게 적용한다. 기존 `BeatSequence`를 보존하고 같은
작품·같은 score beat 위치에서 입력 연주들의 중앙값을 제거한다.

```text
valid[p, i]    = raw.mask[p, i] AND isfinite(raw.values[p, i]) AND score_interval[i] > 0
support[i]     = valid인 연주 수
common[i]      = median(raw.values[p, i] for valid performances)
relative[p, i] = raw.values[p, i] - common[i]
relative.mask = valid AND support >= minimum_support
```

중앙값에는 대상 연주 자신도 포함한다. `minimum_support`는 기본 2이며 2 이상의
정수로 변경할 수 있다. support 미달이면 common과 relative는 `NaN + mask=False`다.
연주가 하나뿐이면 `ValueError`다. 최소 support가 전체 연주 수보다 크면 전부 결측이다.
공통값은 **현재 입력 연주들의 중앙 패턴**이며 작품의 유일한 정답 해석이 아니다.
연주 수가 적으면 기준이 불안정하고 후보가 바뀌면 중앙값도 달라진다.
같은 작품·같은 유효 위치에서는 `relative_a - relative_b = raw_a - raw_b`가 성립한다.
중앙값 제거만으로 모든 작품 정보가 없어지는 것은 아니므로 실제 효과는 별도 분석으로 확인한다.

### 입력과 결과

`PieceFeatureInput(performance_key, piece_key, score_beats, sequence, score_beat_types=None)`에
기존 feature의 `BeatSequence`를 넣는다. `piece_key`는 작품과 반복 구조를 함께 식별해야 한다.
작품 키, sequence 길이(`T = B - 1`), score beat 위치, 제공된 score beat 종류가 같아야 한다.
beat 종류를 제공한 입력과 생략한 입력도 섞지 않는다. 위치 비교는 `rtol=atol=1e-9`다.
연주 시각은 서로 달라도 된다. score beat는 유한·비감소여야 하며 0폭 구간은 집계에서 제외한다.
중복 연주 키, 다른 작품·길이·grid·beat 종류는 `ValueError`다.

`PieceFeatureInput.from_asap_sample(sample, sequence, piece_key=None)`는 `aligned=True`와
양쪽 beat 수를 확인하고 score beat 종류도 복사한다. 기본 작품 키는 `score_path`다.
(n)ASAP에서 `.match`가 다른 작품 폴더(`_no_repeat`, `_extra_repeat` 등)로 대응되면
그 폴더 이름도 키에 포함해 원래 구조와 직접 묶이지 않게 한다. 추가 구조 구분이 필요하면
`piece_key`를 명시한다. 정렬의 음악적 정확성까지 자동으로 검증하지는 않는다.

`separate_piece_feature(inputs, minimum_support=2)`는 연주 키별 `SeparatedFeature`를 반환한다.

| 필드 | 의미 |
|---|---|
| `raw` | 기존 값과 원본 mask를 보존한 `BeatSequence` |
| `common` | 작품의 beat별 중앙값과 최소 support 충족 여부를 담은 `BeatSequence` |
| `relative` | 원본에서 중앙값을 뺀 값과 해당 연주의 유효성을 담은 `BeatSequence` |
| `common_support` | 위치별 유효 연주 수. 페달을 쓰지 않은 연주의 0도 포함 |
| `performance_key`, `piece_key`, `score_beats`, `score_beat_types`, `minimum_support` | 계산 대상·좌표·기준 추적 정보 |

원본 mask가 False이거나 값이 NaN/무한대이면 집계에서 제외하되 원본값과 mask는
`raw`에 그대로 보존한다. 원본이 결측이어도 다른 연주가 충분하면 common은 제공되며
그 연주의 relative만 결측이다. 입출력은 frozen dataclass와 독립된 읽기 전용 배열을 사용한다.
Pedaling 세 지표는 독립적으로 처리하므로 위치별 support가 서로 다를 수 있다.

### 사용 예시

```python
from collections import defaultdict
import numpy as np
from preprocessing import ASAPLoader, load_match, load_midi
from features import (
    BeatSequence, PieceFeatureInput, extract_articulation, extract_dynamics,
    extract_pedaling, separate_piece_feature,
    standardize_piece_feature,
)

loader = ASAPLoader("/path/to/ASAP", note_alignment_root="/path/to/nASAP")
target_score = (loader.root / "Bach/Fugue/bwv_848/midi_score.mid").resolve()
include_non_robust_articulation = False  # 호출자가 정렬 품질 정책을 선택한다.
groups = defaultdict(list)
for sample in loader.iter_samples(aligned_only=True):
    if sample.score_path != target_score:
        continue
    midi = load_midi(sample.performance_path)
    pedal = extract_pedaling(midi, sample.performance_beats)
    sequences = {
        "dynamics": extract_dynamics(midi, sample.performance_beats).sequence,
        "pedal_depth": pedal.depth,
        "pedal_down_ratio": pedal.down_ratio,
        "pedal_changes": pedal.changes,
    }
    if include_non_robust_articulation or sample.robust_note_alignment is True:
        if sample.note_alignment_path is not None:
            sequences["articulation"] = extract_articulation(
                load_match(sample.note_alignment_path), sample.performance_beats,
            ).sequence
        else:
            T = len(sample.performance_beats) - 1
            sequences["articulation"] = BeatSequence(np.full(T, np.nan), np.zeros(T, dtype=bool))
    for name, sequence in sequences.items():
        item = PieceFeatureInput.from_asap_sample(sample, sequence)
        groups[(item.piece_key, name)].append(item)

separated = {
    group_key: separate_piece_feature(inputs)
    for group_key, inputs in groups.items() if len(inputs) >= 2
}
# separated[(piece_key, "dynamics")][performance_key].relative.values
# separated[(piece_key, "dynamics")][performance_key].relative.mask
normalized = {
    (piece_key, name): standardize_piece_feature(list(results.values()), feature_name=name)
    for (piece_key, name), results in separated.items()
}
# normalized[(piece_key, "dynamics")][performance_key].standardized.values
# normalized[(piece_key, "dynamics")][performance_key].scale.value
```

Articulation의 non-robust 포함 여부는 **호출자가 입력 집합을 구성할 때** 정한다.
공통 제거 함수는 정렬 품질에 따라 임의로 연주를 제외하지 않는다. 정렬 파일이 없으면
전체 `NaN + mask=False`인 sequence를 넣어 결과를 추적하거나 Articulation 입력에서 제외한다.

Tempo·Rubato의 기존 `individual_tempo_sequence`·`relative_rubato_sequence`에는 이미 공통
패턴이 제거돼 있으므로 다시 넣지 않는다. 기존 계산·상태·mask는 유지한다.
이번 정규화의 대상은 Dynamics·Pedaling·Articulation이다. Tempo·Rubato는 기존 값을 그대로
사용하는 정책이며 자동 표준화하지 않는다. 필요할 때만 기존 residual에 `fit_residual_scale`과
`standardize_sequence`를 명시적으로 호출할 수 있다. 원래의 상태·mask 모델은 별도로 유지한다.
`changes`는 beat당 전환 횟수 그대로이며 beat 길이 영향이 모두 제거됐다고 가정하지 않는다.
추론 시 후보가 부족하면 상대값을 임의로 0으로 채우지 않고 별도 reference 집합이나
raw 사용 정책을 후속 단계에서 결정한다. 평가 시 reference 집합도 명시해야 한다.
작품별 scale은 작품 내 편차의 상대적 크기를, 학습 데이터 전체 scale은 작품 간 residual의
크기 차이를 보존하는 방향이다.

### Figure를 근거로 선택한 scale 정책

ASAP 1,036개 정렬 연주의 raw beat를 추출하고, 비교 가능한 D/P 172개 그룹 및 robust
Articulation 135개 그룹에서 MAD·IQR·표준편차를 비교했다. 상세 figure, CSV와 해석은
[feature 정규화 보고서](../../analysis/feature_normalization/README.md)에 있다.

| `feature_name` | 기본 `method` | 근거 |
|---|---|---|
| `dynamics` | `mad` | 모든 그룹에서 양수, IQR과 유사하며 tail에 덜 민감 |
| `articulation` | `mad` | 모든 그룹에서 양수, log2 원시값의 큰 tail이 scale을 지배하지 않게 함 |
| `pedal_depth` | `std` | 28/172 그룹에서 MAD=0, 작은 robust scale이 정상적인 on/off 편차를 과도하게 확대 |
| `pedal_down_ratio` | `std` | 63/172 그룹에서 MAD=0, 경계값과 0이 많은 분포에서 MAD/IQR 불안정 |
| `pedal_changes` | `std` | residual의 68.4%가 0, 147/172 그룹에서 MAD=0, 전체 IQR도 0 |

```text
R = 작품·feature의 모든 유효 relative beat (연주·beat를 함께 pooling)
MAD scale = 1.4826 * median(abs(R - median(R)))
IQR scale = (percentile(R, 75) - percentile(R, 25)) / 1.3489795003921634
SD scale  = std(R, ddof=0)
standardized[p, i] = relative[p, i] / scale
```

Scale 추정에서 쓰는 중앙값/평균으로 residual을 다시 center하지 않는다. clipping, log1p,
winsorization도 적용하지 않는다. Pedaling은 세 하위 지표별 독립 SD를 사용하며 SD가 robust
estimator라는 뜻은 아니다. 특히 counts의 큰 이상치는 SD에 영향을 줄 수 있다.

`standardize_piece_feature(list(separated.values()), feature_name="dynamics", method="iqr")`처럼
`mad`·`iqr`·`std`를 명시해 비교할 수 있다. 같은 작품·grid·공통값·support를 공유하는 입력만
받으며 공통 패턴을 다시 계산하지 않는다. 결과의 `raw`, `common`, `relative`, 위치별
`common_support`는 그대로 보존한다. 추가 필드는 `standardized: BeatSequence`,
`scale: ResidualScale`, `feature_name`이다. 입력과 독립된 읽기 전용 배열과 frozen dataclass를 사용한다.

`ResidualScale`에는 `value`, `requested_method`, 실제 `method`, `valid_count`, `mad`, `iqr`,
`std`가 들어간다. MAD가 `<=1e-12`이면 IQR→SD, IQR이 퇴화하면 SD 순으로 fallback한다.
이 tolerance는 부동소수점 잡음을 거르는 기준이며 분모를 작은 epsilon으로 대체하지 않는다.
모든 scale이 퇴화하면 `method="unit"`, scale=1로 residual과 유효 mask를 보존한다.
원본이 모두 같은 연주들의 residual 0도 유효하다. 유효 residual이 하나도 없으면
`method="empty"`, scale=NaN이고 표준화도 `NaN + mask=False`다.

작품별 scale은 작품 내 상대적 편차를 비교하기 위한 선택이다. 작품 간 절대 차이까지 비교하려면
학습 데이터의 residual들로 `fit_residual_scale`을 한 번 fit한 뒤 `standardize_sequence`로
재사용할 수 있다. 전체 corpus의 global scale은 진단용 비교에만 사용했다. 학습/평가에서는
공통 reference와 global scale을 학습 집합에서만 fit하고 고정해야 하며, 현재 전체 corpus
그림을 독립 평가 결과로 해석하면 안 된다. 후보 부족 시 임의의 0 대신 결측을 유지하고,
별도 reference 집합을 확보하는 것이 우선이다.

## 9. 아직 하지 않은 것 (알려진 한계)

- **Dynamics는 아직 악보 대비 비교가 없다.** 이제 (n)ASAP의 note 단위 정렬을 쓸 수 있지만, 악보 MIDI의 velocity가 상수라 "악보보다 세게/여리게"의 기준이 없다. note 단위 정렬은 현재 Articulation만 쓴다.
- **Articulation은 건반을 뗀 시각만 본다.** 페달로 이어진 소리의 길이는 반영하지 않는다(4-6). 소리 길이 기준 articulation이 필요하면 CC64로 note-off를 늦춘 값을 따로 만들어야 한다.
- **Articulation의 tempo map은 LIS 동점에 민감하다.** 순증가 최장 부분열이 여러 개면 어느 점을 버리느냐에 따라 음의 0~2.8%가 다른 값을 갖는다(검증 6곡 기준). beat 값은 중앙값이라 덜 흔들리지만, 음 단위 분석에서는 주의해야 한다.
- **Articulation은 note 정렬이 있어야 계산된다.** ATEPP나 외부 입력처럼 note 정렬이 없는 데이터에는 parangonar 같은 정렬 도구를 먼저 돌려 match 파일을 만들어야 한다.
- **소스테누토·소프트 페달은 반영되지 않는다.** 로더가 CC64만 읽기 때문이다.
- **작품 정보가 모두 제거되는 것은 아니다.** 전체 분석에서 평균값의 작품 설명력은 크게 감소했지만 표준화된 5–95% 범위에는 Dynamics 0.699, Articulation 0.757의 작품 η²가 남는다. 후보 수·악보 밀도·sequence 길이도 모델 입력에 작품 단서를 줄 수 있다. 독립 작품/작곡가 ID 예측과 추천 성능 실험은 아직 하지 않았다.
- **Pedaling changes는 beat 길이의 영향을 받는다.** 작품 공통값 제거와 scale 나눗셈은 시간당 전환율 보정이 아니다. 초당 횟수는 다른 feature이며 tempo 의존성이 추가되므로 이번에는 beat당 counts의 의미를 보존했다. 긴 마지막 beat 등의 극단값과 초당 횟수 비교는 정규화 보고서에 있다.

## 10. 테스트 · 검증

단위 테스트는 `classicfy-ai` 디렉터리에서 실행한다.

```bash
PYTHONPATH=src ../.venv/bin/python -m unittest discover -s tests -v
```

`tests/unit/features`는 구간 경계, 공통 결과 형식, 각 feature의 계산식과 결측 처리를
검증한다. `test_articulation.py`는 tempo map(화음 중앙값, 꾸밈음 제외, 역행 점 제거),
음 단위 비율, rubato 보정, 제외 규칙, 구간 중앙값을 검증한다. `test_characterization.py`는 대표 입력에 대한 네 feature의 beat-level 값과
summary를 한 번에 고정한다. 테스트 디렉터리의 역할과 실행 방법은 `tests/README.md`에
정리되어 있다.

ASAP 전체 1,036개 연주에 적용해 다른 방식으로 다시 계산한 값과 대조하고, 분포·이상치·같은 곡 여러 연주 비교까지 마친 결과는 `analysis/dynamics_pedaling/README.md`에 있다. Articulation도 같은 방식(독립 재계산, 분포·악보 기호·페달 영향, 이상치, 같은 곡 비교, 추출 과정 그림)으로 검증했고 결과는 `analysis/articulation/README.md`, 스크립트는 `scripts/validate_articulation.py`다.

`unit/features/test_common_pattern.py`는 중앙값·상대값 관계, mask/support, 유효한 0,
입력 오류, 0폭 score 구간 및 읽기 전용 독립 복사본을 검증한다.
`integration/test_common_pattern_pipeline.py`는 MIDI·ASAP annotation·(n)ASAP `.match`
fixture로 세 feature 분리, 정렬 품질 선택 및 정렬 누락을 검증한다.
실제 동일 작품 여러 연주 테스트도 있으며 데이터가 없으면 건너뛴다.
다른 위치의 데이터셋은 `ASAP_ROOT`, `NASAP_ROOT`로 지정한다.

`unit/features/test_normalization.py`는 scale 계산·fallback·결측·원본 보존·공통값 중복 제거
방지를 검증한다. 실제 ASAP/nASAP을 포함한 전체 테스트는 123개 모두 통과했다.
`scripts/validate_feature_normalization.py`는 공식 dataset의 raw 추출부터 정책 비교,
표준화, pair 차이 보존 검사, 극단 beat 원본 재추출까지 실행하고 CSV·JSON·PNG를 저장한다.

## 11. 인접 beat 변화 형태

`features.temporal_shape`는 추출·정규화 이후의 `BeatSequence`를 요약한다.

```python
from features.temporal_shape import fit_change_threshold, summarize_temporal_shape

# reference_sequences는 같은 단위로 정규화한 train 작품의 해당 채널만 사용한다.
threshold = fit_change_threshold(reference_sequences, quantile=0.95)
metrics = summarize_temporal_shape(sequence, large_threshold=threshold.value)
```

`adjacent_changes`는 원래 index에서 유효하고 finite인 두 beat의 차이만 계산하며
결측을 건너서 잇지 않는다. signed/absolute delta의 다섯 분위수, mean absolute·RMS,
고정 임계값 초과 비율, 상승·유지·하강 비율, 9가지 연속 방향 전이와 반전 비율을
반환한다. 방향 tolerance 기본값은 정규화 단위 `1e-6`이며 음악적 감지 한계는 아니다.
큰 변화는 strict `abs(delta) > threshold`이고, 방향 전이와 반전의 분모는 각각
유효한 연속 3 beat / 두 변화가 모두 non-flat인 연속 3 beat다. support count를 함께
반환하고, 분모가 없으면 NaN이다. `fit_change_threshold`는 유효 reference beat 쌍을
모아 분위수를 구하므로 긴 연주에 더 큰 가중이 들어간다.

현재 분석 입력은 작품 공통 패턴을 제거한 정규화 잔차다. Dynamics 상승을 원래
velocity 상승과 동일하게 해석하지 않는다. 정의·실제 데이터 결과·한계·재현 명령은
[변화 형태 baseline](../../analysis/temporal_shape/README.md)에 있다.

## 12. 관측된 지속 길이·반복 지표

```python
from features.temporal_duration import summarize_temporal_duration, lagged_correlation
from features.temporal_shape import adjacent_changes

lengths = summarize_temporal_duration(sequence)
value_correlation = lagged_correlation(sequence, lag=8, min_pairs=32)
delta_correlation = lagged_correlation(adjacent_changes(sequence), lag=8, min_pairs=32)
```

지속 길이는 방향별 연속 delta step 수와 양/0/음의 잔차별 연속 beat 값 수를
각각 요약한다. count·평균·중앙값·p95·최대와 경계별 censored count를 반환한다.
`observed_runs`로 원래 start/stop(exclusive), state, 좌·우 경계 정보를 얻을 수 있다.
방향/수준 tolerance 기본값은 `1e-6`이다. gap·NaN·inf는 run을 끊고, 경계에 닿는
run은 더 길 수 있다. 해당 상태가 없으면 count 0 / 길이 통계 NaN이다.

`lagged_correlation`은 각 lag에서 유효한 paired 값의 Pearson r을 계산한다.
양 끝과 그 사이 모든 entry가 유효해야 하며, 서로 다른 유효 구간을 잇지 않는다.
최소 32쌍, 양쪽 SD > 정규화 단위 `1e-12`가 기본이다. 부족/상수이면 NaN과
`insufficient_pairs` / `constant_pairs`를 반환하고, 유효하면 `ok`다.
높은 r은 지속·드리프트로도 생기므로 음악적 반복이나 추천 성공을 뜻하지 않는다.
전체 정의·그림·실제 결과는 [지속·반복 baseline](../../analysis/temporal_duration/README.md)에 있다.
