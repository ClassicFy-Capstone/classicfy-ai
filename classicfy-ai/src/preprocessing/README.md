# Classicfy AI preprocessing

`preprocessing`은 MIDI 파일의 원본 이벤트, ASAP 데이터셋의 악보-연주 관계, (n)ASAP의 note 단위 정렬을 읽는다. 연주 해석 feature와 score-performance alignment를 새로 계산하지 않는다.

## 공개 인터페이스

- `load_midi(path: str | Path) -> MidiData`: MIDI 파일 하나에서 비드럼 파트의 음표와 원본 CC64 이벤트를 읽는다. 시간 단위는 초다.
- `ASAPLoader(root: str | Path)`: 전달받은 ASAP 루트의 `metadata.csv`와 `asap_annotations.json`을 읽는다.
- `get_sample(performance_key: str) -> AsapSample`: `metadata.csv`의 상대 연주 MIDI 경로를 키로 받아 악보·연주 파일 경로와 annotation을 반환한다.
- `iter_samples(aligned_only: bool = False)`: 메타데이터 순서로 샘플을 순회한다. `aligned_only=True`면 정렬되지 않은 샘플을 제외한다.
- `ASAPLoader(root, note_alignment_root=...)`: (n)ASAP 루트를 함께 주면 샘플마다 note 단위 정렬 파일 경로와 품질 표시를 채운다(아래 절).
- `load_match(path: str | Path) -> NoteAlignment`: (n)ASAP의 `.match` 파일 하나에서 악보 음과 연주 음의 짝, 빠진 악보 음, 추가된 연주 음을 읽는다.

`AsapSample`은 `score_path`, `performance_path`, `aligned`, 양쪽의 beat·downbeat 목록과 원본 박자표 정보를 제공한다. `score_beat_types`와 `performance_beat_types`에는 같은 인덱스에 있는 beat의 `b`, `db`, `bR` 종류가 저장된다. `bR`은 삭제하지 않고 원본 그대로 보존한다. `aligned=False`인 샘플은 양쪽 beat 목록의 길이가 다를 수 있으므로 인덱스로 짝짓지 않는다. `MidiData`는 `notes`, `pedals`, `duration`을 제공한다.

공개 이름은 `preprocessing` 패키지에서 가져온다. `AsapSample`, `BeatType`, `MidiData`, `NoteEvent`, `PedalEvent`, `NoteAlignment`, `ScoreNote`, `PerformedNote`도 같은 경로로 사용할 수 있다.

```python
from preprocessing import ASAPLoader, load_midi

loader = ASAPLoader("/path/to/ASAP")
sample = loader.get_sample("Bach/Fugue/bwv_846/Shi05M.mid")
score = load_midi(sample.score_path)
performance = load_midi(sample.performance_path)
```

존재하지 않는 파일은 `FileNotFoundError`, 잘못된 파일·데이터 형식은 `ValueError`, 알 수 없는 연주 키나 누락된 annotation은 `KeyError`로 처리한다. 경로는 호출자가 전달하며 저장소나 현재 작업 디렉터리의 고정 위치에 의존하지 않는다.

## note 단위 정렬: (n)ASAP

원본 ASAP(`fosfrancesco/asap-dataset`)에는 beat 단위 정렬만 있다. 확장판 **(n)ASAP**(`CPJKU/asap-dataset`, Peter et al., TISMIR 2023)은 같은 연주 MIDI에 대해 음 하나하나를 MusicXML 악보 음과 짝지은 `{연주}.match` 파일과 `robust_note_alignment` 품질 표시를 추가로 제공한다. 이 정렬은 사람이 아니라 자동 정렬 도구가 만든 것이다.

### 데이터 준비

(n)ASAP 저장소 전체는 크므로 필요한 파일만 sparse checkout으로 받으면 된다. 연주 MIDI(`*.mid`)도 함께 받으면 로더가 두 데이터셋의 연주 파일이 같은지 확인한다.

```bash
git clone --filter=blob:none --no-checkout --depth 1 https://github.com/CPJKU/asap-dataset.git nASAP
cd nASAP
git sparse-checkout set --no-cone '/metadata.csv' '/asap_annotations.json' '*.match' '*.mid'
git checkout main
```

Windows Git Bash에서는 `/metadata.csv`가 `C:/Program Files/Git/metadata.csv`로 바뀌므로 `MSYS_NO_PATHCONV=1`을 붙여 실행한다.

### 사용

```python
from preprocessing import ASAPLoader, load_match

loader = ASAPLoader("/path/to/ASAP", note_alignment_root="/path/to/nASAP")
sample = loader.get_sample("Bach/Fugue/bwv_846/Shi05M.mid")
sample.note_alignment_path    # .../nASAP/Bach/Fugue/bwv_846/Shi05M.match
sample.robust_note_alignment  # True
alignment = load_match(sample.note_alignment_path)
```

연주 키와 beat annotation은 계속 원본 ASAP 것을 쓰고, (n)ASAP에서는 match 파일과 품질 표시만 가져온다. 그래서 기존 feature의 시간축과 작품 구분은 그대로다. `note_alignment_root`를 주지 않으면 두 필드는 `None`이다.

### ASAP 연주 키 → (n)ASAP 대응 규칙

1. **같은 키가 있으면 그대로 쓴다.** 대부분의 연주가 여기에 해당한다.
2. **폴더가 옮겨진 연주.** (n)ASAP은 반복 구간을 악보와 다르게 연주한 연주를 `<작품>_no_repeat`, `<작품>_extra_repeat` 같은 형제 폴더로 옮기고 그에 맞는 악보를 따로 두었다. 같은 키가 없으면 **파일 이름이 같고, 상위 폴더가 같고, 폴더 이름이 `<원래 폴더>_`로 시작하는 항목이 정확히 하나일 때만** 대응시킨다. 후보가 둘 이상이면 잘못 짝짓지 않도록 대응시키지 않는다. 실제로 16개 연주가 이 규칙으로 대응된다. 예: `Beethoven/Piano_Sonatas/32-1/Park01.mid` → `Beethoven/Piano_Sonatas/32-1_no_repeat/Park01.match`.
3. **정렬 파일이 없으면 `None`.** (n)ASAP metadata에 적혀 있지만 저장소에 match 파일이 없는 연주가 3개 있다(`Beethoven/Piano_Sonatas/17-2/KaszoS10`, `Beethoven/Piano_Sonatas/29-4/DANILO01`, `Schubert/Impromptu_op142/1/Lisiecki10M`).
4. **연주 MIDI가 다르면 `ValueError`.** (n)ASAP 쪽에도 연주 MIDI가 있으면 `get_sample`에서 두 파일을 바이트 단위로 비교한다. match 파일의 음 ID와 시각은 그 MIDI 기준이므로, 파일이 다르면 조용히 넘기지 않는다. 현재 두 데이터셋의 대응된 연주 MIDI는 모두 같다.

`robust_note_alignment`는 (n)ASAP metadata 값(`1.0` → `True`, `0.0` → `False`, 빈 값 → `None`)이다. 1,067개 연주 중 True가 835개, False가 228개, None이 1개이고, 정렬 파일이 없는 연주가 3개다. non-robust 연주는 Ravel(22개 중 20개), Prokofiev(8개 전부), Liszt, Schubert에 몰려 있다. feature를 계산할 때 non-robust 연주를 뺄지는 호출하는 쪽에서 정한다.

### match 파일 형식과 `load_match`

match 파일은 줄마다 Prolog 사실 하나다. `load_match`는 아래 네 종류만 해석하고 `sustain`, `soft`, `meta`, `scoreprop` 등은 무시한다(페달은 연주 MIDI에서 직접 읽는다).

| 줄 | 결과 |
|---|---|
| `info(midiClockUnits,480).`, `info(midiClockRate,500000).` | tick → 초 변환값. `초 = tick × Rate / 1,000,000 / Units` |
| `snote(...)-note(...).` | `matches`에 `(ScoreNote, PerformedNote)` 한 쌍 |
| `snote(...)-deletion.` | `deletions`에 `ScoreNote` |
| `insertion-note(...).` | `insertions`에 `PerformedNote` |

(n)ASAP에는 `note(...)` 형식이 다른 두 버전이 섞여 있다(1.0.0이 381개, 5.0이 682개). 둘 다 처리한다.

```
snote(n2-1,[C,n],4,1:1,1/8,1/8,0.5000,1.0000,[v2,staff1])-note(n0,60,480,1294,36,0,0).      # 1.0.0
       └ ID, 음이름, 옥타브, 마디:박, 박 내 위치, 음가, onset(beat), offset(beat), 속성
                                                          └ ID, MIDI 음높이, onset, offset(tick), velocity, channel, track
snote(n6-1,[C,n],2,1:1,0,5/8,0.0,2.5,[])-note(n1,[C,n],2,1450,5069,5791,84).                 # 5.0
                                          └ ID, 음이름, 옥타브, onset, offset, adjusted offset(tick), velocity
```

- `ScoreNote`: `note_id`, `pitch`(음이름·변화표·옥타브를 MIDI 번호로 변환. `x`와 `##`는 겹올림, `bb`는 겹내림), `onset_beats`, `offset_beats`, `attributes`(성부 `v1`, 보표 `staff1`, `staccato`, `grace`, `trillmark` 등). `is_grace`는 `grace` 속성이거나 음가가 0 이하일 때 참이다.
- `PerformedNote`: `note_id`, `pitch`, `onset`, `offset`(초), `velocity`. `offset`은 두 버전 모두 **건반을 뗀 시각(MIDI note-off)**이다. 5.0의 adjusted offset(페달로 늘어난 소리의 끝)은 쓰지 않는다.
- match의 tick을 초로 바꾼 값은 연주 MIDI를 `load_midi`로 읽은 음의 시작·끝 시각과 같다. 검증 대상 1,036개 연주에서 정렬된 음의 평균 99.993%가 2ms 안에서 같은 음높이·시각의 MIDI 음과 일치했다(가장 낮은 연주도 97%).
- 형식이 맞지 않는 줄은 파일 경로와 그 줄을 담아 `ValueError`로 알린다.

## 테스트

`classicfy-ai` 디렉터리에서 실행한다.

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

실제 ASAP 샘플 테스트는 저장소 상위의 `datasets/ASAP`을 찾고, 데이터셋이 없으면 건너뛴다. 다른 위치에 있다면 `ASAP_ROOT` 환경 변수를 지정한다. (n)ASAP은 `datasets/nASAP` 또는 `NASAP_ROOT`에서 찾고, 없으면 note 정렬 테스트만 건너뛴다. 전체 데이터셋 순회 검증은 별도로 수행한다.

로더의 단위 테스트는 `tests/unit/preprocessing`, 로더에서 feature extraction까지의
연결 검증은 `tests/integration`에 둔다. 자세한 테스트 구조는 `tests/README.md`를
참고한다.

## 정렬 engine과 독립적인 score MIDI 읽기

`preprocessing.score_midi.load_score_midi`는 기존 ATEPP의 symbolic quarter-note reader를 제공한다. `read_score_midi`는 notes와 encoded extent·duplicate/zero-length/unclosed/orphan 진단을 함께 반환한다. `atepp_alignment.load_score_midi`도 같은 함수를 재노출한다. 새 score-context 경로는 numba를 import하지 않는다. 자세한 계약은 [악보 맥락 구현](../../analysis/performance_autoencoder_plan/README.md)에 있다.
