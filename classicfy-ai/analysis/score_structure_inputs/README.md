# 악보 구조 입력 및 기존 코드 재사용 점검

이슈 #16의 첫 단계인 **기존 MusicXML 로더와 ASAP beat 정렬 코드의 재사용 범위 확인** 결과다.
기존 beat·음표 정렬은 재사용할 수 있다. 악보의 구조 정보를 보존하는 MusicXML 로더와,
그 구조를 기존 feature의 beat 위치에 연결하는 모듈은 추가해야 한다.

이번 단계에서는 파일·기호·ID·마디 대응표를 점검했다. 프레이즈 추정, 조성 긴장도 계산,
구조별 임베딩 생성 및 검색 실험은 다음 단계다.

이 문서는 1번 점검 당시의 상태를 기록한다. 이후 구현한 로더는
[MusicXML 로더 안내](../musicxml_loader/README.md)에서 확인할 수 있다.

## 점검 대상과 결과

기존 [메타데이터 분석](../metadata_style_validation/README.md)의 `performances.csv`를 그대로
사용했다. 대상은 55개 악보·570개 연주이며, 아래 비율은 ASAP 전체가 아닌 이 대상에 대한 결과다.

| 점검 | 결과 | 다음 단계에서의 의미 |
|---|---:|---|
| 원본 ASAP MusicXML 파일 | 55 / 55 | 악보 구조를 읽을 입력이 있다 |
| 기존 feature와 같은 score beat grid hash | 570 / 570 | 기존 beat index를 그대로 기준으로 사용할 수 있다 |
| ASAP aligned / nASAP robust note alignment | 각각 570 / 570 | 기존 정렬을 재계산할 필요 없이 연결 검증을 시작할 수 있다 |
| 원본 XML에 slur 시작 표기 | 41 / 55 | 참고 기호로 사용할 수 있다. slur 자체를 프레이즈 정답으로 쓰지는 않는다 |
| 원본 XML에 반복·ending 표기 | 4 / 55 | 악보에 쓰인 위치와 실제 진행 순서를 구분해야 한다 |
| 원본 XML에 음표 ID | 0 / 55 | `.match`의 음표 ID를 원본 XML에 바로 연결할 수 없다 |
| 현재 받은 nASAP XML에 음표 ID | 55 / 55 | ID를 이용한 연결 후보가 있다 |
| `.match` 악보 음 ID의 마지막 접미사 제거 후 nASAP XML ID 포함률 | 100% | 문자열 대응은 확인했다. 음높이·위치·반복 회차의 검증은 별도로 필요하다 |
| 원본 XML의 key에 mode가 명시된 악보 | 6 / 55 | 조표만으로 장조·단조를 확정하지 않고 모드 정보의 출처를 기록해야 한다 |
| 마디 대응표 길이와 score downbeat 수 일치 | 547 / 570 | 나머지 23연주는 대응표가 빠져 있다 |
| `bR`을 포함한 score grid | 1 / 55 | 불규칙 beat를 삭제하거나 일반 beat로 바꾸지 않는다 |

`robust_note_alignment=True`는 nASAP의 기존 품질 표시이며 이번 작업에서 정확도를 새로
측정한 결과가 아니다. ID 포함률도 구조의 시간 정렬 정확도를 뜻하지 않는다.

## 그대로 쓸 수 있는 코드

| 코드 | 재사용할 기능 | 구조 정보를 위해 보완할 점 |
|---|---|---|
| [`ASAPLoader`, `AsapSample`](../../src/preprocessing/asap_loader.py) | 악보·연주 MIDI 경로, 양쪽 beat/downbeat, 박자표, nASAP 정렬 경로와 품질 표시 | XML 경로·악보 조표·`downbeats_score_map`을 공개 인터페이스로 제공하지 않는다 |
| [`load_match`](../../src/preprocessing/note_alignment.py) | 음표 짝, 빠진 음·추가 음, 악보 위치와 연주 시각 | 현재 `ScoreNote`는 원문의 `마디:박`과 음 이름·변화표를 보존하지 않는다. 구조 레코드가 별도로 필요하다 |
| [`build_beat_grid`, `assign_windows`](../../src/features/beat_grid.py) | 원본 beat 경계와 유효 구간 mask, 반열린 구간 배정 | 구조도 같은 원본 index에 배정하고 결측 구간을 연결하지 않는다 |
| [`build_tempo_map`](../../src/features/articulation.py) | 정렬된 악보 음 위치를 연주 초로 연결 | 악보의 마디·반복 회차를 해석하는 모듈은 아니다 |
| [`load_score_midi`](../../src/preprocessing/atepp_alignment.py) | MIDI tick / PPQ에 따른 4분음표 단위 위치 | 음 이름·성부·보표·쉼표·slur가 사라지므로 MusicXML 로더를 대신할 수 없다 |
| [정렬 벤치마크의 변환 예제](../../scripts/benchmark_research_alignment.py) | `time_to_tick(t) / resolution`으로 악보 MIDI 초를 4분음표 단위 위치로 변환 | nASAP의 악보 좌표와 항상 단위·시작점이 같다고 가정하지 않는다 |
| [`common_pattern`](../../src/features/common_pattern.py), [`validate_embedding`](../../scripts/validate_embedding.py) | 공통 패턴 제거, 유효 mask와 `shared_indices` 보존 | 구조 분석 때문에 기존 잔차·정규화 기준이나 beat 번호를 바꾸지 않는다 |

점검 당시 저장소에는 음표·쉼표·성부·slur·조표와 악보 시간축을 함께 보존하는 공통 MusicXML
로더가 없다. Partitura는 선택 의존성인 `requirements-alignment.txt`에 `1.9.0`으로 선언되어
있지만, 점검한 `.venv`에는 설치되어 있지 않다. 이번 파일 점검은 표준 XML 파서로 수행했다.

## 연결할 때 해결해야 할 예외

### 반복 표기가 있는 4개 악보

아래 네 악보에 속한 23연주는 `downbeats_score_map`이 목록이 아니라 비유한 수치 `NaN`이다.
55개 중 반복·ending 표기가 발견된 악보 네 개와 일치한다.

- `Beethoven/Piano_Sonatas/24-1_no_2_repeat/midi_score.mid`
- `Beethoven/Piano_Sonatas/26-1/midi_score.mid`
- `Beethoven/Piano_Sonatas/31-2/midi_score.mid`
- `Beethoven/Piano_Sonatas/32-1/midi_score.mid`

따라서 기존 마디 대응표만으로 모든 악보를 연결할 수는 없다. MusicXML의 **쓰인 마디**와
**반복을 반영한 진행 위치·회차**를 분리하고, score MIDI 및 nASAP 정렬과 대조해야 한다.
반복 기호를 전부 기계적으로 펼친 결과가 해당 score MIDI와 같다고 가정하지 않는다.

### 첫 마디 번호 `0`

마디 대응표가 있는 악보 중 31개는 첫 항목 `0`만 XML의 마디 번호와 일치하지 않는다.
그 뒤의 항목들은 XML에 존재한다. 전체 연주의 대응표 항목을 합친 직접 마디 번호 일치율은
약 99.62%지만, 이는 시간 정렬의 검증 결과가 아니다.

첫 `0`이 앞부분의 경계 표시인지 번호 규칙의 차이인지 확인해야 한다. 모든 마디 번호에
일괄적으로 `+1`을 적용하지 않고, 해당 위치를 확인 전까지 별도 상태로 둔다.
`1-2`처럼 복수 마디를 가리키는 항목도 단일 마디로 축약하지 않는다.
연주별 예외는 [`review.csv`](review.csv)에 원본 번호와 함께 기록했다.

### 음표 ID와 시간 단위

nASAP XML의 `n1`과 `.match`의 `n1-1`처럼 ID를 연결할 후보가 있다. 접미사를 제거한 ID만
저장하면 서로 다른 진행 위치를 합칠 수 있으므로 원래 ID도 보존한다. 이번 점검은
문자열 포함 여부만 확인했다. 다음 단계에서 접미사와 반복 회차의 관계 및 음높이·성부·악보
위치를 대조한다.

시간축은 아래 네 종류를 구분한다.

| 좌표 | 의미 |
|---|---|
| `sample.score_beats` | 악보 MIDI 위의 초 단위 beat 경계 |
| MusicXML 또는 score MIDI의 quarter position | 4분음표 길이를 1로 둔 악보 위치 |
| 원본 ASAP beat interval index | 기존 feature 배열의 위치. 경계가 B개면 구간은 B−1개 |
| `sample.performance_beats` / 정렬된 연주 음의 시각 | 연주 MIDI 위의 초 단위 위치 |

4분음표 단위 위치를 ASAP의 beat 번호로 바로 간주하지 않는다. 박자·못갖춘마디·반복을
반영한 변환을 확인하고, 0 길이 구간과 `bR` 및 유효 mask를 보존한다. 압축된 feature 배열을
사용할 때에도 `shared_indices`로 원본 beat interval index를 복원해야 한다.

## 다음 구현 단계의 인터페이스

먼저 **악보 구조를 읽는 부분**과 **기존 beat에 연결하는 부분**을 분리한다.

1. `ScoreStructure`에 음표 ID, 음 이름·변화표, 4분음표 단위 위치·길이, 성부·보표,
   쉼표, 마디 번호·순서, 박자·조표·mode, slur와 반복 표기를 보존한다.
   원본 XML에 ID가 없으면 part·마디 순서·음표 순서로 안정적인 내부 ID를 만든다.
2. `ScoreBeatBridge`에 쓰인 마디와 진행 회차, 기존 beat interval index의 대응,
   사용한 연결 방법·coverage·실패 상태를 기록한다. 기존 정렬 모듈을 내부에서 재사용한다.
3. XML 경로·조표·마디 대응표를 제공할 별도 score metadata 인터페이스를 추가한다.
   기존 `AsapSample`과 feature 추출 결과에 영향을 주지 않는 방식으로 시작한다.

첫 연결 검증은 음악적 특징이나 검색 성적에 따라 표본을 고르지 않고,
반복 없는 악보·못갖춘마디·박자 변화 사례와 위 반복 악보 4개를 포함한다.
반복 없는 예제부터 XML↔MIDI↔ASAP beat의 음높이·위치를 확인하고 예외 처리를 확장한다.
연결이 검증된 다음에 프레이즈 후보와 조성 긴장도를 계산한다.

프레이즈·긴장도는 연주 velocity나 타이밍을 사용하지 않고 악보에서 계산한다.
그 구간에서 연주 feature가 어떻게 움직이는지를 이후에 결합해야, 작품의 구조와 연주자의
표현을 구분할 수 있다.

## 산출물과 재현

- [`works.csv`](works.csv): 원본 ASAP XML 55개의 기호·ID·마디 정보와 파일 해시
- [`nasap_works.csv`](nasap_works.csv): 대응 nASAP XML 55개의 기호·ID 정보와 파일 해시
- [`performances.csv`](performances.csv): 570연주의 grid, 정렬, 마디 대응표·ID 점검
- [`review.csv`](review.csv): 대응표 누락·마디 번호 불일치 등 검토 대상
- [`summary.json`](summary.json): 전체 수치와 설치된 라이브러리 버전
- [`provenance.json`](provenance.json): 입력 목록·annotation·코드·결과 CSV 해시

저장소 루트에서 실행한다.

```bash
.venv/bin/python classicfy-ai/scripts/audit_score_structure_inputs.py \
  --asap-root ../datasets/ASAP \
  --nasap-root ../datasets/nASAP
```

이 명령은 기존 feature 파일을 변경하지 않는다. 다른 출력 위치를 사용할 때는 `--out`을 지정한다.

재사용 대상의 기존 테스트 **30개가 모두 통과**했다.

| 테스트 | 통과 수 |
|---|---:|
| `test_asap_loader.py` | 15 |
| `test_note_alignment.py` | 5 |
| `test_beat_grid.py` | 6 |
| 통합 `test_asap_pipeline.py` | 4 |

단위 테스트의 실행 예:

```bash
PYTHONPATH=classicfy-ai/src .venv/bin/python -m unittest discover \
  -s classicfy-ai/tests/unit/preprocessing -p 'test_asap_loader.py' -v
```

다른 전처리 테스트에는 같은 경로에서 파일 패턴을 바꾸고, beat grid에는
`tests/unit/features`, 통합 테스트에는 `tests/integration`을 사용한다.
