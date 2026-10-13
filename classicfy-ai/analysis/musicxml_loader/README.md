# MusicXML 악보 구조 로더

이슈 #16의 두 번째 단계로, **음표·마디·slur·조표를 보존하는 로더**를 구현했다.
`load_musicxml(path)`는 악보에 쓰인 순서와 위치를 `ScoreStructure`로 읽는다.
다음 프레이즈·긴장도 계산 모듈에서 사용할 악보 입력이다.

## 사용 방법과 보존 정보

```python
from preprocessing import load_musicxml

score = load_musicxml("../datasets/nASAP/Bach/Prelude/bwv_848/xml_score.musicxml")
for note in score.notes:
    if note.pitch is not None:
        print(note.xml_id, note.pitch.step, note.pitch.alter,
              float(note.onset_quarters), float(note.duration_quarters))
```

| 출력 | 보존하는 내용 |
|---|---|
| `notes` | 음 이름·변화표·옥타브, 시작 위치·길이, 성부·보표, 쉼표·꾸밈음·화음·cue 표시 |
| `measures` | part별 원본 마디 번호, 등장 순서, 못갖춘마디·non-controlling 표시, 실제 표기 길이와 박자 기준 길이 |
| `key_signatures` | 위치·보표별 조표, 명시된 mode, 비전통적 조표, 원본 key XML |
| `time_signatures` | 박자 변경 위치, 복합·가산 박자, senza-misura, 원본 time XML |
| `slurs` / `note.slurs` | 해당 음의 slur 시작·이어짐·끝, 번호와 원본 속성 |
| `barline_marks` | 반복·ending·segno·coda의 마디·위치와 원본 속성 |
| `diagnostics` | 표기 길이와 박자의 차이, 시간 표기의 보정 등 후속 연결 시 확인할 사항 |

`voice`, `staff`, `mode`가 없으면 `None`으로 둔다. 조표만 보고 장조·단조를 결정하지 않는다.
음표의 `xml_id`는 원본 값을 유지하고, 파일에 ID가 없더라도 `part:m순서:n순서` 형태의
안정적인 내부 `note_id`를 제공한다. 중복된 마디 번호도 등장 순서로 구분한다.

slur는 원본 기호로 저장한다. 시작·끝을 자동으로 짝지어 프레이즈를 만들지는 않는다.
tie와 나머지 notations XML도 보존한다. 원본 ASAP XML과 ID가 있는 nASAP XML 모두 읽는다.

## 시간 위치의 의미

시간과 길이는 `Fraction`으로 저장한 **4분음표 단위 악보 위치**다.
예를 들어 `1/3`은 4분음표의 3분의 1 길이이며, ASAP beat index나 연주 초가 아니다.
각 part에서 악보에 쓰인 순서대로 진행한다.

- 일반 음표·쉼표는 표기된 duration만큼 위치를 이동한다.
- 화음의 추가 음은 앞선 화음 기준 음과 같은 시작 위치에 놓고 위치를 이동하지 않는다.
- 꾸밈음은 별도 이벤트로 보존하며 악보 duration을 0으로 둔다.
- `backup`과 `forward`를 반영해 여러 성부·보표의 위치를 읽는다.
- divisions 변경을 반영하고, 못갖춘마디를 정규 마디 길이로 늘리지 않는다.
- 마디는 음표 끝과 forward를 포함한 **표기된 시간 범위**만큼 진행한다.
  박자보다 짧거나 긴 마디를 자동 보정하지 않고 진단 기록을 남긴다.

화음과 성부의 시간 처리는 [MusicXML chord 명세](https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/chord/)와
[backup 명세](https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/backup/)를 기준으로 구현했다.
[grace 표기](https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/grace/)의 재생 시간 속성은 보존하지만,
그 속성으로 연주 시간을 추정하지 않는다.

반복을 펼치지 않으므로 한 번 쓰인 마디는 한 번 읽는다. 반복을 반영한 score MIDI나 실제
연주 시간축과의 연결은 다음 `ScoreBeatBridge` 단계에서 검증한다. part별 마디 배치가 다르면
진단을 남기고 각 part의 좌표를 유지한다.

## 실제 악보 검증

[1번 점검](../score_structure_inputs/README.md)에서 고정한 악보 목록과 파일 해시를 사용했다.
원본 XML의 요소 수·음표 ID·voice·slur 속성을 별도 XML 읽기로 대조하고, 내부 ID 중복과
음표의 음수 위치·길이를 검사했다. 시계열 위치의 세부 규칙은 합성 악보 테스트로 검증했다.

| 대상 | 정상 로딩 | 음표·쉼표 이벤트 | 마디 | slur 표기 |
|---|---:|---:|---:|---:|
| 원본 ASAP | 55 / 55 | 179,582 | 9,280 | 14,708 |
| nASAP | 55 / 55 | 179,582 | 9,280 | 14,708 |

각 데이터셋에서 박자 길이와 표기된 마디 길이가 다른 곳 74개를 진단으로 남겼다.
이 숫자 자체가 데이터 오류 수를 뜻하지는 않는다. 못갖춘마디 표시, 분할 마디, 장식 음가 등을
후속 연결 시 함께 확인해야 한다.

두 악보의 되감기 표기도 별도로 기록했다.

- Balakirev `Islamey`의 257마디
- Beethoven `24-1_no_2_repeat`의 44마디

이곳은 반올림된 음가 뒤에 `backup`이 마디 시작을 잠시 넘어가고 바로 뒤의 `forward`로
복원되는 형태다. 원래 두 움직임을 그대로 적용하며 `compensated_backup_underflow` 진단을
남긴다. 0으로 강제 보정하지 않는다. 위치가 복원되기 전에 음표·조표 등의 이벤트가 나오거나
마디가 끝나면 `ValueError`로 처리한다.

**파일을 읽고 정보를 보존한 결과이며, XML↔MIDI↔ASAP beat 정렬 정확도를 검증한 결과는 아니다.**
이 로더는 프레이즈·긴장도를 계산하거나 기존 임베딩을 변경하지 않는다.

## 지원 범위와 테스트

압축되지 않은 `score-partwise` MusicXML 및 XML namespace를 지원한다.
압축 `.mxl`, `score-timewise`, unpitched 음은 현재 지원하지 않으며 오류로 알린다.
잘못된 duration·divisions, 복원되지 않은 되감기, 화음 기준 음 누락, 중복 XML 음표 ID 등은
파일·part·마디 위치가 포함된 `ValueError`로 처리한다.

새 단위 테스트 14개는 못갖춘마디, 화음, 꾸밈음, 여러 성부, divisions 변경, 반복·slur·조표
보존, 복합 박자, 분수 음높이, namespace, ID 안정성과 잘못된 입력 처리를 검증한다.
기존 ASAP/MIDI/match 테스트 23개와 ASAP pipeline 통합 테스트 4개도 통과했다.

전처리 폴더 전체 실행에서는 기존 ATEPP·연구 정렬 테스트 모듈 2개가 `numba` 미설치로
import에 실패했다. 해당 선택 의존성은 이번 로더에 필요하지 않으며 전체 테스트 통과로 보고하지 않는다.

저장소 루트에서 실행:

```bash
PYTHONPATH=classicfy-ai/src .venv/bin/python -m unittest discover \
  -s classicfy-ai/tests/unit/preprocessing -p 'test_musicxml_loader.py' -v

.venv/bin/python classicfy-ai/scripts/validate_musicxml_loader.py \
  --asap-root ../datasets/ASAP --nasap-root ../datasets/nASAP
```

산출물:

- [`scores.csv`](scores.csv): 파일별 요소 수·ID 보존·로딩 상태와 XML 해시
- [`diagnostics.csv`](diagnostics.csv): 마디별 진단과 원본 마디 번호
- [`summary.json`](summary.json): 110개 파일의 검증 집계
- [`provenance.json`](provenance.json): 입력 목록·구현·결과 파일의 해시

다음 작업은 이 악보 좌표를 기존 ASAP beat index에 연결하고, 음높이·위치·반복 회차를
대조하는 모듈이다. 이 연결이 확인된 구간부터 프레이즈·긴장도 계산을 적용한다.
