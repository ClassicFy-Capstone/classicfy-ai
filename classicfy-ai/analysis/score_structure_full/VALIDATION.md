# 전체 연결 검증 결과

2026-10-13, 고정 55 grid·570연주 대상. 실패 0연주, 부분 연결 82연주다.
좌표를 확인하지 못한 1,677개 구간을 제외했고, 기존 공통 feature 유효 구간 중
추가 제외된 구간은 577개다. 일부 좌표 미검증 구간은 기존 feature mask에서도
이미 제외되어 있었기 때문에 두 수는 다르다.

| 작품/grid | 연주 수 | 좌표 제외 구간(연주 합계) | 기존 feature 중 추가 제외 |
|---|---:|---:|---:|
| Balakirev Islamey | 10 | 40 | 10 |
| Beethoven 17-1 no repeat | 11 | 132 | 22 |
| Beethoven 24-1 no 2 repeat | 5 | 20 | 20 |
| Beethoven 28-1 | 5 | 10 | 0 |
| Beethoven 32-1 | 5 | 20 | 20 |
| Chopin Ballade 4 | 12 | 72 | 12 |
| Chopin Scherzo 31 | 12 | 936 | 132 |
| Chopin Sonata 3 2nd | 9 | 27 | 9 |
| Liszt Ballade 2 | 6 | 84 | 72 |
| Scriabin Sonata 5 | 7 | 336 | 280 |

제외 사유는 `coverage.csv`의 part·visit 상태에 기록했다.
마디별 match 좌표 충돌은 35방문, MIDI anchor가 없는 방문은 637개다.
Scriabin 5의 추가 part는 음표 지원이 적으며, 그 part까지 확인한 구간만
결합 mask에 포함했다. 반복/ending 경계와 XML–MIDI 마디 길이 차이는
`break_before`로 표시한다. 음 하나의 미대응과 그 마디 전체의 좌표 미검증은 구분한다.

## 독립 검산

[verification.json](verification.json)의 별도 검산은 production bridge·MusicXML 로더·
match 로더를 import하지 않고 표준 XML 파서, raw match 정규식과 pretty_midi tick 좌표를
이용한다. 기존 7채널 값은 보관한 collector 커밋에서 다시 복원하여 전체 배열과 mask를
정확히 비교했다. 이는 기존 값 보존의 검증이며 연주 스타일 정답 또는 청취 검증은 아니다.

| 확인 | 결과 |
|---|---:|
| nASAP XML 음표/쉼표 이벤트 | 179,582 |
| 연주별 음표 occurrence | 2,027,229 |
| part별 원래 beat 연결 행 | 294,899 |
| XML/match/MIDI/구간 수치 비교 | 18,671,374 통과 |
| 기존 feature 값 및 mask 비교 | 3,454,620 통과 |
| 최대 좌표 차이 | 0.002083333334 quarter (1 MIDI tick 이내) |

294,899행은 복수 part의 beat 행을 포함한다. 중복을 제거한 연주별 원래 구간 수는
287,885개다. 최대 오차는 허용한 MIDI note-on tick 차이를 포함하며, beat index나
기존 feature 값이 바뀌었다는 의미가 아니다.

원본 raw cache SHA-256은 기존 학습 데이터의 `dataset.json`과도 일치함을 따로 확인했다:
`79ab737a08335c7fd7e9a912159de65b7c26795e1ef22e45114e5a8c6eb8ad2e`.

## 테스트 및 실행 제한

신규 `test_score_measure_bridge.py` **9개**와 기존 전처리 **57개**,
beat grid **6개**, ASAP pipeline **4개**, 반복 pipeline **1개**가 통과했다.
서로 다른 테스트 기준 합계 **77개**, skip 0이다. 신규 테스트는
마디 단위 변환, 모호한 단일 onset, 결합 마디, 잘못된 경계, 반복 못갖춘마디,
충돌 mask·원래 index, 복수 part와 0길이 구간을 검증한다.

전처리 디렉터리 전체 discover에서는 별도 선택 정렬 모듈
`test_atepp_alignment`와 `test_research_alignment` 두 개가 `numba` 미설치로
import에 실패했다. 이번 연결 모듈에는 해당 의존성이 없으며 전체 test suite 통과를
주장하지 않는다. pretty_midi의 기존 non-zero track 메타 이벤트 경고가 있었지만,
실제 score tick/PPQ 좌표의 별도 대조는 통과했다.

새 코드의 Python compile 및 `git diff --check`도 통과했다.
전체 55 grid 그림과 세 사례 그림의 레이아웃을 확인했다.

다음 단계는 유효 mask 안에서 악보 기반 프레이즈 후보와 조성 긴장도를 정의·검증하는 것이다.
노트별 예외를 추적하고, 구조 반응 비교에서는 제외 구간과 경계 이동에 대한 민감도를
함께 확인한다.
