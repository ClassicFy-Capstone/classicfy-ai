# ATEPP–ASAP 전체 메타데이터 중복 조사

집계일: 2026-10-28 KST. 품질 승인·정렬 성공·다중 연주·학습 split 등의 필터를 적용하지 않았다.

## 같은 작품·악장 단위의 대응

| 항목 | ASAP | ATEPP 1.2 |
|---|---:|---:|
| 전체 작품 항목 | 222 | 1,595 |
| 전체 연주 항목 | 1,067 | 11,674 |
| 같은 작품·악장으로 대응하는 항목 | 201 / 222 (90.54%) | 202 / 1,595 (12.66%) |
| 위 공통 작품의 연주 | 953 / 1,067 (89.32%) | 3,474 / 11,674 (29.76%) |

같은 곡의 반복판·중복 ID를 합치면 위 대응은 **공통 음악 작품 200개**다. 여기서 작품은 독립 소품이나 악장 단위다. 소나타의 여러 악장을 전체 소나타 1곡으로 합치지 않았다.

- ASAP `Schumann/Toccata`와 `Schumann/Toccata_repeat`는 모두 Op.7이므로, 메타데이터 2항목이 음악 작품 1개에 대응한다.
- ATEPP는 Chopin Barcarolle Op.60이 ID `1630`, `1652`, Polonaise Op.53이 ID `1617`, `1680`으로 나뉘어 있다. 따라서 공통 음악 작품 200개에 ATEPP ID 202개가 대응한다.
- 전체 대비 비율의 분모와 분자는 원본 데이터셋의 작품 항목 기준을 유지했다. ATEPP 전체 1,595개 ID를 전부 음악학적으로 재분류·중복 제거한 것은 아니다.

## 악장 분할 차이까지 포함한 확장 범위

ASAP Beethoven sonata 31의 `3_4`는 ATEPP ID `1195`, `1193`으로, ASAP Schubert Wanderer Fantasy 전체는 ATEPP ID `896`, `894`, `897`, `895`로 분리돼 있다. 같은 음악적 내용은 존재하지만 파일·악장 단위의 1:1 대응이 아니므로 주 집계와 구분했다.

| 항목 | ASAP | ATEPP |
|---|---:|---:|
| 대응 작품 항목 | 203 / 222 (91.44%) | 208 / 1,595 (13.04%) |
| 해당 작품 범위의 연주 | 967 / 1,067 (90.63%) | 3,535 / 11,674 (30.28%) |

분할된 여러 악장의 ATEPP 연주 항목을 각각 센 수다. 동일 연주자·동일 녹음의 악장을 연결해 전체 연주로 재구성한 수가 아니다.

## 동일 연주·동일 녹음의 중복

**확인된 동일 녹음은 0개이며, 실제 중복이 0개라고 확정한 것은 아니다.** 공통 작품의 연주 수와 동일 녹음 중복 수는 다르다.

ASAP 메타데이터에는 연주자 전체 이름 열이 없고 파일명에 성·이니셜이 있다. ATEPP는 artist·앨범·YouTube URL을 제공하나, 두 데이터셋에 공통인 녹음 ID가 없다. 현재 `datasets/ATEPP`에는 메타데이터 CSV만 있어 전체 MIDI/오디오 비교도 수행하지 않았다.

보조 검사로 ATEPP 49명 전원의 성을 정규화해 ASAP 파일명 접두사와 대조했다. 후보가 생긴 이름은 Yuja Wang뿐이며 ASAP 파일 26개가 Wang 계열이었다. 이 26개와 ATEPP Yuja Wang 항목 사이에는 같은 작품 후보 쌍이 0개였다. 이 검사는 전체 이름 연결표가 아닌 파일명 대리 기준이므로 완전한 개인 식별/음원 동일성 판정으로 취급하지 않는다. 연주 중복률은 확정하지 않는다.

## 작품 매칭 기준

- ASAP 작품: 공식 권고대로 `(composer, title)`, 연주: `midi_performance` 고유 경로.
- ATEPP 작품: `composition_id`, 연주: `perf_id` 고유 ID.
- 서로 다른 작곡가 표기를 대응시키고, 작품번호(BWV, Op., Hob., D., K., S.)·소나타 번호·악장·개별 소품을 확인했다. 단순 제목 문자열 일치로 매칭하지 않았다.
- Bach는 BWV + Prelude/Fugue를 함께 비교하며, Book II 제목의 BWV 870–893 범위 표기를 개별 작품번호로 오인하지 않았다.
- 나머지는 `count_overlap.py`의 명시적 ID 대응표를 사용했다. 원본 트랙 표기와 일부 ASAP MusicXML 제목을 확인했다. Haydn은 폴더의 Hob. XVI 번호와 별도 소나타 일련번호를 혼동하지 않았다.
- 음악 작품의 동일성은 메타데이터 수준의 대응이며, 악보 판본·반복 유무·세부 음표·편집 구간의 완전 일치 검사는 아니다.
- `verification.json`에 원본 행 집계, 작곡가 일치, ID 중복, strict/expanded 집합 합산을 별도로 검산한 결과를 보존했다. 이 검산은 수치·연결 무결성을 확인하며 모든 음악적 라벨의 정확성을 독립 증명하지는 않는다.

## 작곡가별 분포

아래 공통 범위는 분할 사례 2개까지 포함한다.

| 작곡가 | ASAP 전체 작품 항목 | 같은 단위 | 분할 대응 | ASAP 공통 범위 연주 / 전체 | ATEPP 공통 범위 ID | ATEPP 공통 범위 연주 |
|---|---:|---:|---:|---:|---:|---:|
| Bach | 59 | 58 | 0 | 166 / 169 | 58 | 174 |
| Balakirev | 1 | 0 | 0 | 0 / 10 | 0 | 0 |
| Beethoven | 57 | 56 | 1 | 271 / 271 | 58 | 1907 |
| Brahms | 1 | 0 | 0 | 0 / 1 | 0 | 0 |
| Chopin | 34 | 34 | 0 | 289 / 289 | 36 | 762 |
| Debussy | 2 | 1 | 0 | 2 / 3 | 1 | 22 |
| Glinka | 1 | 0 | 0 | 0 / 2 | 0 | 0 |
| Haydn | 11 | 5 | 0 | 32 / 44 | 5 | 27 |
| Liszt | 16 | 9 | 0 | 58 / 121 | 9 | 77 |
| Mozart | 6 | 6 | 0 | 16 / 16 | 6 | 104 |
| Prokofiev | 1 | 0 | 0 | 0 / 8 | 0 | 0 |
| Rachmaninoff | 4 | 4 | 0 | 8 / 8 | 4 | 23 |
| Ravel | 4 | 4 | 0 | 22 / 22 | 4 | 67 |
| Schubert | 13 | 12 | 1 | 62 / 62 | 16 | 149 |
| Schumann | 10 | 10 | 0 | 28 / 28 | 9 | 211 |
| Scriabin | 2 | 2 | 0 | 13 / 13 | 2 | 12 |

## 파일과 재현

- `work_crosswalk.csv`: ASAP 전체 222항목의 대응 ID·판정·각 데이터셋 연주 수·원본 트랙명·악보 경로.
- `unmatched_asap_works.csv`: 대응이 없는 ASAP 19항목.
- `matched_atepp_compositions.csv`: 분할 범위를 포함한 ATEPP 대응 ID 208개.
- `matched_asap_performances.csv`, `matched_atepp_performances.csv`: 공통 범위의 원본 연주 행과 대응 범위.
- `performance_identity_audit.csv`: 성 접두사와 작품을 이용한 보조 후보 검사.
- `summary.json`: 집계 결과·분모·원본 SHA256.
- `overlap_summary.png`, `.svg`, `.pdf`: 요약 그림.

```bash
python3 classicfy-ai/analysis/dataset_compatibility/asap_atepp_overlap/count_overlap.py
MPLCONFIGDIR=/tmp/classicfy-matplotlib .venv/bin/python classicfy-ai/analysis/dataset_compatibility/asap_atepp_overlap/plot_overlap.py
```

원본:
- `datasets/ASAP/metadata.csv`와 `datasets/ASAP/README.md`.
- `datasets/ATEPP/ATEPP-metadata-1.2.csv`.
- ASAP 공식 설명: https://github.com/fosfrancesco/asap-dataset/blob/master/README.md
- ATEPP 공식 설명: https://github.com/tangjjbetsy/ATEPP/blob/master/README.md

과거 논문이나 배포본에 적힌 전체 수 대신 현재 로컬 메타데이터 파일을 직접 집계했다. ASAP의 236개 악보 파일·판본 수와 222개 작품 수를 구분했다.
