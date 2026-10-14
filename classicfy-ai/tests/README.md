# Classicfy AI tests

테스트는 검증 대상의 책임에 따라 나눈다.

`unit/embedding/test_standalone_structure.py`의 4개 테스트는 전곡 B 값을 바꿔도 단독 P/H 거리가
바뀌지 않는 조건, 좌표 수와 무관한 동일 블록 가중치, query 값에 독립적인 train scale,
미정의 입력을 채우지 않는 처리를 확인한다. 기존 구조 검색 7개와 총 11개가 skip 없이 통과했다.
[단독 표현 비교](../analysis/standalone_structure/README.md)는 기존 335연주의 순위/점수 재현과
독립 검산 2,990,332개를 포함한다. 음악적 프레이즈·조성 정답을 검증하는 테스트는 아니다.

`unit/preprocessing/test_musicxml_loader.py`는 쓰인 악보의 못갖춘마디·화음·꾸밈음·성부별
위치, divisions 변경, slur·조표·반복 보존, namespace·ID 안정성 및 잘못된 입력 처리를 검증한다.
실제 악보 110개에 대한 필드 보존 검증은 [MusicXML 로더 보고서](../analysis/musicxml_loader/README.md)에 있다.

`unit/preprocessing/test_score_beat_bridge.py`의 13개 테스트는 XML/MIDI 위치 차이, 타이·꾸밈음,
MIDI 중복·겹친 음 길이, 경계·0폭/bR·결측 index 보존 및 미지원 반복을 검증한다.
`integration/test_score_structure_pipeline.py`는 고정한 BWV 848 9연주의 public API 연결과 기존
Dynamics 원본 값/mask를 대조하며, 해당 XML/MIDI/match가 없으면 skip한다. 이번 관련 61개 테스트는
실제 입력 환경에서 skip 없이 통과했다. 전체 suite 결과와 구분한다.
[독립 계산 및 재현 안내](../analysis/score_beat_bridge/README.md)를 참고한다.

`unit/preprocessing/test_score_occurrence_bridge.py`의 7개 테스트는 반복 occurrence 분리,
원본 XML 좌표·방문 번호, 양쪽 근거로 확인한 쉼표 마디, ending 생략, 미확인 전환의 mask와
구조 경계의 인접 pair 차단을 검증한다. `integration/test_score_occurrence_pipeline.py`는
Beethoven 31-2 고정 6연주의 459개 원래 구간·230개 방문과 반복/ending 5곳을 대조한다.
반복 pilot 당시 관련 69개 테스트는 skip 없이 통과했다. [반복 연결 검증 안내](../analysis/score_occurrence_bridge/README.md)를 참고한다.

`unit/preprocessing/test_score_measure_bridge.py`의 9개 테스트는 마디별 match 박 단위,
결합 마디의 정확한 경계, 반복 못갖춘마디, 좌표 충돌 mask, 잘못된 일정 거부,
0길이 연주 구간, ID 충돌, 복수 part와 단일 onset의 미확정 단위를 검증한다.
전체 55 grid·570연주 실데이터의 별도 검산과 선택 의존성 테스트의 실행 제한은
[전체 연결 검증](../analysis/score_structure_full/VALIDATION.md)에 기록한다.

`unit/preprocessing/test_phrase_candidates.py`의 14개 테스트는 문서 순서의 slur 짝·성부 교차·
번호 재사용·모호한 짝·꾸밈음, 성부별 쉼과 전체 쉼의 구분, 고정 쉼 길이,
종결형 화음의 조표/장단조 가설 및 명시적 구간 상대 위치를 검증한다.
`integration/test_phrase_candidate_pipeline.py`의 3개 테스트는 실제 Bach 장식음 slur,
Beethoven 경계의 반복 방문별 원래 index, Mozart 도착음의 경계 위치 불확실성을 검증한다.
이번 신규 17개와 재사용 로더/마디 연결 23개, 총 40개가 skip 없이 통과했다.
[악보 후보 보고서](../analysis/phrase_candidates/README.md)에 독립 수치 98,211개 검산 결과와 재현 방법이 있다.
이 검증은 음악적 프레이즈 인식 정확도나 검색 효과의 검증이 아니다.

```text
tests/
├── unit/
│   ├── features/       # beat grid, 결과 모델, 다섯 feature와 summary 계산
│   ├── embedding/      # 시계열 구간·마스크·CNN 복원 학습 및 임베딩 추출
│   ├── preprocessing/  # ASAP annotation, MIDI 원본, (n)ASAP match 파일 로딩
│   └── tools/          # 청취 평가 음원·blind 순서·응답 분석·HTTP Range
└── integration/        # loader 출력이 feature 입력까지 이어지는 전체 흐름
```

`unit/features/test_characterization.py`는 하나의 대표 입력에 대한 Tempo, Rubato,
Dynamics, Pedaling의 beat-level 값과 summary를 함께 고정한다. 각 feature의 경계 조건과
계산 세부사항은 같은 디렉터리의 개별 테스트에서 검증한다.

전체 테스트는 `classicfy-ai` 디렉터리에서 실행한다.

```bash
PYTHONPATH=src ../.venv/bin/python -m unittest discover -s tests -v
```

`integration/test_asap_pipeline.py`의 실제 ASAP 테스트는 저장소 상위
`datasets/ASAP`을 기본 경로로 사용한다. 데이터셋이 없으면 해당 테스트만 건너뛰며,
다른 위치를 쓰려면 `ASAP_ROOT` 환경 변수를 지정한다.
(n)ASAP note 정렬을 쓰는 테스트는 `datasets/nASAP` 또는 `NASAP_ROOT`를 쓰고, 둘 중
하나라도 없으면 건너뛴다.

`unit/features/test_common_pattern.py`는 작품 공통 중앙값 제거와 원본 보존·support·grid
검증을 다룬다. `integration/test_common_pattern_pipeline.py`는 세 feature의 추출부터
공통 패턴 분리까지 MIDI와 `.match` fixture로 검증하고 실제 ASAP/(n)ASAP 테스트도 제공한다.

`unit/features/test_normalization.py`는 MAD/IQR/SD 수식, 0 scale fallback, mask 보존,
분리 결과 재사용과 작품별 scale 공유를 검증한다. 공통 pipeline 통합 테스트에서도 다섯
D/P/A 채널의 표준화값을 확인한다. 공식 데이터셋 다운로드 후 전체 123개 테스트가 skip 없이
통과했다. 데이터 없이 실행하면 실제 데이터에 의존하는 4개 테스트만 skip된다.

`unit/features/test_embedding_validation.py`는 임베딩 분석의 동점 검색 점수, 순위,
Pedaling 묶음 가중치, feature 제외, 결측 seed 및 입력 불변을 검증한다.
해당 분석 추가 후 전체 130개 테스트가 실제 데이터 환경에서 skip 없이 통과했다.

`unit/features/test_interpretation_examples.py`는 다른 작품 최근접 선택, B·C의 작품 일치,
동점 대조 선택, 표시 백분위의 동점 처리와 입력 보존을 검증한다.
해당 분석 추가 후 전체 134개 테스트가 실제 데이터 환경에서 skip 없이 통과했다.

`unit/features/test_feature_search_roles.py`는 후보 제외·동점 순위/Top 5·Pedaling 묶음 가중치,
검색에서 제외한 feature의 독립 검사·무작위 대조 50% 기준·중간 거리 대조·작품 동일 가중 집계를 검증한다.
해당 분석 추가 후 전체 142개 테스트가 실제 데이터 환경에서 skip 없이 통과했다.

`unit/tools/test_listening_study.py`는 실제 PCM 프레임 보존, 원본/잘린 ASAP 음원의 offset,
ZIP64·부분 다운로드·캐시 검증, 참가자별 순서, 동일/반복 문항, 응답 검증과 B/C 복원·동점 분모,
HTTP Range 재생·구간 이동 및 pack 밖 파일 차단을 다룬다. 합성 fixture는 자동 검증용이며
사람의 청취 결과로 저장하지 않는다.

청취 도구 추가 후 실제 데이터 환경에서 전체 163개 테스트가 skip 없이 통과했다.

`unit/embedding/test_temporal_embedding.py`는 시계열의 원래 beat 위치 보존,
작품별 split, padding·마스킹, 숨긴 정답 누출 방지, 5feature 오차 비중,
인코더 학습, checkpoint 재로드 및 같은 작품 후보 제외를 검증한다.
실행 방법과 모델 구조는 [시계열 임베딩 안내](../src/embedding/README.md)에 있다.

`unit/embedding/test_temporal_evaluation.py`는 구간 평균만 맞추는 경우와 변화 모양 복원을
구분하고, 인접 hidden beat 조건·관측값만 사용하는 baseline·7채널 묶음 순서 변경을 검증한다.

`unit/embedding/test_temporal_neighbors.py`는 다른 작품 후보의 feature 대조에서
결측 beat를 잇지 않는 변화량 계산, Pedaling의 동일 feature 비중,
최근접 후보를 제외한 대조·동점 처리, 결과와 무관한 대표 사례 선정을 검증한다.
[검색 관찰 보고서](../analysis/temporal_neighbors/README.md)는 feature 기반 대조이며 사람의 청취 정답 평가가 아니다.

`unit/features/test_asap_metadata.py`는 작품명/폴더 기반 유형 분류의 충돌·미분류 보존,
메타데이터의 정확한 연결, 소나타 악장/BWV 작품군 제외, 동일 작곡가의 B/C 후보 구성,
train-only 지원 수, 양방향 파일럿 조건 및 CSV/NPZ 벡터·identity 일치를 검증한다.
[메타데이터 점검 보고서](../analysis/metadata_style_validation/README.md)는 후보 구성 분석이며
CNN 거리 또는 청취 유사성의 성공률을 평가하지 않는다.

`unit/features/test_metadata_summary_embedding.py`는 기존 14D의 5개 음악 요소 동일 가중치,
대표값/범위 거리 분해, 대표값으로 고정한 최근접 B/C, 세 작품군 제외, 선정/비교 동점,
작품군 pair 집계와 세 점의 가중 PCA 거리 보존을 검증한다.
[평균·범위 파일럿](../analysis/metadata_summary_embedding/README.md)은 수치적 배치 성향 비교이며
청취 스타일 정답이나 추천 정확도를 검증한 결과가 아니다.

`unit/features/test_metadata_temporal_embeddings.py`는 시간·관계 좌표의 train-only scale,
A/B/C 공통 좌표와 그룹 지지, 결측을 0으로 대체하지 않는 거리, 페달의 동일 음악 요소 비중,
cosine/유클리드의 차이, 조건부 분모·작품 동일 비중 및 모든 작품군 쌍 지원 조건을 검증한다.
[보강 임베딩 비교](../analysis/metadata_temporal_embedding/README.md)는 #13의 24개 checkpoint를
재추출하고 #14 통계를 고정 후보에 적용하며, 별도 CLI로 거리와 집계를 독립 재계산한다.

`unit/features/test_metadata_information_design.py`는 추가 정보 실험 정답의 시간적 편중,
양/음의 지역 관계 강도·변동, 값 배율 불변성, 결측 비연결, 상수 관계·평균 변화 0의 미정의,
불완전 마지막 구간 제외 및 기존 인코더 split의 작품군 중복 제외를 검증한다.
[실험 설계](../analysis/metadata_information_design/README.md)는 정답·표본·CV fold를 고정한 준비 단계이며
예측기 학습·추가 설명력 수치 결과는 아직 포함하지 않는다.

`unit/embedding/test_information_probe.py`는 작품군/작품/연주의 동일 가중치,
weighted ridge의 독립 최소제곱 일치, 절편 비규제, 학습-only 메타데이터 어휘,
규제 동점, 작품 내 평균 제거, paired bootstrap, 작품/grid/split 안의 섞기,
평가 정답이 CV·예측에 영향을 주지 않는 조건을 검증한다.
[고정 설계 실행 결과](../analysis/metadata_information_results/README.md)는 전체 대조군과
독립 예측·CV·오차 재계산 및 figure를 포함한다.

`unit/embedding/test_retrieval_probe.py`는 메타데이터 동점의 비전이 처리,
작품군당 1연주 검색과 K 지원 부족, 연주/작품군 겹침의 구분, 분리 블록·결측 비연결,
검색 쪽 측정치에 평가 쪽 값이 영향을 주지 않는 조건, 상수 관계의 미정의,
스타일 백분위의 무작위 기대값과 조건부 섞기의 메타데이터·길이 구간 보존을 검증한다.
[다른 작품 검색 결과](../analysis/metadata_retrieval_results/README.md)는 별도 구간의 스타일 일치,
같은 작품 연주 대조, 후보 수 균형, 원래 #14 구현을 이용한 독립 수치 검증을 포함한다.

`test_structure_response.py`의 6개 테스트는 구간 중점·경계 이동·반복/결측 차분 차단·분리 view와 알려진 선형 반응을 검증한다. 프레이즈 후보·마디 연결·실제 후보 pipeline을 포함한 이번 관련 32개 테스트와 독립 783,603개 산술 검산을 통과했다.

`test_tonal_tension.py`의 7개 테스트는 enharmonic spelling·scale·positive overlap·쉼과 점프 reset·조성 미확정/충돌·tertile 동점을 검증한다. 독립 raw XML/scalar tension 및 두 view 반응 검산도 제공한다.

`test_structure_retrieval.py`의 7개 테스트는 search/heldout view 분리, block 차분, 조건 차원과 shift 재현, 알려진 선형 긴장 반응, constant 좌표 제외 및 작품군 cap을 검증한다. 이번 관련 54개 tests는 skip 없이 통과했다. 비교의 독립 12,467,842개 수치 검산과 9 CSV/JSON·291 배열 재현은 결과 보고서를 참조한다. 전체 suite 결과와 구분한다.
