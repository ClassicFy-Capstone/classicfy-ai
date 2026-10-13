# Classicfy AI tests

테스트는 검증 대상의 책임에 따라 나눈다.

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
