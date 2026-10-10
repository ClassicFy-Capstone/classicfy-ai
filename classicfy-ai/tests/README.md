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

`unit/features/test_temporal_shape.py`는 수작업 변화량·분포·방향 전이, 결측 인접성,
빈/상수 시퀀스, strict 임계값, 기준 고정, offset 불변과 입력 보존을 검증한다.
`test_temporal_shape_analysis.py`는 원래 beat 위치 복원, test 값의 임계값 fit 제외,
중복/잘못된 index 거부와 기존 source 변경 시 cache 재사용 거부를 검증한다.
두 파일의 신규 테스트 13개가 통과했다. [변화 형태 보고서](../analysis/temporal_shape/README.md).
