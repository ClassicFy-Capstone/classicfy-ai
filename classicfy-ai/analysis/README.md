# 연주 feature 분석

분석 그림, CSV·JSON 결과와 보고서를 이 폴더에서 주제별로 관리한다.
각 주제의 README를 먼저 열면 그림의 위치와 분석 내용을 확인할 수 있다.

| 폴더 | 분석 주제 | 시작 문서 |
|---|---|---|
| `atepp_style_diversity/` | 정규화 후 동일 작품 내 다섯 feature 다양성·다른 작곡가의 연주 피처 유사성 | [ATEPP 연주 다양성과 작곡가 간 비교](atepp_style_diversity/README.md) |
| `atepp_normalization/` | ATEPP 공통 패턴 제거·표준화·다섯 feature figure·14차원 결측/검색 분석 | [ATEPP 정규화 및 피처 분석](atepp_normalization/README.md) |
| `alignment_comparison/` | ASAP의 DualDTW·TheGlueNote beat/note 정렬 비교 및 정렬기 선택 | [정렬 비교 보고서](alignment_comparison/README.md) |
| `atepp_research/` | 선택한 연구 기반 정렬기로 ATEPP 피처 재추출·정규화·동일 추천 검증 | [재추출 검증 보고서](atepp_research/README.md) |
| `atepp/` | ATEPP 다운로드·자동 정렬·기존 피처 확장, 무결성·시대/작곡가 정보·추천 순위 안정성 검증 | [ATEPP 확장 및 추천 검증](atepp/README.md) |
| `tempo/` | 실제 한 작품의 Tempo 계산 확인·연주 차이, 파일당 그래프 하나 | [Tempo 그림 안내](tempo/README.md) |
| `rubato/` | 실제 한 작품의 전체 빠르기·공통 변화 제거와 국소 편차, 파일당 그래프 하나 | [Rubato 그림 안내](rubato/README.md) |
| `dynamics/` | 음표 velocity·강약 곡선·공통 제거·MAD 표준화와 평균/변화 폭 | [Dynamics 그림 안내](dynamics/README.md) |
| `pedaling/` | 실제 CC64와 깊이·on 시간·전환 횟수의 공통 제거 및 SD 표준화 | [Pedaling 그림 안내](pedaling/README.md) |
| `_articulation/` | 실제 음 길이·beat 중앙값·공통 제거·MAD 표준화, 파일당 그래프 하나 | [Articulation 그림 안내](_articulation/README.md) |
| `_embedding_validation/` | 다섯 feature 결합의 동일 작품 검색·결측 안정성·feature 제외 비교 | [임베딩 검증 그림 안내](_embedding_validation/README.md) |
| `temporal_embedding/` | 1D CNN 20epoch의 검증 오차·구간 내 변화 모양·관측 순서 영향 | [시계열 복원 학습 결과](temporal_embedding/README.md) |
| `temporal_neighbors/` | 현재 1D CNN의 다른 작품 검색과 대표 성향·변화 폭·beat별 변화 크기 대조 | [CNN 성향 검색 관찰](temporal_neighbors/README.md) |
| `composer_style_example/` | 베토벤·바흐 4연주를 feature로 선정해 CNN 배치와 작곡가/스타일 쌍 거리 비교 | [작곡가와 연주 스타일 figure](composer_style_example/README.md) |
| `metadata_style_validation/` | 현재 CNN 대상의 작품 유형 분류·작곡가 분포·동일 작곡가 후보 구성과 첫 파일럿 선정 | [메타데이터·후보 구성 점검](metadata_style_validation/README.md) |
| `metadata_summary_embedding/` | 쇼팽 발라드·에튀드의 대표값 선정 후 14D 평균·범위 비교, 작품군 균형 대조와 범위 기여 진단 | [평균·범위 임베딩 파일럿](metadata_summary_embedding/README.md) |
| `metadata_temporal_embedding/` | 고정 후보의 시간 통계·공동 변화와 24개 재추출 CNN/BiLSTM/Transformer 임베딩 비교 | [보강 임베딩 작품 유형 비교](metadata_temporal_embedding/README.md) |
| `metadata_same_work/` | 기존 BWV 848 세 연주의 동일 메타데이터와 14D/CNN/BiLSTM 전곡 거리 비교 | [같은 메타데이터 안의 연주 차이](metadata_same_work/README.md) |
| `metadata_information_design/` | 메타데이터 대비 시간·공동 변화 정보의 추가 설명력: 측정치·평가 표본·fold·대조군을 고정한 실험 설계 | [추가 정보 실험 설계](metadata_information_design/README.md) |
| `metadata_information_results/` | 고정한 전곡 정보 예측 실험·20회 섞기/잡음 대조·작품 내 차이·독립 수치 검증 | [추가 정보 실험 결과](metadata_information_results/README.md) |
| `metadata_retrieval_design/` | 다른 작품 Top K 검색의 메타데이터 대비 차별성·별도 구간 스타일 일치·지식 보강 실험 설계 | [검색 축 실험 설계](metadata_retrieval_design/README.md) |
| `metadata_retrieval_results/` | 메타데이터별 Top K 겹침·분리 구간 스타일 검색·같은 작품/조건부 섞기·후보 수 균형 대조 | [검색 축 실험 결과](metadata_retrieval_results/README.md) |
| `score_structure_inputs/` | MusicXML·nASAP 음표 ID·마디 대응표 점검과 기존 beat 정렬 코드 재사용 범위 | [악보 구조 입력 점검](score_structure_inputs/README.md) |
| `musicxml_loader/` | 음표·마디·slur·조표를 보존하는 MusicXML 로더와 실제 악보 110개 검증 | [MusicXML 로더 안내](musicxml_loader/README.md) |
| `score_beat_bridge/` | BWV 848 9연주의 XML·score MIDI·원래 beat 연결 및 기존 Dynamics 값 보존 검증 | [악보와 원래 beat 연결](score_beat_bridge/README.md) |
| `score_occurrence_bridge/` | Beethoven 31-2 6연주의 반복 회차·첫/둘째 ending·쉼표 마디와 원래 beat 연결 | [반복 구조와 원래 beat 연결](score_occurrence_bridge/README.md) |
| `score_structure_full/` | 전체 55 grid·570연주의 반복·결합 마디·복수 part 연결과 기존 7채널 값·mask 보존 | [전체 악보 구조 연결](score_structure_full/README.md) |
| `phrase_candidates/` | 55악보의 slur·쉼·종결형 화음 경계 후보와 3작품의 첫 악보 검수·원래 beat 방문별 연결 | [후보 추출과 검수](phrase_candidates/README.md), [다른 기기에서 이어가기](phrase_candidates/HANDOFF.md) |
| `phrase_response/` | 55악보의 구간 가설·570연주 반응과 경계 민감도·독립 계산 | [프레이즈 반응 계산](phrase_response/README.md) |
| `tonal_tension/` | 55악보의 Spiral Array 3지표·조성 가설/불확실성과 잔차 반응 | [조성 긴장도 계산](tonal_tension/README.md) |
| `structure_response_comparison/` | 같은 표본의 프레이즈·긴장도 Top 5 비교·별도 구간 평가와 구조 섞기/후보 수 대조 | [구조 반응의 유용성 비교](structure_response_comparison/README.md) |
| `structure_conditioned_style_plan/` | 16번 작업 완료 근거·이슈별 상태와 원래 beat 연결부터 구조 기반 검색까지의 구현 계획 | [16번 진행 추적 및 다음 계획](structure_conditioned_style_plan/README.md) |
| `_interpretation_embedding/` | 같은 작품 3연주의 차이·MIDI 원자료와 다른 작품 A·B·C 유사 특징 검색 | [연주 차이와 검색 사례](_interpretation_embedding/README.md) |
| `_feature_search_roles/` | 다른 작품 검색에서 feature별 후보/순위 변화와 검색에 안 쓴 feature의 일치 검사 | [feature별 검색 역할](_feature_search_roles/README.md) |
| `_listening_evaluation/` | 3인 청취 평가의 연주 선정·발췌·비공개 거리와 준비 기록 | [진행자 자료](_listening_evaluation/README.md) · [참가자 도구](../tools/listening_evaluation/README.md) |
| `dynamics_pedaling/` | 원본 Dynamics·Pedaling 추출 검증과 작품별 차이 | [원본 feature 분석 보고서](dynamics_pedaling/README.md) |
| `articulation/` | 원본 Articulation 추출 검증, 악보 기호·페달 영향 | [Articulation 분석 보고서](articulation/README.md) |
| `feature_normalization/` | 작품 공통 제거 및 MAD·IQR·SD 비교·적용 | [정규화 분석 보고서](feature_normalization/README.md) |
| `custom_comparisons/` | 선택한 작품의 여러 연주 곡선·히트맵 비교 | [선택 작품 비교 안내](custom_comparisons/README.md) |

```text
classicfy-ai/analysis/
├── README.md
├── tempo/
├── rubato/
├── dynamics/
├── pedaling/
│   ├── depth/
│   ├── down_ratio/
│   └── changes/
├── dynamics_pedaling/
├── articulation/
├── _articulation/
├── _embedding_validation/
├── temporal_embedding/
├── temporal_neighbors/
├── composer_style_example/
├── metadata_style_validation/
├── metadata_summary_embedding/
├── metadata_temporal_embedding/
├── metadata_same_work/
├── metadata_information_design/
├── metadata_information_results/
├── metadata_retrieval_design/
├── metadata_retrieval_results/
├── score_structure_inputs/
├── musicxml_loader/
├── score_beat_bridge/
├── score_occurrence_bridge/
├── score_structure_full/
├── structure_conditioned_style_plan/
├── _interpretation_embedding/
│   ├── same_work/
│   ├── cross_work/
│   └── source_midi/
├── _feature_search_roles/
│   ├── global/
│   ├── examples/
│   └── heldout/
├── _listening_evaluation/
├── feature_normalization/
└── custom_comparisons/
    └── dynamics_pedaling/
```

기존 Dynamics·Pedaling 전체 분석은 `dynamics_pedaling/`에 보존한다. 쉬운 개별 예시는
`dynamics/`와 `pedaling/`에서 읽는다. 정규화 분석은 여러 feature를 비교하는 주제이므로
별도 폴더에 둔다.

Articulation 개별 예시는 `_articulation/`에 둔다. 이름 앞의 `_`로 기존 전체 데이터
보고서 `articulation/`와 구분한다.

## 분석 실행과 저장 규칙

분석 스크립트는 `classicfy-ai/scripts/`에 있다. 아래 스크립트의 기본 출력은 실행
디렉터리와 관계없이 이 `analysis/` 아래의 해당 주제 폴더다. `--out`을 명시하면 지정한
경로를 사용한다. 상대 경로의 `--out`은 실행 디렉터리를 기준으로 해석한다.

저장소 루트에서 실행하는 예시:

```bash
.venv/bin/python classicfy-ai/scripts/validate_dynamics_pedaling.py \
  --asap-root ../datasets/ASAP
.venv/bin/python classicfy-ai/scripts/validate_articulation.py \
  --asap-root ../datasets/ASAP --nasap-root ../datasets/nASAP
.venv/bin/python classicfy-ai/scripts/validate_feature_normalization.py
.venv/bin/python classicfy-ai/scripts/validate_tempo.py
.venv/bin/python classicfy-ai/scripts/validate_rubato.py
.venv/bin/python classicfy-ai/scripts/validate_dynamics.py
.venv/bin/python classicfy-ai/scripts/validate_pedaling.py
.venv/bin/python classicfy-ai/scripts/validate_articulation_examples.py
.venv/bin/python classicfy-ai/scripts/validate_embedding.py
.venv/bin/python classicfy-ai/scripts/validate_interpretation_examples.py
.venv/bin/python classicfy-ai/scripts/validate_feature_search_roles.py
.venv/bin/python classicfy-ai/scripts/validate_temporal_embedding.py --out /tmp/classicfy-temporal-evaluation
.venv/bin/python classicfy-ai/scripts/observe_temporal_neighbors.py --out /tmp/classicfy-temporal-neighbors
```

선택 작품 비교는 `--works`를 사용하고, `--out`을 `analysis/custom_comparisons/<주제>/`로
지정한다. 주제별 출력 폴더를 나누면 같은 이름의 그림이 서로 덮어써지지 않는다.
자세한 예시는 [선택 작품 비교 안내](custom_comparisons/README.md)에 있다.

새 분석은 `analysis/<분석_주제>/` 안에 README와 그림·CSV·JSON을 함께 저장한다.
그림 이름과 README의 상대 링크를 유지한다. Dataset과 대용량 beat 캐시는 저장소 밖
`Classicfy/datasets/`에 두고, 분석 문서에는 생성 조건과 재현 명령을 기록한다.

기존 `docs/analysis/`의 Tempo·Rubato 자료와 `reports/`의 보고서를 이 구조로 옮겼다.
이동 시 기존 그림·CSV·JSON의 내용은 보존했다.
