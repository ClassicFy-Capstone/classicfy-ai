# 프레이즈·긴장 반응 표현의 후속 실험과 평가 방향

2026-10-14. 평균 중복 진단부터 개별 곡선 비교, 반응 모델 파일럿, 같은 구간의 여러 연주를 비교하는 거리 학습까지의 코드·보고서·CSV·그림을 보존한다. 하위 디렉터리의 파일은 **실행 당시의 byte 그대로인 스냅샷**이다. 과거 문서의 로컬 경로와 당시의 미실행/완료 상태도 역사 기록으로 남겼다. 현재 해석과 이어갈 작업은 이 문서와 [평가 방향](EVALUATION_FRAMEWORK.md)을 먼저 읽는다.

**이번 거리 학습은 구간별 관측 차이를 벡터에 표현하는 능력을 개선했지만, 실제 연주 검색의 개선 폭은 작았다. 기존 전체 시계열 모델보다 좋거나 나쁘다는 결론은 아직 내릴 수 없다.** 마지막 논의에서 확인한 핵심 과제는 고정된 프레이즈·긴장 score를 평가 기준으로 놓고 기존 전체 흐름 기반 유사도가 그 기준을 얼마나 반영하는지 먼저 비교하는 것이다. 이 핵심 head-to-head 비교는 아직 실행하지 않았다.

## 저장한 내용

| 디렉터리 | 내용 | 주요 결과/상태 |
|---|---|---|
| [summary_mean_overlap](summary_mean_overlap/README.md) | 평균·범위와 구조 구간 요약의 중복 진단 | 구간 평균과 전곡 평균의 높은 상관; 평균 제거 효과는 대조 조건에서 일관되지 않음 |
| [curve_comparison](curve_comparison/README.md) | 개별 17점 곡선·변화량·긴장 lag 비교와 통제 | 원래335개에서 지원되는329개를 고정; 정규화 모양/순서 통제로 단순 곡선의 높은 점수 해석을 제한 |
| [response_model](response_model/README.md) | 관측 곡선+악보 맥락의 작은 CNN, 복원/반응/맥락 전용 비교 | 두 방향×네 조건,8개 모델, seed1개; 높은 프레이즈 점수의 작품 선택 영향과 낮은 긴장 검색 효과 |
| [conditional_metric_model](conditional_metric_model/DESIGN.md) | 악보 경로 분리·동일 구간의 연주 차이 학습 설계 | 정확한 악보 좌표/반복 방문으로 구간 대응,13개 테스트와8개1-step 점검 |
| [conditional_metric_run01](conditional_metric_run01/README.md) | 네 조건×두 방향×세 seed,24개 모델의 학습·검색 비교 |18개 테스트, 주 검산701,142개와 추가 대응 진단114,960개 통과 |

마지막24개 모델에서 같은 구간의 관측 거리와 벡터 거리 순위 상관은 악보 분리 조건의 거리 학습 추가로 프레이즈 .573→.781, 긴장 .537→.748로 높아졌다. 같은 작품 내 반대-view 검색 점수는 프레이즈53.51→54.55, 긴장52.54→53.78로 개선 폭이 작다. 이 score는 정확도가 아니며 무작위 연주 선택 기대값은50이다.

다른 작품 검색에서는 주 모델의 실제 선택과 반환 작품 안의 무작위 연주 기대값 차이가 test에서 프레이즈+.96점/긴장+.29점이다. 작품 선택의 영향이 남는다. 같은 구간끼리만 검색하는 추가 진단도 큰 도약을 보이지 않았다. 추가 진단은 주 결과를 본 뒤 정의했으며 별도 탐색 결과로 표시했다.

기존 test를 반복 개발에 사용했고 test/validation은 각각3개 작품군이다. seed SD는 초기화/학습 변동이며 모집단 신뢰구간이 아니다. 추정한 구간·수치 score가 실제 지각적 음악 구조/취향 정답인지 검증된 것도 아니다.

## 파일 보존과 외부 산출물

193개의 원본 코드/문서/CSV/JSON/PNG 스냅샷을 저장했다. 학습 이력과 메타데이터, 수정 전 실제 학습 소스도 해당 디렉터리에 있다. 체크포인트 `.pt`와 배열 `.npz` 121개는 대용량 실행 산출물이므로 외부 `Classicfy/diagnostics`에 유지한다. 파일별 SHA-256·크기·저장 위치 구분은 [archive_manifest.json](archive_manifest.json)에 있다. 하위 `manifest.json`은 외부 파일까지 포함한 원래 완료 실행의 기록이다.

현재 기기에는 외부 산출물이 모두 있다. 다른 기기에서 완료 실행을 검산하려면 이121개 파일과 기존 `Classicfy/datasets`의 입력 자료를 별도로 가져와야 한다. Git clone만으로 원본 데이터/체크포인트를 얻을 수는 없다. 이 문서 작성 과정에서 원본 MIDI·MusicXML이나 사용자 환경 설정은 Git에 추가하지 않았다.

## 복원·검산·재학습

스냅샷 코드는 당시 `workspace/diagnostics/<stage>` 배치를 전제로 한다. 이 저장소 안에서 하위 코드를 바로 실행하면 그 경로 계약이 맞지 않는다. [복원 CLI](../../scripts/restore_structure_response_followup.py)는 원래 스냅샷과 외부 산출물을 모두 해시 검증한 뒤 새 replay workspace에 배치한다. 기존 디렉터리는 덮어쓰지 않으며 저장소/데이터는 symlink로 참조한다.

저장소 루트에서 실행한다.

```bash
.venv/bin/python classicfy-ai/scripts/restore_structure_response_followup.py --check-only
.venv/bin/python classicfy-ai/scripts/restore_structure_response_followup.py \
  --out ../diagnostics/structure_response_followup_replay01

.venv/bin/python ../diagnostics/structure_response_followup_replay01/diagnostics/conditional_metric_run01/test_design.py
.venv/bin/python ../diagnostics/structure_response_followup_replay01/diagnostics/conditional_metric_run01/test_pipeline.py
.venv/bin/python ../diagnostics/structure_response_followup_replay01/diagnostics/conditional_metric_run01/verify.py
.venv/bin/python ../diagnostics/structure_response_followup_replay01/diagnostics/conditional_metric_run01/verify_aligned.py
```

다른 기기에서는 `--artifacts`와 `--datasets`로 가져온 경로를 지정한다. 같은 Python 의존 환경이 필요하다. 복원한 완료 결과를 보존하며 새24개 모델 실험을 실행하려면 다음과 같이 새 형제 디렉터리를 지정한다.

```bash
.venv/bin/python ../diagnostics/structure_response_followup_replay01/diagnostics/conditional_metric_run01/run.py \
  --out ../diagnostics/structure_response_followup_replay01/diagnostics/conditional_metric_repro02
```

이는 검증된 기존 곡선/배열부터의 복원·재학습 경로다. MusicXML부터 모든 예전 전처리를 새 경로에서 생성하는 것은 과거 absolute provenance 경로와 원본 자료 준비를 별도로 맞춰야 한다. 완료 실행의 배열을 재생성해서 기존 해시 참조를 덮어쓰지 않는다.

실제 아카이브와 외부 파일314개를 임시 workspace에 복원한 뒤 모델/평가 테스트18개, 주 수치 검산701,142개와 추가 검산114,960개를 다시 통과했다. 복원 CLI 테스트6개도 통과했다. [패키징 검증 기록](packaging_verification.json)에 결과를 남겼으며 임시 복원본은 검증 후 제거했다. 원래 실험 파일은 변경하지 않았다.

이 아카이브는 제품 검색 모델을 교체하거나 전체 시계열 baseline보다 우수함을 선언하는 변경이 아니다. 후속 비교의 기준과 미완료 항목은 [평가 방향](EVALUATION_FRAMEWORK.md)에 명시했다.
