# ATEPP 앞단 전수 검증

2026-10-29 KST. AE 모델·loss·optimizer·학습 loop는 실행하지 않았다.

## 결과

| 단계 | 결과 |
|---|---:|
| 원본 연주 MIDI 이벤트 파싱 | 11,674 / 11,674 |
| 고유 XML/MXL 악보 준비 | 284개 통과, 35개 거부 |
| 자동 정렬·R 추출·physical 역변환·캐시 검사 | 2,801개 통과 |
| R scaler의 train 연주 | 2,070개 |
| 적격 full 8마디 AE 입력 | 67,542개 |
| 실제 pre-AE collator/DataLoader 배치 | 34,437개, 실패 0개 |
| Padding 위치 mask 검사 | 921,006곳 |
| g의 full 8마디 / tail | 67,545 / 2,307 |
| core 9열 관측 조건을 통과한 g | 47,807개 |
| g scaler active feature | 23 / 23 |
| 적격 Triplet | 31,736개 |
| 관련 회귀 테스트 | 79개 통과 |

전체 입력의 첫 제외 사유는 다음과 같다. 합계는 원본 연주 수와 일치한다.

- `no_XML`: 6,583개
- `quality_label_excluded`: 1,149개
- `score_unsupported`: 726개
- `extracted`: 2,801개
- `alignment_rejected`: 415개

미통과 자료를 valid 0이나 임의 XML/정렬로 보완하지 않았다. 자동 정렬은 기존 DualDTW
품질 gate를 사용하며 수동 정렬의 정답이 아니다. 동봉된 score MIDI와 XML의 동등성을 추정하지 않고,
XML에서 새 canonical score MIDI를 만든 뒤 tick·pitch·offset 왕복 검사를 통과한 alias만 사용했다.
긴 Liszt 소나타 10개는 1 worker로 재검사했으며 최종 resource-limit 미처리 자료는 없다.

## 수정·확인한 문제

- MXL rootfile의 media-type 생략과 일반 XML이 .mxl 확장자로 제공되는 ATEPP 파일을 지원했다.
  복수 rootfile·잘못된 XML·지원하지 않는 navigation은 계속 거부한다.
- 8마디 경계에 쉼이 있을 때 onset 기반 범위가 tempo beat를 이웃 구간으로 배정하는 입력 마스크를 수정했다.
  회귀 예시는 33/31개에서 올바른 32/32개로 수정됐고, 실제 932개 구간에 영향이 있었다.
  전 구간의 수정된 beat 소유권·중복 없음·R/P/M·길이·padding·정규화 1회 적용을 검사했다.
- 같은 verified delta snapshot을 구간들이 공유하도록 해 반복 hash 비용을 줄였다.
- g 후보 수와 후보 work는 인덱스로 계산하고, 선택한 work의 pool만 만든다.
  work 균등 표집, signature, 거리 식, cutoff, 기본 임계값은 유지했다.
  벡터 거리의 reduction 순서도 scalar와 일치시켜 inclusive cutoff의 부동소수점 차이를 막았다.

R의 원본 값·mask는 바꾸지 않았다. 독립 재계산한 train 정규화 평균의 최대 절댓값은
1.32e-08, 2차 모멘트의 1 대비 최대 오차는
4.94e-08이다. tempo는 note 복제 수가 아닌
독립 valid beat 1,300,671개로 계산했다. 공식 ZIP의 원본 파일
12,331개는 크기·CRC 불일치가 없었다.

## g·샘플 선정

거리 보정 signature 상태: `{'valid': 8}`.
각 signature에서 최소 100, 최대 20,000 ordered pair 규칙을 유지했다.
실제 signature·pair 수·cutoff·delta와 단계별 탈락 사유는 final_summary.json 및
calibration.json에 있다. 전체 N×N 거리 행렬을 만들지 않는다.

Triplet은 서로 다른 작품의 positive와 같은 판본·반복 경로·8마디의 다른 연주 negative만
포함한다. 저장된 동일 J_a로 D_ap/D_an과 gap 조건을 확인했다. g와의 일치가 청취 스타일의
정답이라는 의미는 아니다. AE용 R에 작품 기준 잔차를 적용하지 않았다.

이번 split은 검증용 별도 프로토콜이다. 명시적 작품 대응에서 확인되는 기존 ASAP work split을
계승하고, 나머지는 동일 악보/녹음 그룹을 묶어 seed=2026의 안정적인 SHA256 70/15/15 분할을
기록했다. 기존 실험 split은 수정하지 않았다. 실제 학습 채택은 별도로 결정할 사항이다.
ScoreEncoder는 미학습 가중치의 shape/mask/유한값 계약만 검사했다.

## 재현과 산출물

classicfy-ai 디렉터리에서 실행한다. 정렬은 alignment_environment.json의 별도 환경
(Parangonar 3.3.3, Numba 포함)을 사용했고, 검증은 프로젝트 환경을 사용했다.

```sh
.venv/bin/python scripts/validate_atepp_note_pipeline.py scan --root ../../datasets/ATEPP --out ../../diagnostics/atepp_pre_ae_validation/run02 --workers 4
../../diagnostics/atepp_pre_ae_validation/runtime/bin/python scripts/validate_atepp_note_pipeline.py align --root ../../datasets/ATEPP --out ../../diagnostics/atepp_pre_ae_validation/run02 --workers 4
../../diagnostics/atepp_pre_ae_validation/runtime/bin/python scripts/validate_atepp_note_pipeline.py align --root ../../datasets/ATEPP --out ../../diagnostics/atepp_pre_ae_validation/run02 --workers 1 --retry-resources --max-note-cells 500000000
.venv/bin/python scripts/validate_atepp_note_pipeline.py finish --root ../../datasets/ATEPP --out ../../diagnostics/atepp_pre_ae_validation/run02
.venv/bin/python scripts/validate_atepp_note_inputs.py --out ../../diagnostics/atepp_pre_ae_validation/run02 --workers 2
```

scan/align은 행별 append-only log에서 재개할 수 있다. 수정 후 기존 거부 악보를 재검사하려면
`scan --retry-rejected`을 사용한다.

전체 R/window/profile 검증 캐시가 완료된 경우 다음으로 거리·Triplet 집계를 재개할 수 있다.

```sh
.venv/bin/python scripts/validate_atepp_note_pipeline.py finalize --root ../../datasets/ATEPP --out ../../diagnostics/atepp_pre_ae_validation/run01
```

- 원본 검사: raw.jsonl, scores.jsonl, scan_summary.json, source_integrity.json。
- 정렬/추출: alignments.jsonl, alignment_summary.json, scores/*/bridge.json, performances/*/R。
- 입력: R_scaler.json, S_scale.json, window_index.json, windows/*, input_batches.jsonl, input_batch_summary.json。
- g: reference_index.json, references/*, profile_index.json, profiles/*, profiles_scaled/*, g_scaler.json。
- 거리/관계: calibration.json, triplets.json, metric_progress.log。
- 통합 결과: final_summary.json, normalization_check.json, regression_tests.log。

대용량 원본·캐시는 workspace의 diagnostics에서 관리하며 Git에 추가하지 않았다.

전체 로컬 결과: `workspace/diagnostics/atepp_pre_ae_validation/run01/`.
