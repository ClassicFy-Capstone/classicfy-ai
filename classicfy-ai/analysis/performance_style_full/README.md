# 전곡 23차원 g 부분 계산 결과

**전체 11,674개 연주에 대한 g 처리는 미완료다.** 아래 결과는 기존 AE 전처리 index에
들어 있던 2,801개만 대상으로 실행했다. 나머지 8,873개는 이번 g 실행에서 계산을 시도하지 않았다.
기존 전처리 제외 상태를 g 계산 실패나 영구적인 계산 불가 판정으로 해석하지 않는다.
전체 split 배정은 train 8,450 / validation 1,661 / test 1,563개다.
[원본 전체 연주별 처리 대조표](../../../../diagnostics/performance_style/full_population_audit/coverage.csv)와
[전체 모집단 집계](../../../../diagnostics/performance_style/full_population_audit/summary.json)를 별도로 남겼다.

17번 AE 입력 index의 2,801개 연주에 대해 연주당 한 개의 전곡 프로필 행을 계산했다.
원본 ATEPP 전체가 아닌 기존 AE 검증 입력 집합이며, 새 duration/시대 필터는 적용하지 않았다.
관측량이나 기준이 부족한 차원은 결측 mask로 남겨 전체 행 수와 유효 프로필 수를 구분한다.

| Split | 작품 | 연주 행 | 하나 이상 유효 | Core 9차원 유효 | 23차원 모두 유효 | 표준화 후 5영역 관측 |
|---|---:|---:|---:|---:|---:|---:|
| train | 185 | 2070 | 2008 | 1955 | 1794 | 1949 |
| validation | 35 | 314 | 306 | 299 | 271 | 299 |
| test | 34 | 417 | 406 | 405 | 371 | 405 |

Train scaler 활성 차원은 **23/23**이다. 비활성 차원: 없음.
표준화 후 5영역 관측 수는 각 영역에 유효 차원이 하나 이상 있다는 뜻이며,
모든 후보를 같은 signature로 비교할 수 있는 최종 학습 쌍의 수는 아니다.

## 계산 정의와 범위

전체 악보 traversal의 원본 R을 작품·판본·반복 경로별 peer reference와 비교한다.
중앙값·Q90−Q10·tempo roughness를 전곡에서 직접 다시 계산하며 마지막 tail도 포함한다.
Tempo는 악보상 quarter 길이, 다른 채널은 화음 균형 onset 가중치를 사용한다.
정확히 같은 beat 구간만 대응하고 결측을 건너뛰지 않는다.
분위수는 기존 g의 동일 값 합산 후 weighted midpoint 선형 보간 정의를 유지한다.

기준 위치당 독립 관측 최소 3개,
coverage 80%, beat/onset group 8개,
roughness 인접 쌍 7개를 초기 기준으로 사용했다.
이 기준은 기존 8마디 g에서 계승한 운영 기준이다.

Train reference와 scaler는 train에서만 계산했다. Validation/test reference는 각 split의
peer 연주로 **평가용 상대 g를 작성하는 데만** 사용했다. 학습 scaler에 반영하지 않았다.
대상 연주도 자신의 reference 집합에 포함된다. 단독 unseen 연주에서 이 상대 g가 바로
계산된다는 뜻이나, 청취자 스타일 정답 또는 작품 정보 제거의 증거는 아니다.

전곡 통계 범위와 기존 AE full 8마디 범위의 차이는 연주별 `ae_scope_coverage`에 있다.
원본 R pedal 관측과 전처리 품질을 계승하며, g는 CC의 전곡 시간 점유율 통계가 아니다.
작품 균등 scaler 비교에는 가중치와 분위수 보간 정의의 차이가 함께 포함된다.

## 확인 자료

- [차원별 분포·표준화·품질 상관](feature_statistics.csv): 23차원 × 3 splits.
- [연주별 유효성·관측률·AE 범위](profile_quality.csv): 전체 입력 연주별 한 행.
- [계산 계약](../../src/embedding/PERFORMANCE_STYLE.md): 통계·가중치·결측·실행 방법.

원본 R/XML/window/index 실행 전후 해시, split 중복 0, 모든 입력의 1:1 출력,
저장 reference 배열 해시를 검증했다. 추가 독립 계산으로 전체 표준화와 train scaler
23열 및 각 split에서 선택한 총 9개 표본의 통계량을 확인했다.
별도 합성/CLI 테스트는 테스트 실행 기록을 참조한다.

실행 산출물: `/Users/ihun-ui/Desktop/Classicfy/diagnostics/performance_style/run02`

`profiles.npz`에는 performance ID, split, work, raw/scaled g와 두 mask가 있다.
`profiles_scaled.jsonl`에는 채널별 coverage·기준 표본 수·구간별 관측률·실패 사유가 있다.
`report.json`에는 split별 집계·차원별 분포·길이/관측률/기준 집합 크기와의 상관이 있다.
`protocol.json`, `source_audit.json`, `verification.json`, `independent_verification.json`으로 재현 근거를 남겼다.

## 재실행

`classicfy-ai`에서 새 출력 디렉터리를 지정한다.

```sh
/path/to/python scripts/prepare_performance_style.py --index /path/to/ae_inputs/index.json --out /path/to/new-run
/path/to/python scripts/validate_performance_style.py --directory /path/to/new-run
/path/to/python scripts/report_performance_style.py --directory /path/to/new-run --out /path/to/new-report
```

이번 단계는 전곡 g 추출과 품질 확인이다. AE 거리 loss 연결·추가 학습·청취 평가는 실행하지 않았다.
