# 전곡 23차원 연주 프로필

`performance_style_profile.py`는 개별 연주 레코드의 전체 악보 traversal에서
23차원 `g`를 계산한다. 여러 악장을 합치지 않으며 마지막 8마디 미만의 tail도 포함한다.
기존 8마디 g와 별도의 `whole-performance-style-v1` schema를 사용한다.

## 통계 정의

입력은 AE 표준화 전 원본 R 캐시다. 기존 log2·/127·quarter 단위 변환을 다시 적용하지 않는다.
동일 split·작품·악보 SHA·반복 경로의 독립 녹음들이 위치별 reference를 제공한다.
기준 위치/채널당 최소 3개 관측의 중앙값을 쓰고 대상 연주도 포함한다.
같은 recording ID는 기준 집합에 한 번만 기여한다.

음표는 score occurrence key로, tempo는 정확히 같은 악보상 `[q_start, q_end)` beat 구간으로
대응한다. 연주별 beat table의 길이가 달라도 행 번호로 맞추거나 보간하지 않는다.
원본 R과 reference의 차이를 구한 뒤 **전곡 관측값에서 직접** 통계를 다시 계산한다.
8마디 요약을 평균하지 않는다.

| 영역 | 통계 | 차원 |
|---|---|---:|
| Tempo | log2 tempo 잔차의 중앙값 | 1 |
| Timing | tempo 잔차의 Q90−Q10·roughness, onset 잔차의 중앙값·Q90−Q10 | 4 |
| Dynamics | velocity 잔차의 중앙값·Q90−Q10 | 2 |
| Articulation | log2 duration ratio 잔차의 중앙값·Q90−Q10 | 2 |
| Pedaling | 기존 R pedal 7채널의 잔차 중앙값·Q90−Q10 | 14 |

분위수는 기존 g와 같은 **동일 값의 가중치 합산 후 weighted midpoint 선형 보간**이다.
일반적인 numpy 표본 분위수와는 보간 정의가 다르다.
Tempo에는 독립 beat의 악보상 quarter 길이를 가중한다.
나머지 채널에는 동일 onset의 eligible attack 수의 역수를 가중한다.
Grace와 tie continuation은 제외하며 결측 화음 음표의 가중치를 다른 음표에 재분배하지 않는다.
Pedal은 R에 저장된 음표의 onset/offset/hold/gap 관측 요약이며, pedal CC의 전곡 시간 점유율은 아니다.

Roughness는 원본 beat table의 실제 인접 유효 beat 쌍에서
`sum(abs(delta_tempo[i+1] - delta_tempo[i])) / sum(q[i+1] - q[i])`로 계산한다.
결측 beat를 건너뛰어 연결하지 않는다.

## 결측·split·표준화

초기 조건은 coverage .8, 유효 beat/onset group 8개, roughness 인접 쌍 7개다.
이 조건은 기존 8마디 g에서 계승한 운영 기준이며 전곡 유효성의 검증된 임계값은 아니다.
값이 없는 차원은 0과 false mask로 저장하며 관측된 0과 구분한다.
채널별 관측률·reference 관측률·유효 표본 수·전곡 앞/중간/뒤 관측률·실패 사유를 남긴다.
전체 악보 범위 중 기존 AE index의 full window가 덮는 비율도 기록한다.
현재 열린 파일이나 별도 기간/시대 표본으로 입력을 암묵적으로 바꾸지 않는다.

Train reference는 train 연주만 사용한다. Validation/test reference는 각 split 내부에서
**평가용 g 라벨을 작성하기 위해서만** 만든다. 이 값은 단독 unseen 연주만으로 계산할 수 없다.
평가 g의 존재를 작품 정보를 제거했다는 증거나 청취자 정답으로 해석하지 않는다.

Scaler는 train의 고유 녹음별 g만 이용해 median/IQR로 fit한다.
IQR이 epsilon 이하면 population std로 대체하고, 관측 20개 미만/상수 차원은 비활성화한다.
기존 AE R scaler나 8마디 g scaler를 재사용하지 않는다.
별도의 작품 균등 가중 scaler도 민감도 자료로 저장한다. 이 보조 scaler의 분위수는
weighted midpoint 방식을 사용하므로 primary numpy-linear scaler와의 차이는
작품별 가중치와 분위수 보간 방식의 영향을 함께 포함한다.

## 실행

`classicfy-ai`에서 기존 입력 index와 **새** 출력 경로를 지정한다.
AE 학습·loss·checkpoint·기존 입력 파일은 변경하지 않는다.

```sh
/path/to/python scripts/prepare_performance_style.py \
  --index /path/to/ae_inputs/index.json --out /path/to/performance_style/run

/path/to/python scripts/validate_performance_style.py \
  --directory /path/to/performance_style/run

/path/to/python scripts/report_performance_style.py \
  --directory /path/to/performance_style/run --out /path/to/new-report

PYTHONPATH=src:tests /path/to/python -m unittest \
  integration.test_performance_style integration.test_style_triplets -q
```

선택적인 `--config config.json`은 `PerformanceStyleConfig`의 임계값을 지정한다.
한 cohort씩 읽어 처리하고 진행을 stdout에 출력한다. 출력 덮어쓰기는 거부한다.
실패한 실행은 부분 산출물을 유지하며 `verification.json`이 있는 성공 실행과 구분한다.

산출물은 `protocol.json`, `source_audit.json`, cohort별 `references/*.json`·`*.npz`,
`profiles_raw.jsonl`, `profiles_scaled.jsonl`, `profiles.npz`, `scaler_train.json`,
`scaler_train_work_balanced.json`, `report.json`, `verification.json`이다.
추가 검증 CLI는 전 행의 표준화와 NPZ, train scaler 23열 전체, split별 표본의
통계량을 별도 계산으로 확인해 `independent_verification.json`에 기록한다.

원본 R/XML/window/index 해시를 실행 전후 확인하고, reference 배열의 저장 해시도 검사한다.
입력 fingerprint·코드 해시·설정·split 및 평가용 reference 역할을 protocol에 고정한다.
