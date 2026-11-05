# 연주별 채널 평균 baseline

각 연주 전체의 유효 관측으로 11개 채널의 평균을 계산하고, 해당 연주의 모든 8마디 구간에 동일하게 출력했다. 8마디별 평균이 아니다. Tempo는 유효 beat당 한 번, 나머지는 채널별 관측 음표만 평균낸다. Tail의 유효 관측도 연주 전체 평균에는 포함하며, 평가는 기존 full 8마디 구간만 사용한다.

기존 train R scaler를 유지했다. Tempo와 articulation은 log2 표현에서 평균을 구했다. 같은 Huber(delta=1), mask, beat 중복 제거, 그룹·구간 균등 평균을 사용했다.

| Split | Train 전체 평균 | 해당 연주 평균 |
|---|---:|---:|
| train | 0.352929 | 0.264625 |
| validation | 0.344794 | 0.217938 |
| test | 0.347824 | 0.238729 |

아래 AE는 학습 중 epoch 13의 validation 최저 모델이다. 최종 모델 평가가 아니다.

| Feature | Validation 연주 평균 | Validation AE |
|---|---:|---:|
| tempo | 0.047358 | 0.023998 |
| onset | 0.059586 | 0.058317 |
| velocity | 0.325036 | 0.120268 |
| articulation | 0.350729 | 0.207645 |
| pedal | 0.306983 | 0.231748 |

페달 7개 채널은 기존 평가처럼 한 그룹으로 평균한다. 11개 개별 채널 loss와 관측 구간 수는 summary.json에 저장했다.

연주 전체 정답에서 얻은 평균을 사용하는 설명용 baseline이다. AE의 8마디 입력보다 넓은 범위의 정보를 사용한다. Test 표는 baseline만 계산한 값이며 AE의 최종 test 결과가 아니다. 학습 설정·정규화·체크포인트 선택은 변경하지 않았다.

기존 train/validation 평균 baseline의 loss가 앞선 독립 계산과 정확히 일치함을 확인했다. 작은 수기 예제로 beat별 평균, 미확인 제외, 채널별 평균 및 Huber 집계도 확인했다.

재현:
```sh
.venv/bin/python scripts/evaluate_note_ae_performance_mean.py --corpus /path/to/indexed-corpus --out /path/to/new-summary.json
```
