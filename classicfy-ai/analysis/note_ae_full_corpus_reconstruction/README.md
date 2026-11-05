# 검증 통과 ATEPP 전체 데이터의 복원 AE

검증을 통과한 254작품·2,801연주를 사용한다. 원본 ATEPP 전체가 아니다.

| Split | 작품 | 연주 | 8마디 구간 |
|---|---:|---:|---:|
| train | 185 | 2,070 | 51,638 |
| validation | 35 | 314 | 6,474 |
| test | 34 | 417 | 9,430 |

작품·악보 SHA·동일 MIDI hash의 split 간 중복은 0개다. R/S scaler는 train에서만 계산했다.

기존 ScoreEncoder/AE와 5그룹 Huber 복원 loss를 유지한다. Triplet·평균 보조 head·초기 tempo 조건은 사용하지 않는다. 처음부터 학습하며, 최대 100 epoch / validation patience 15 / min_delta 0.0001이다. 매 epoch 학습 51,638개 구간을 한 번씩 사용한다. 배치 수 제한은 없다. GPU 배치 128과 길이별 묶음으로 padding을 줄였다.

**상태: 학습 진행 중. 최종 test 성능은 아직 평가하지 않았다.**

완료 epoch: 27. 전체 train 평가 loss: 0.118123, validation loss: 0.116564. 이 값은 최종 선택 모델의 평가 결과가 아니다.

최근 학습 진행 기록: epoch 28, 31,926/51,638구간.

7작품 pilot과는 split·scaler·배치 크기도 달라 데이터 증가만의 효과를 분리한 실험은 아니다. 한 seed와 자동 정렬 데이터에 대한 결과다. 이전 pilot에서 일부 작품을 이미 확인했으므로 완전히 미열람 상태의 독립 benchmark로 해석하지 않는다.

Validation 복원 loss로 체크포인트를 선택한 뒤 train·validation·test를 평가한다. Test로 설정을 변경하지 않는다. 원본 로그와 체크포인트는 `workspace/diagnostics/full_corpus_reconstruction/run01/`에 보관한다.
