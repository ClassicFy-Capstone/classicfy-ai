# G 상한 완화 AXY 청취

저장소 루트에서 Python 표준 라이브러리로 실행한다. YouTube 재생에는 인터넷 연결이 필요하다.

```sh
python classicfy-ai/tools/ae_selected_era_listening_relaxed/serve.py --port 8771
```

[http://localhost:8771/](http://localhost:8771/)

전체 5,082개 대표구간에서 시대별 임베딩을 계산한 뒤, 같은 시대 안에서 기준 A와 가장 가까운 다른 작품의 연주 B를 찾았다. B와 같은 작품의 연주 중 A에서 가장 먼 C를 골라 거리 차이가 큰 조합을 사용했다. 바로크 3 / 고전과 전환 3 / 낭만과 인상주의 4문항이며, 바로크 03 교체와 고전과 전환 01 복원을 반영했다.

YouTube 시작·종료 시각은 데이터셋 매핑과 대조했다. 최종 교체 구간의 실제 재생과 대표구간 경계는 청음 검수가 남아 있다. 화면에서 후보의 거리와 모델 순서는 응답 제출 후 공개된다.

- [G 기울기 상한 비교 및 보존 기준](../../analysis/era_g_budget/comparison.md)
- [곡 구조와 G 순서를 함께 검수한 AXY](../ae_selected_era_listening/README.txt)
