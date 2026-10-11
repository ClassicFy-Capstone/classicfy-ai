# ATEPP _feature_search_roles

14좌표와 성향만 포함한 7좌표의 검색 결과를 비교한다. search_changes.csv: 후보 변경·기존 후보 순위·Top5 유지. heldout_agreement.csv: 검색에서 제외한 feature로 B 작품 내 다른 연주와 비교(동점=0.5). source work/grid macro 및 bootstrap. 제거마다 scale을 재fit하지 않는다.

전체 조건과 재현은 [상위 보고서](../README.md).

![01_anchor_rank](examples/01_anchor_rank.png)

![02_anchor_advantage](examples/02_anchor_advantage.png)

![03_typical_profile](examples/03_typical_profile.png)

![04_typical_candidates](examples/04_typical_candidates.png)

![05_typical_advantage](examples/05_typical_advantage.png)

![01_neighbor_changes](global/01_neighbor_changes.png)

![02_top5_retained](global/02_top5_retained.png)

![03_heldout_agreement](global/03_heldout_agreement.png)

![04_pair_removal](global/04_pair_removal.png)

![05_summary_sensitivity](global/05_summary_sensitivity.png)

![01_Tempo](heldout/01_Tempo.png)

![02_Rubato](heldout/02_Rubato.png)

![03_Dynamics](heldout/03_Dynamics.png)

![04_Articulation](heldout/04_Articulation.png)

![05_Pedaling](heldout/05_Pedaling.png)
