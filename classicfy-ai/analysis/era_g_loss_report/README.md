# AE 및 G loss 실험 보고서

2026-11-13 평가와 2026-11-15 전체 자료 최종 학습의 주요 리포트만 보관합니다.
데이터셋, 원본 미디어, 캐시, 체크포인트, 임베딩 배열과 배포 ZIP은 Git에 포함하지 않습니다.

- `G_only_training_evaluation_report.pdf`: 그림 5개를 포함한 7쪽 평가 보고서. train 전용 모델의 validation/test 및 보조 복원 항 비교입니다.
- `report_data.json`: 보고서 표와 그림의 수치 및 작품 bootstrap 구간입니다. 원본 임베딩 배열은 없습니다.
- `loss_ablation_comparison.json`, `loss_ablation.png`: 공통/복원만/G만/복원+G의 비교입니다.
- `weak_reconstruction_comparison.json`, `weak_reconstruction_validation.png`: 약한 추가 복원 항의 validation 검색입니다.
- `protected_g_final_*`, `protected_g_training_curves.png`: 전체 5082개를 사용하는 보호 G 최종 학습과 검증입니다. 전체 학습 자료 진단으로 독립 test가 아닙니다.
- `protected_g_coverage_summary.json`: 5082구간 사용 및 직접 G 3900구간/anchor-only 1182구간의 집계입니다.
- `common_ae_*`: 공통 AE 사전학습의 기존 결과입니다.
- `model_architecture.png`, `model_architecture.svg`: 최종 전체 자료 모델의 공통 사전학습 → 시대별 보호 G 학습 구조입니다. 기존 `architecture.png`는 평가 보고서의 간단한 개요로 유지합니다. `python scripts/draw_era_model_architecture.py`로 재생성합니다.

실행 명령과 모델 구조는 `../../src/embedding/ERA_NOTE_AE.md`를 참고하세요.
AXY 원본 UI와 선정/재생 검증은 `../ae_selected_era_listening`에 있습니다.
리포트 내 로컬 경로와 과거 파일 hash는 실행 당시의 provenance이며, 파일 본체를 Git에 포함한다는 뜻이 아닙니다.
무효화한 run02 결과는 성능 근거로 포함하지 않습니다.
