# Epoch 8 AE: 단일 기준 연주의 표현·메타데이터 탐색 사례

2026-10-30에 만든 발표 그림, 원래 벡터, 후보 순위, 검증 결과와 생성 코드를 보존한다.
기준 A는 **Vladimir Horowitz의 슈만 《Kreisleriana》 Op.16 6번, ATEPP:00957**이다.
모든 그림은 같은 epoch 8 checkpoint를 사용한다. 이후 학습의 최종 checkpoint와 구분한다.

## 발표 그림

| 사례 | 그림과 수치 | 관찰 |
|---|---|---|
| 같은 작품의 22개 녹음 | [PNG](same_work_checkpoint/same_work_checkpoint.png) · [PDF](same_work_checkpoint/same_work_checkpoint.pdf) · [설명](same_work_checkpoint/README.md) | feature 최근접 B와 최원거리 C를 먼저 선정. AE 거리 0.0115 < 0.1708 |
| 연주자 중복 제거, 13명 | [PNG](same_work_checkpoint_unique_artists/same_work_checkpoint.png) · [PDF](same_work_checkpoint_unique_artists/same_work_checkpoint.pdf) · [설명](same_work_checkpoint_unique_artists/README.md) | A 유지, 다른 연주자는 최소 ID 녹음. B=Grimaud, C=Uchida. AE 거리 0.0903 < 0.1708 |
| 다른 작품의 전체 검색 후 메타데이터별 최근접 | [PNG](cross_work_checkpoint/cross_work_checkpoint.png) · [PDF](cross_work_checkpoint/cross_work_checkpoint.pdf) · [설명](cross_work_checkpoint/README.md) | 다른 작곡가·시대인 Debussy가 전체 4위지만 5D feature 근접 2074/2085위로 표현 프로필은 멂 |
| 연주자·작곡가·시대 조건별 AE 비교 | [PNG](ae_metadata_comparison/ae_metadata_comparison.png) · [PDF](ae_metadata_comparison/ae_metadata_comparison.pdf) · [설명](ae_metadata_comparison/README.md) | 같은 연주자 14위·0.1242, 같은 작곡가 1위·0.0517, 같은 시대·다른 작곡가 2위·0.0671 |

각 폴더에 편집용 SVG도 있다. 같은 작품 그림의 왼쪽은 64D AE 벡터의 PCA이며,
오른쪽 5D feature 프로필은 별도로 계산한 설명용 값이다. 전곡 벡터는 모든 full 8마디의
unit e를 균등 평균한 뒤 L2 정규화했다. 마지막 tail은 제외한다.

학습은 검증 통과 AtEPP 2,801연주/254작품 중 train 2,070연주/185작품을 사용한다.
다른 작품 검색은 train/validation/test 카탈로그를 합쳐 같은 작품을 제외한 2,779연주/253작품에서 수행했다.
원본 AtEPP 전체나 독립적인 heldout 성능 평가를 뜻하지 않는다. 기준 A는 하나다.

## Horowitz 논문과의 비교에서 확인한 점

같은 연주자의 다른 작품 최근접은 전체 14/2779위, **상위 약 0.50%**이므로 절대적으로 멀다고 할 수 없다.
같은 작곡가 1위 후보는 **같은 《Kreisleriana》의 다른 악장**이다. 기존 work_id는 악장 단위다.
《Kreisleriana》 전체를 제외한 같은 작곡가 후보는 19개이고, 최근접은 Ashkenazy의
《Arabeske》 Op.18로 **전체 211위, cosine 거리 0.2895427282**다.
[전체 19개 후보](same_composer_outside_cycle.csv)를 보존했다.
따라서 앞의 조건별 최소 거리 비교에서 작곡가 정보가 연주자 정보보다 강하다는 일반적인 결론을 내리지 않는다.
후보 수가 다르고 그룹도 일부 겹친다.

[Zhang & Dixon, Disentangling the Horowitz Factor](https://www.huanz.space/assets/Disentangling_the_Horowitz_factor__Learning_content_and_style_from_expressive_piano_performance.pdf)의
Fig.3은 별도로 분리한 style 변수 Z_S를 보여준다. 논문은 content/style branch, content VQ,
상호정보량 최소화와 자기지도 변형을 다룬다. 해당 시각화는 5개 스타일과 5개 발췌의 조합 표본이다.
현재 checkpoint는 악보 조건부 단일 잠재벡터의 복원 전용 학습이며 lambda_style=0이다.
악보 조건 입력만으로 잠재벡터에서 content가 제거된다고 보장되지 않는다.
복원 성능과 cosine 공간의 스타일 유사성은 별도로 검증해야 한다.
작품 정보의 영향은 가능한 설명이며 이 한 사례로 인과적으로 확정하지 않는다.

## 데이터 없이 그림 다시 그리기

저장소 루트에서 실행한다. 원본 MIDI·전처리 corpus·모델 checkpoint 없이 저장된 벡터로 4개 그림의
PNG/SVG/PDF를 다시 만든다. 현재 프로젝트 Python 의존성(numpy, matplotlib, torch 등)이 필요하다.

```sh
.venv/bin/python classicfy-ai/scripts/render_note_ae_checkpoint_cases.py --out /tmp/classicfy-ae-cases
```

재현 스크립트는 41개 원본 산출물/코드의 SHA256을 검증하고, 저장된 수치만 render 함수에 전달한다.
`generators/`는 최초 workspace 생성 코드를 원문 그대로 보존한 것으로, 그 main 명령과 하위 README의
최초 추론 명령은 저장소 밖의 `Classicfy/presentation/`, `Classicfy/diagnostics/`, `Classicfy/datasets/` 구조를 전제로 한다.
현재 저장소에서 그림만 재현할 때는 위 entry point를 사용한다.

체크포인트와 대용량 구간별 추론 chunks는 Git에 넣지 않았다. 원래 로컬 workspace에 유지하며,
checkpoint SHA256 `2614c425bc248d04d42e45382b3d7b951c8d608436169546e981d54e11e0d5d4`를 기록했다.
원본 데이터로 재추론하려면 이 checkpoint와 최초 생성 코드가 참조하는 전처리 corpus/feature 자료가 필요하다.
`cross_work_checkpoint/catalogue_embeddings.npz`에 전체 2,801개 전곡 벡터,
`all_ranked_candidates.csv`에 A와 검색 후보의 전체 순위를 보존했다.
원본과의 바이트 일치 목록은 `archive_manifest.json`, 사례별 수치 검증은 각 provenance/verification JSON에 있다.

커밋 전 검증은 `validation.json`에 기록했다. AE pipeline/indexed data 테스트 33개가 통과했고,
원본 데이터 없이 다시 생성한 네 PNG가 기존 그림과 픽셀 단위로 정확히 일치했다.
