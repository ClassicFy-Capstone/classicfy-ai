# ATEPP–ASAP 데이터 호환성 및 다음 컴퓨터 작업 인계

2026-10-28 KST. 브랜치: `feat/#16-performance-autoencoder`.

## 여기서부터 이어서 작업

저장소 루트에서 다음 명령을 실행한다. Git LFS를 먼저 설치해야 한다.

```bash
git fetch origin
git switch 'feat/#16-performance-autoencoder'
git pull --ff-only
git lfs install --local
git lfs pull
python classicfy-ai/analysis/dataset_compatibility/check_handoff.py
```

보고서·코드·CSV·그림은 일반 Git으로, **계속 작업할 4개 NPZ 입력 캐시(약 80 MiB)는 Git LFS로 함께 전달**한다. checkout 후 NPZ가 100여 바이트라면 실제 데이터 대신 LFS pointer만 받은 상태이므로 `git lfs pull`이 필요하다. 이 데이터는 새로 만든 quarter-note 특징 입력이며 기존 모델의 학습 캐시나 checkpoint가 아니다.

## 완료한 내용

| 자료 | 결과와 시작 문서 |
|---|---|
| ATEPP–ASAP 작품 대응 | [전체 메타데이터 비교](asap_atepp_overlap/README.md): 같은 작품·악장 단위 공통 음악 작품 200개. 원본 ID/반복판을 유지하면 ASAP 201항목 ↔ ATEPP 202 ID |
| ASAP 동일 MIDI 중복 | [중복 조사](asap_duplicate_audit/README.md): Beethoven 18-2의 Schmitt01 두 경로가 바이트까지 같음. 과거 D/P/A 기초 표에는 둘 다 포함됐지만 현재 570연주 임베딩 모집단에는 둘 다 없음 |
| 특징 분포 차이 | [차이 분석 및 전처리](asap_atepp_feature_gap/README.md): 공통 134작품 자료, ASAP 496연주/ATEPP 1,510연주. 작품을 분리한 출처 구분 AUC는 raw 0.938 → 표준화 후 0.769. 전사 오류율·AE 성능이 아닌 관측된 데이터셋 차이 |
| 시간축 통일 | 정렬된 ASAP 1,036개에서 동일 MIDI 복제 1개를 제외한 **1,035개를 quarter-note grid로 변환**. 정규화 reference를 만들 수 있는 968개를 별도 입력으로 저장 |

기존 학습 코드·cache·checkpoint는 바꾸지 않았다. 원본 데이터셋과 기존 모델 파일은 이 브랜치에 새로 넣지 않았다.

## 바로 사용할 입력

`asap_atepp_feature_gap/all_asap_quarter_normalized.npz`:

- `manifest`: JSON. 각 항목에 `key`, 음악 작품 `work`, 악보/정렬 그룹 `group`, `channels`, `scales`가 있다.
- `i/q`: 원래 악보 quarter 좌표 경계. 간격은 1.
- `i/values`: `[7,T]` float32, 공통 패턴 제거·표준화 값.
- `i/valid`: `[7,T]` boolean. 결측 위치는 `values=0`과 별도 mask로 구분한다.
- `i/raw`: 같은 비교 구간의 원시 특징.

```python
import json
import numpy as np

path = 'classicfy-ai/analysis/dataset_compatibility/asap_atepp_feature_gap/all_asap_quarter_normalized.npz'
with np.load(path, allow_pickle=False) as z:
    records = json.loads(str(z['manifest']))
    values = z['0/values']
    valid = z['0/valid']
    q = z['0/q']
```

채널 순서: `tempo`, `rubato`, `dynamics`, `articulation`, `pedal_depth`, `pedal_down_ratio`, `pedal_changes`.

현재 `WindowDataset(window_size=64, stride=32, minimum_coverage=.5)`에서 767연주·12,889개 window가 만들어지는 것까지 확인했다. 각 window는 **64 quarter notes**다. 전체 968개가 모두 모델 입력으로 적격인 것은 아니다. Articulation 결측과 coverage 등은 mask로 확인한다.

## 다음에 해야 할 작업

1. ATEPP의 원본 MIDI·DualDTW raw/정규화 cache를 가진 컴퓨터에서 7채널 시계열을 같은 형식으로 export한다. 이 Mac에는 ATEPP 메타데이터와 저장된 분석 요약만 있어 ATEPP 시계열 학습 입력은 아직 만들지 않았다.
2. 현재 ASAP 전용 cache 경로와 동일 corpus hash만 허용하는 encode 경로를, 명시적인 새 입력 규격을 받는 학습/추론 경로로 확장한다. provenance 검사를 단순 삭제하는 방식으로 처리하지 않는다.
3. 같은 quarter 규격으로 모델을 학습·평가한다. 기존 native-beat checkpoint와 시간축 의미가 달라졌으므로 shape가 같다는 이유로 검증 없이 재사용하지 않는다.
4. 같은 최종 encoder로 두 데이터셋을 임베딩한다. 따로 학습한 두 encoder의 좌표를 직접 비교하지 않는다.
5. 새 연주 평가와 새 작품 평가를 구분해 split한다. 공통 작품의 ASAP 연주 평가는 새 작품 일반화 검증이 아니다. `work`별로 반복판·악보 변형까지 함께 split한다.

현재 차이는 연주자·판본·정렬·reference 크기·센서/전사 조건을 모두 포함한다. 전사 오차만 분리하려면 동일 녹음의 센서 MIDI와 같은 전사기로 얻은 MIDI를 대조해야 한다. 새 quarter 경계의 36.4%는 annotation 사이의 시각 보간이므로 긴 beat 내부 타이밍의 정답을 복원한 것은 아니다.

## 원본 데이터 경로와 재현

Python 3.12에서 검증했다. 기존 프로젝트 환경 또는 `requirements.txt`를 사용한다. 새 패키지는 추가하지 않았다. 아래 구조를 기본으로 찾으며, 다른 경로라면 `CLASSICFY_DATASETS`를 지정한다.

```text
workspace/
  classicfy-ai/                 # 이 Git 저장소; 폴더 이름은 자유
    classicfy-ai/src/
    classicfy-ai/analysis/
  datasets/
    ASAP/metadata.csv
    ASAP/asap_annotations.json
    ASAP/<composer>/...         # 연주/악보 MIDI
    nASAP/metadata.csv
    nASAP/<composer>/...        # note match 정렬 자료
    ATEPP/ATEPP-metadata-1.2.csv # 전체 작품 중복 재집계 때 사용
```

PowerShell:

```powershell
$env:CLASSICFY_DATASETS = 'D:\Classicfy\datasets'
python classicfy-ai/analysis/dataset_compatibility/asap_atepp_feature_gap/verify_quarter_grid.py
python classicfy-ai/analysis/dataset_compatibility/asap_atepp_feature_gap/verify_results.py
```

macOS/Linux:

```bash
export CLASSICFY_DATASETS=/path/to/datasets
python classicfy-ai/analysis/dataset_compatibility/asap_atepp_feature_gap/verify_quarter_grid.py
python classicfy-ai/analysis/dataset_compatibility/asap_atepp_feature_gap/verify_results.py
```

원본을 다시 받을 때의 출처:

- ASAP: `https://github.com/fosfrancesco/asap-dataset.git`, 검사에 사용한 revision `afc815c75c42e83a79c03feb6da8a35e77d4c6b8`.
- nASAP: `https://github.com/CPJKU/asap-dataset.git`, revision `4097b45757bed854818cf87e77b92323ebf90615`. `.match` 정렬 파일과 metadata가 필요하다.
- ATEPP 전체 metadata: `https://raw.githubusercontent.com/tangjjbetsy/ATEPP/master/ATEPP-metadata-1.2.csv`, 사용 파일 SHA256 `b6bed1190cb85e607282fe4b761bc42d017f8144851940a4103aaa1a1d84536d`. 원본 ATEPP MIDI는 해당 데이터셋의 접근 절차를 따른다.

대용량 NPZ를 받은 상태라면 모델 입력 분석과 저장된 CSV 재검토에는 원본 다운로드가 필요 없다. `verify_results.py`의 원본 이벤트 독립 재계산에는 ASAP MIDI가 필요하다. 전체 추출 재현 순서는 [세부 보고서](asap_atepp_feature_gap/README.md)에 있다. 그림 재생성에는 Apple SD Gothic Neo, Malgun Gothic, Noto Sans CJK KR, NanumGothic 중 하나가 필요하며 OS에 맞게 자동 선택한다.
