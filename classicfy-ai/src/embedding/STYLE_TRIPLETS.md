# 8마디 연주 프로필과 Triplet 전처리

`g`는 작품의 train 연주 집합에 비해 어떻게 다르게 연주했는지를 요약하는
23차원 규칙 기반 프로필이다. 청취자 스타일의 정답이나 작품 정보 제거의 증거가 아니다.
`z`는 후속 AE의 학습 표현이며, 이번 코드는 AE·loss·학습 loop를 포함하지 않는다.

## 입력 계약

`prepare_note_performance.py`가 저장한 **원본 R 캐시**와 그에 대응하는 MusicXML을 사용한다.
AE 정규화 전 `values`를 읽으며, 중복 log·/127 변환이나 R 파일 수정을 하지 않는다.
`R_scaled`만 있는 파일은 이 진입점의 입력으로 받지 않는다. 먼저 저장된 R scaler로
역변환하고 원본 R의 식별 정보·마스크·독립 beat 테이블을 갖춘 캐시를 복원해야 한다.
학습된 ScoreEncoder의 C는 필요하지 않다.

입력 JSON 예시(경로는 JSON 파일의 디렉터리 기준):

```json
{
  "records": [
    {
      "performance_cache": "R/record-a",
      "xml_path": "scores/work-a.musicxml",
      "work_key": "work-a/movement-1",
      "split": "train",
      "performance_id": "dataset:performance-a",
      "recording_group": "original-recording-a"
    }
  ]
}
```

- `performance_cache`, `xml_path`는 필수다. `work_key`와 `split`은 R에 저장된 값과 일치해야 한다.
- `performance_id` 기본값은 `dataset:record_key`다.
- `recording_group`을 생략하면 원본 MIDI SHA256을 사용한다. 둘 다 없으면 거부한다.
  파일 내용이 다른 동일 녹음은 사용자가 동일한 recording_group으로 지정해야 한다.
  동일 MIDI SHA256에 서로 다른 recording_group을 지정하면 거부한다.
- 반복 경로를 명시한 R에는 동일한 zero-based 마디 인덱스 목록 `traversal`을 제공한다.
- work·동일 악보·녹음의 split 교차를 거부한다. 새 split을 생성하지 않는다.
- 판본은 XML SHA256, 반복 경로는 traversal ID로 엄격히 분리한다. 다른 XML 파일의
  동등성을 자동 추정하지 않는다. beat reference도 동일한 q 구간 키에서만 공유한다.
- 여러 마디 쉼표는 실제 rest 마디들이 전개된 XML이어야 한다.

## 계산

1. XML의 전체 measure visits로 8마디/stride 8 window를 만든다. pickup과 무음 마디를
   포함하고 1–7마디 tail은 저장하되 g·Triplet에서 제외한다. R은 전곡에서 계산한 값을 유지한다.
2. 동일 work/판본/반복 경로의 train 독립 연주가 위치·채널별 3개 이상이면 중앙값을
   reference로 고정한다. target도 cohort에 포함될 수 있다. 복제 녹음은 한 번만 센다.
3. 별도 delta를 계산한다. reference 없는 위치는 unknown이며 raw R로 대체하지 않는다.
4. tempo는 독립 beat의 window 겹침 q 길이, 다른 채널은 grace·tie continuation을 제외한
   악보 타건의 `1/onset 타건 수`로 가중한다. 결측 타건의 가중치를 재분배하지 않는다.
5. 동일 값을 합친 weighted midpoint 선형 분위수로 median과 Q90−Q10을 구한다.
   tempo roughness는 window 안에 시작점이 있는 실제 인접 valid beat 쌍만 사용한다.
6. train full-window 프로필의 median/IQR로 g scaler를 fit한다. IQR이 작으면 population
   std로 대체한다. 20표본 미만·상수 피처는 비활성화하고 scale=1로 저장한다.
7. anchor의 core 0–8이 관측되어야 한다. 훈련된 core와 anchor에서 관측된 훈련 pedal
   피처의 고정 signature로 모든 후보를 비교한다. 영역 내부 평균 제곱 차이 후 영역 간 평균을 쓴다.
8. signature별 train pair에서 global cutoff와 gap을 고정한다. 다른 작품의 near positive와
   같은 판본·반복·8마디인 다른 연주의 far negative만 선택한다. 조건은 자동 완화하지 않는다.

고정 열 순서·블록은 `features.style_profile.G_NAMES`, `G_BLOCKS`와 모든 artifact의 schema에 있다.
페달 14열의 개수로 영역 비중이 증가하지 않는다. 정보가 없는 값은 유한한 0과 mask로 저장한다.
R scaler와 g scaler는 별개이며 g에 L2 정규화나 clipping을 적용하지 않는다.

기본값은 `StyleConfig`에 있다: coverage .80, valid beats/onset groups 각 8,
roughness edges 7, calibration pairs 최소 100/최대 20,000, anchor 후보 10,
near/far Q20/Q80, gap=train 거리 IQR의 .1배(std fallback), seed 2026.
필요한 명시적 설정은 `make-windows --config config.json`으로 전달하며 모든 산출물에 기록된다.
테스트에서 낮춘 임계값은 production 기본값을 바꾸지 않는다.

Calibration은 work→anchor→candidate work→candidate 순의 균등 확률로 ordered pair를
중복 없이 표집한다. 최대 20,000개를 유지하며 가능 pair가 작으면 모두 사용한다.
표집이 거의 소진되는 불균형 사례에는 같은 확률을 반영한 remaining-key reservoir로 마무리한다.
전체 N×N 거리 행렬은 만들지 않는다. typed 후보 인덱스와 anchor당 거리 벡터를 사용하며,
영역별 거리 식과 비교 피처 signature는 그대로 유지한다.

## 실행 순서

`classicfy-ai`에서 실행한다. `input.json`에는 기존 work split이 지정된 R 캐시를 넣는다.
아래의 `run`은 새 출력 디렉터리이며 기존 파일을 덮어쓰지 않는다.

```sh
.venv/bin/python scripts/prepare_style_triplets.py make-windows --manifest input.json --out run/windows.json
.venv/bin/python scripts/prepare_style_triplets.py fit-reference --windows run/windows.json --split train --out run/reference.json
.venv/bin/python scripts/prepare_style_triplets.py extract-profiles --windows run/windows.json --reference run/reference.json --out run/profiles.json
.venv/bin/python scripts/prepare_style_triplets.py fit-profile-scaler --profiles run/profiles.json --split train --out run/scaler.json
.venv/bin/python scripts/prepare_style_triplets.py scale-profiles --profiles run/profiles.json --scaler run/scaler.json --out run/scaled-profiles.json
.venv/bin/python scripts/prepare_style_triplets.py calibrate-distances --profiles run/profiles.json --scaler run/scaler.json --split train --seed 2026 --out run/calibration.json
.venv/bin/python scripts/prepare_style_triplets.py build-triplets --profiles run/profiles.json --scaler run/scaler.json --calibration run/calibration.json --split train --seed 2026 --out run/triplets.json
.venv/bin/python scripts/prepare_style_triplets.py report --windows run/windows.json --reference run/reference.json --profiles run/profiles.json --scaler run/scaler.json --calibration run/calibration.json --triplets run/triplets.json --out run/report.json
```

각 stage의 ID는 schema/config/입력 dependency를 포함한 content hash다. window source를 다시
읽을 때 XML·R 대응과 R checksum을 검증한다. 다른 reference·scaler·calibration을 섞으면 거부한다.
`profiles.json`에는 raw g·mask·coverage·reference counts·탈락 사유,
`scaled-profiles.json`에는 g_scaled·훈련 피처 mask·scaler ID가 추가된다.
Triplet에는 원래 R 입력과 악보 구간, 거리·영역별 거리·signature·cutoff·seed가 연결된다.
Triplet이 없는 구간도 원래 AE 복원 입력으로 사용할 수 있다.

## 검증과 해석

```sh
PYTHONPATH=src:tests .venv/bin/python -m unittest integration.test_style_triplets integration.test_note_performance_pipeline -q
```

합성 검증은 구간 경계·반복 회차, reference train-only/독립 표본, chord 균형과 결측 coverage,
분위수·roughness, 비활성 피처, 영역 비중, signature, cutoff/gap, 재현성, 모든 CLI stage와
원본 R 파일 유지를 확인한다. 실데이터 실행 결과는 해당 run의 report와 source audit를 확인한다.
표본 부족·core 비활성·calibration 부족·positive/negative/gap 부족은 실패 사유로 저장되며,
Triplet 0개도 정상적인 데이터 진단 결과다. unseen work에 reference를 자동 fit하지 않는다.

g와의 일치도는 이 규칙을 따르는지 확인하는 지표다. 청취 스타일 유사성은 별도 평가가 필요하다.
