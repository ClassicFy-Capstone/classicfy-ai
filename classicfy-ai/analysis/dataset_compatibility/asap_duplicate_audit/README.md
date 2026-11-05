# ASAP 동일 연주 중복 및 기존 분석 포함 여부

2026-10-28 KST. 원본 데이터와 기존 분석 결과는 변경하지 않았다.

## 확인 결과

전체 ASAP 메타데이터 1,067개 경로의 MIDI를 전수 확인했다. 파일 바이트 SHA256, 음표+CC64 이벤트, 선행 무음을 제거한 이벤트, 선행 무음을 제거한 음표 이벤트를 각각 대조했다. 이벤트 시각은 마이크로초까지 비교했다. 네 기준 모두 동일한 한 쌍만 검출했다.

- `Beethoven/Piano_Sonatas/18-2/Schmitt01.mid`
- `Beethoven/Piano_Sonatas/18-2_no_repeat/Schmitt01.mid`

둘 다 메타데이터의 작곡가는 Beethoven, 작품명은 Piano_Sonatas_18-2다. 바이트 SHA256은 `98e8ee6c2887444058298d69112036517a77bf642ad3d3e15ced92e0d4bcb8b7`로 같다. 따라서 단순히 연주자 이름이 같은 별도 녹음이 아니라 완전히 같은 MIDI 복제다. 전체 파일 경로 수는 1,067이지만 바이트가 고유한 MIDI는 1,066개다.

## 기존 결과에 포함된 사례

| 항목 | 18-2 경로 | 18-2_no_repeat 경로 |
|---|---:|---:|
| Dynamics·Pedaling summary CSV 행 | 235 | 237 |
| Articulation summary CSV 행 | 235 | 237 |
| beat 구간 수 | 462 | 340 |
| Dynamics mean | 0.44875938022732176 | 0.44927992964526087 |
| Pedal depth mean | 0.09308251346607616 | 0.12648270947449172 |
| Articulation mean | -1.6891861803462775 | -1.680436406527916 |

동일 MIDI를 서로 다른 경로/악보/beat annotation으로 추출해 별도 행으로 계산한 사례다. 실제 연주 차이가 없어도 정렬 구간의 차이로 특징 요약값이 달라졌다. 첫 경로에는 short_beat_interval 또는 low_beat_coverage 플래그가 붙었지만 summary 표에 두 경로가 함께 남아 있다. 플래그만으로 중복 등록이 제거된 것은 아니다.

다만 두 기초 분석 코드는 작품을 파일의 부모 폴더로 묶는다. 두 경로는 서로 다른 work 그룹에 들어간다. 따라서 이 두 복제 파일끼리 같은 그룹의 서로 다른 연주 쌍으로 직접 비교한 것은 아니다. 첫 그룹에서 Schmitt는 ChenGuang04와, 두 번째 그룹에서는 KOLESO06과 각각 비교되는 구조다. 전체 summary에는 같은 MIDI가 두 행으로 들어가므로 경로별 빈도/분포를 집계하면 중복 가중 가능성이 있다. 실제 영향 크기는 재집계하지 않았다.

## 후속 분석에서의 포함 여부

| 저장된 결과 | 18-2 | 18-2_no_repeat |
|---|---|---|
| Dynamics·Pedaling 기초 분석 | 포함 | 포함 |
| Articulation 기초 분석 | 포함 | 포함 |
| Feature normalization performance_summary | 미포함 | 포함(채널별 5행) |
| _embedding_validation/embeddings.csv (570연주) | 미포함 | 미포함 |
| metadata_style_validation/performances.csv | 미포함 | 미포함 |
| tempo/performance_summary.csv | 미포함 | 미포함 |
| rubato/performance_summary.csv | 미포함 | 미포함 |

정규화 결과에는 이 중복 쌍이 함께 들어가지 않았다. 570연주 임베딩 표와 메타데이터 검증 표에는 둘 다 없다. 이것이 명시적 내용 중복 제거 장치가 있었다는 뜻은 아니다. 확인한 현재 모집단에 이 쌍이 함께 들어가지 않는다는 뜻이다. 다른 파일로 편집·재전사·부분 발췌된 동일 녹음까지 모두 검출한 검사는 아니다.

## 이전 전체 수 집계와의 관계

이전 ATEPP–ASAP 보고서의 ASAP 1,067개, 공통 작품 연주 953개(분할 포함 967개)는 경로 기준 수다. 이번에 확인된 바이트 동일 복제 한 개를 제거하면 각각 1,066개, 952개, 966개가 된다. 악보와 정렬 파일 중 어느 경로를 정본으로 유지할지 판단하기 전에는 기존 파일/결과를 자동 삭제하거나 다시 계산하지 않았다.

## 자료와 재현

- `summary.json`: 전수 검사 범위, 기준별 중복 그룹, 원본 메타데이터 hash.
- `performance_identity_catalog.csv`: 전체 MIDI 1,067개의 네 종류 내용 hash와 식별자.
- `duplicate_members.csv`: 기준별 중복 파일과 beat 수.
- `analysis_membership.csv`: 기존 분석 표의 포함 여부·행 번호·추출 특징.

```bash
.venv/bin/python classicfy-ai/analysis/dataset_compatibility/asap_duplicate_audit/audit.py
python3 classicfy-ai/analysis/dataset_compatibility/asap_duplicate_audit/trace.py
```
