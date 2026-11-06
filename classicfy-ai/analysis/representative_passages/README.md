# 작품별 대표 미리듣기: 근거와 v2 구현

작성일: 2026-11-06. 목적은 새 작품의 성격을 파악하면서 연주 버전을 고르기 좋은 짧은 구간을 제공하는 것이다. **반복대표성과 음악 내용의 중심성이 음악적 아이덴티티를 보장한다는 이론은 확인되지 않았다.** 두 값은 근거가 있는 후보 생성용 근사 지표이며, 대표성·청취 효용의 직접 검증이 필요하다.

## 문헌에서 뒷받침되는 범위

### 반복·변형과 음악적 통일성

Schoenberg의 *The Musical Idea*는 동기 하나만이 아니라 구절·주제·리듬·화성·형식을 포괄하는 작품 전체의 통일성을 다룬다. 출판사 소개와 학술 분석에서 확인할 수 있는 범위이며, 책 전체를 직접 검토했다고 주장하지 않는다. Boss의 음악이론 연구는 여러 계층에서의 동기 반복이 조성 작품의 유기적 통일성에 기여함을 논하고, 관련 개념에 대한 학자들 간 이견도 설명한다. 따라서 동기·선율·리듬의 반복과 변형을 추적할 근거는 있지만, 반복 횟수가 가장 많으면 가장 중요한 구간이라는 결론은 나오지 않는다. 이 관점을 ATEPP의 모든 시대·형식에 동일하게 일반화하지 않는다.

- Schoenberg, *The Musical Idea and the Logic, Technique, and Art of Its Presentation*, edited by Severine Neff, 2006 paperback edition. [Indiana University Press 소개](https://iupress.org/9780253218353/the-musical-idea-and-the-logic-technique-and-art-of-its-presentation-new-paperback-english-edition/).
- Boss, J. F. (1999), [*“Schenkerian-Schoenbergian Analysis” and Hidden Repetition in the Opening Movement of Beethoven’s Piano Sonata Op. 10, No. 1*](https://mtosmt.org/issues/mto.99.5.1/mto.99.5.1.boss.html), Music Theory Online 5(1). 이 문헌은 동기적 통일성에 관한 음악이론적 논의이며 자동 대표 구간 검출기의 성능 증거가 아니다.

### 음악 내용의 중심성은 MIR의 계산 가설

Cooper와 Foote(2002)는 전체 작품과의 평균 유사도가 높은 연속 구간을 대표 요약으로 선택한다. 이는 중심성에 대한 직접적인 알고리즘 근거다. 그러나 논문은 통계적으로 비슷한 구간이 의미상 최선이라는 보장이 없음을 명시하고, 베토벤 5번의 유명한 동기를 놓치는 요약을 예로 든다. 따라서 중심성은 음악학적으로 중요한 구간을 판정하는 공리가 아니라, 자동으로 계산할 수 있는 대표성의 근사다. v2는 원 논문의 오디오 MFCC 대신 MIDI 내용 특징을 사용하므로 원 방법을 그대로 재현한 결과도 아니다.

- Cooper, M. & Foote, J. (2002), [*Automatic Music Summarization via Similarity Analysis*](https://archives.ismir.net/ismir2002/paper/000010.pdf), ISMIR.

### 요약 목적과 반복성

Peeters 등(2002)은 요약의 적합성이 찾고자 하는 정보에 달린다고 설명하고, 반복 기반 방법의 적용 범위를 제한한다. FMP thumbnailing은 반복 유사도와 비중첩 반복의 커버리지를 결합한다. 이는 반복대표성을 설계하는 계산 근거이며, 사람이 느끼는 작품 정체성의 보장과 다르다. v2의 subsequence-DTW 반복 커버리지는 이 원리에 착안한 별도 근사이며 FMP의 전역 최적 경로군 알고리즘과 동일하지 않다.

- Peeters, G., La Burthe, A. & Rodet, X. (2002), [*Toward Automatic Music Audio Summary Generation from Signal Analysis*](https://ismir2002.ircam.fr/proceedings/02-FP03-3.pdf), ISMIR.
- Müller, Jiang & Grosche (2013), [FMP Audio Thumbnailing 해설과 참고 논문](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C4/C4S3_AudioThumbnailing.html).

### 형식 역할과 맥락도 필요

Caplin의 형식 이론은 구간이 작품 조직에서 수행하는 역할을 강조한다. sentence의 presentation·continuation·cadential 기능처럼, 같은 재료의 반복만으로 역할을 모두 설명하지 않는다. 이 근거는 반복·중심성만으로 충분하다고 주장하지 않도록 제한하고, 구절 문맥과 마무리를 보존할 필요성을 뒷받침한다. 적용 범위는 특히 하이든·모차르트·베토벤의 고전 형식이다. v2가 이러한 기능을 자동 분석했다고 주장하지 않는다.

- Caplin (1998), [*Classical Form*](https://academic.oup.com/book/49159), [Sentence 장 소개](https://academic.oup.com/book/49159/chapter-abstract/422032516).

## 이 프로젝트에서의 조작적 정의

대표 후보 = 짧은 연속 구간으로서 (1) 작품의 다른 위치와 선율·화성·리듬·텍스처상의 관련성이 있고 (2) 정적인 반복 반주만으로 높은 점수를 받는 것을 억제하며 (3) 서로 거의 같은 후보의 중복을 줄이고 (4) 실제로 청취 가능한 시작·끝을 갖는 후보.

현재 구현은 위 정의의 일부를 근사한다. onset chroma와 최고/최저 음의 흐름을 4개 순서 블록으로 요약하며, 성부 분리·완전한 선율 음정 인식·화성 기능 분석은 구현하지 않았다. 평균 유사도는 통계적 중심성, 반복 커버리지는 반복대표성, 구간 내 특징 변화는 정적 반주 억제 지표다. 유명하지만 한 번만 등장하는 선율이나 중요한 예외 구간은 놓칠 수 있다.

반복과 중심성의 0.5:0.5 결합, 최고 대표성 점수의 80% 이상 유지, DTW 유사도 0.65, 반복 유사도 0.72 등의 수치는 **검증 전 엔지니어링 기본값**이다. 음악학 문헌에서 도출한 상수나 보장된 정확도 기준이 아니다. 반복만/중심성만/두 값 결합/첫 30초를 비교하는 ablation과 청취 검증으로 정해야 한다.

## 구현과 입력

- `src/preprocessing/representative_passages.py`: 작품 기준 후보, 탄력적 반복 매칭, 내용 중심성, 경계 보정, 연주 간 내용 DTW, 강건한 스타일 차이·leave-one-out 안정성.
- `scripts/run_representative_passages.py`: composition_id별 처리, 기준 연주 선정, 최대 4개 공통 후보를 각 연주에 대응, 하나를 공통 선택해 MIDI 출력.
- `scripts/run_atepp_theme_candidates.py`: 기본 실행은 v2로 전환. 이전 실험은 `--legacy`로만 실행한다. 기존 v1 출력은 변경하지 않는다.

입력은 기존 전체 ATEPP audit의 local_midi와 작품 ID다. 악보·형식·종지 주석은 입력에 필요하지 않다. quality는 기준 선택·스타일 비교 지원 연주를 고르는 데 사용한다. 모든 입력 연주에 처리 상태를 남긴다.

기준은 품질 경고가 적은 연주 중 길이가 중앙값에 가까운 연주다. 후보는 곡 전체의 여러 길이 창에서 생성한다. 곡 앞 35% 제한과 도입부/30초 선호 가중치는 제거했다. 각 후보가 개별적으로 대표성 근사 조건을 만족해야 하며 최대 4개를 강제로 채우지 않는다.

시간 변화는 query 전체에서 일정한 반음 이동을 유지하는 subsequence-DTW로 다룬다. 반복 위치의 겹침과 자기 자신을 제외한다. 대표성은 앞 단계에서 결정하며 스타일 차이가 대표성이 낮은 후보를 되살리지 못한다.

연주 간 대응은 실제 내용 DTW이며 시간 비율 비교가 아니다. 다만 전역 단조 DTW는 반복 생략을 항상 올바르게 해결하지 못하므로, 지역 유사도나 늘어남/축소 비율이 불량한 후보는 거부한다. 반복 구조를 확정한 정확한 음표 정렬이라는 뜻은 아니다. 거부된 연주에 독립 구간을 대체 선택하지 않는다.

선택은 최소 3개의 유효한 스타일 비교 자료가 있을 때 후보별 쌍별 거리의 중앙값을 계산한다. leave-one-performance-out 점수의 하위 20% 값과 전체 중앙값 중 작은 값을 보수적 선택 점수로 사용한다. 자료가 적으면 이 민감도 분석도 불안정하다. 고정 척도의 상대 시간·velocity 변화폭·음 길이 분산을 사용한다. **학습된 스타일 임베딩이 아닌 진단용 대리 특징**이며 matched-note articulation 또는 원음의 음량 정답이 아니다. 유효 비교가 부족하면 대표성 1순위로 fallback하고, 변별력을 측정한 것처럼 표시하지 않는다. 이 안정성은 동일 연주자 그룹을 분리한 독립 평가를 대신하지 않는다.

모든 결과의 identity_verified=False, style_embedding_trained=False를 유지한다. 개선된 정확도나 음악적 아이덴티티 보존을 입증한 결과는 아직 없다.

## 실행

프로젝트 루트 `capstone-2026-2-classicfy`에서:

```powershell
$env:PYTHONPATH='classicfy-ai/src'
.\.venv\Scripts\python.exe -m unittest discover -s classicfy-ai/tests/unit/preprocessing -p test_representative_passages.py -v
# 전체 데이터: 기존 v1 출력과 분리된 폴더에 생성
.\.venv\Scripts\python.exe classicfy-ai/scripts/run_atepp_theme_candidates.py --workers 2
# 일부 작품으로 실행 검증
.\.venv\Scripts\python.exe classicfy-ai/scripts/run_atepp_theme_candidates.py --work-ids 1000,1016,1017 --out ../outputs/atepp_representative_v2_validation
# 구형 실험을 명시적으로 실행
.\.venv\Scripts\python.exe classicfy-ai/scripts/run_atepp_theme_candidates.py --legacy --limit 20 --out ../outputs/legacy_smoke
```

후보와 선택 근거는 `candidates.csv`, 연주별 상태는 `segments.csv`, 대응 거부/실패는 `review_queue.csv`, 전체 대응과 스타일 비교·안정성은 `works/<composition_id>.json`에 저장한다. `unvalidated`는 낮은 불확실성이라는 뜻이 아니라 아직 청취 효용을 보정하지 않았다는 뜻이다.

## 다음 검증

전체에 적용 가능한 구현과 전체에서 정확한 대표성을 보장하는 것은 다르다. 주석 없는 작품을 포함해 시대·형식·전사 품질별 표본을 추출하고, 독립 청취자가 대표성·잘림의 자연스러움·해석 비교 용이성을 평가해야 한다. 구간 선택에 쓴 연주와 다른 연주에서의 변별력, 사용자 탐색 시간까지 따로 확인한다. v1 주제군 IoU 47.5%와 v2 대표성 점수를 같은 정확도로 비교하지 않는다.

## 이번 실행 검증 범위

- 기존 회귀 테스트 14개 및 새 테스트 5개 통과. 새 테스트는 속도 변화 반복 대응, 속도·offset이 다른 연주 대응, velocity 변화에 대한 내용 후보 불변성, 단일 연주의 fallback, 극단적인 녹음 하나에 대한 강건성을 확인했다.
- 단일 연주 작품 3개: 12후보 생성 / MIDI 3개 출력.
- 여러 연주가 있는 composition_id 1000, 1016, 1017: 주석 없이 40개 입력 모두 상태 기록, 12개 작품 공통 후보, 선택 구간 MIDI 38개 내보내기 및 원본 내용 검증 통과, 대응 불확실 2개 거부.
- 결과: `outputs/atepp_representative_v2_checked/segments.csv`, `works/*.json`. 이 ID들은 실행 검증용으로 선택했으며 독립적인 음악적 대표성 시험 세트라는 의미는 아니다.
- 위 항목은 전수 실행 이전의 실행 검증이다. 아래 전수 실행과 구분한다.

## v2 전수 실행 및 주제부 주석 평가

2026-11-06, `outputs/atepp_representative_v2_full`에 전수 실행했다. 기존 v1 결과는 보존했다. 주석은 후보 생성과 선택에 사용하지 않았고, 실행을 마친 뒤 기존에 전이해 둔 주석으로만 비교했다. 평가 후 선택 가중치나 추출 구간은 변경하지 않았다.

- 공식 ATEPP 11,674개 연주 / composition_id 1,595개 모두 처리 상태 기록.
- 후보 6,087개, MIDI 11,403개 출력(97.7%), 대응 불확실 271개 보류. 읽기·후보 생성·내보내기 오류는 없었다.
- 원본 11,674개의 해시 재검사에서 변경 0개. 출력 MIDI와 명세·공통 후보 대응 검증 오류 0개. 내보내기 시 원본 대비 검사한 음표는 2,992,104개다.
- 스타일 대리 특징으로 선택한 작품 855개, 유효 비교 연주 부족으로 대표성 1순위에 fallback한 작품 740개. 학습된 스타일 임베딩의 전수 추출 결과가 아니다.

작품 내 연주 평균 → 작품 동일 가중치 평균이며, 대응 거부는 0점으로 포함했다. 평가는 두 참조 집합으로 분리한다. 최초 주제군은 17작품/293연주, 확보된 모든 주제군은 21작품/382연주다.

| 참조 | 지표 | v2 실제 선택 | 첫 30초 | v2와 같은 길이의 도입부 |
|---|---|---:|---:|---:|
| 최초 제시부 제1주제군/변주 Theme | IoU≥0.5 | 5.9% | 56.8% | 62.7% |
| 확보된 주제군 중 어느 하나 | IoU≥0.5 | 9.5% | 46.0% | 50.8% |
| 확보된 주제군의 합집합 | 미리듣기 80% 이상 포함 | 19.3% | 46.5% | 59.7% |

**주제부 추적 성능은 기준선보다 나빠졌으며, v2를 개선된 주제부 추적기로 채택할 근거가 없다.** 짧은 구간이라 IoU가 불리한 점을 감안해 포함률로 비교해도 개선되지 않았다. 다만 이 결과가 음악적 아이덴티티나 추천 효용의 직접 평가를 대신하지는 않는다.

주제군 어느 하나와의 IoU≥0.5는 대표성 1순위만 선택하면 25.6%, 주석을 보고 후보 중 최선을 사후 선택하는 oracle은 56.0%다. 최초 주제군만 대상으로 한 oracle은 11.8%다. 따라서 스타일 대리 특징에 따른 최종 선택만의 문제가 아니며, 현재 상위 후보 집합도 최초 주제군을 충분히 보존하지 못한다. oracle은 평가 연주마다 주석을 보고 선택한 상한이며 실제 자동 선택 성능이 아니다.

참조 시간은 이전 BPSD→ATEPP 자동 정렬로 전이한 값이다. 독립 수작업 시간 정답이 아니며 주제 위치 누락이 있다. 연결되는 형식 주석 31작품 중 지원하는 주제 라벨이 있는 작품은 27개지만, 신뢰할 수 있는 시간 참조가 확보된 21개 작품만 주제 일치도에 사용했다. 기존 개발/시험 분할도 별도 제공하나 새 미관측 시험으로 주장하지 않는다. 이전 47.5% 수치와는 평가 작품·분모가 달라 그대로 비교하면 안 된다.

모든 결과는 `unvalidated`다. `review_queue.csv`는 대응 거부 271개, `review_flags_all.csv`는 전체의 검수 사유를 담는다. 실제 출력 중 60초 초과 26개가 있으며, 작품별 스타일 선택 안정성 0.6 미만은 189개다. 구간 길이 중앙값은 27.15초이며 모든 연주에 15~35초가 보장되는 것은 아니다.

재현 명령:

```powershell
.\.venv\Scripts\python.exe -u classicfy-ai/scripts/run_atepp_theme_candidates.py --workers 4 --out ../outputs/atepp_representative_v2_full
.\.venv\Scripts\python.exe classicfy-ai/scripts/evaluate_representative_passages.py ../outputs/atepp_representative_v2_full
.\.venv\Scripts\python.exe classicfy-ai/scripts/verify_representative_passages.py ../outputs/atepp_representative_v2_full
```

전수 실행은 원본과 코드 해시가 같은 작품 체크포인트만 재사용한다. 평가 결과와 전체/성공본 조건부 지표는 출력 폴더의 `README.md`, `theme_evaluation.json`, `theme_evaluation_performance.csv`, `theme_evaluation_work.csv`에 있다. 파일 검증은 `verification.json`이다. 출력 ZIP에는 해당 실행의 소스 스냅샷을 포함한다.

다음 개발은 이 결과를 실험 기준점으로 고정한 뒤 진행해야 한다. 우선 별도 수작업 청취·구절 참조를 마련하고, 주제군의 초기 출현과 변형·재현을 묶은 후보군을 비교한다. 대표성 점수의 편향과 반복 매칭 오류를 확인한 뒤, 음표 내용에 대응한 연주 특징으로 선택 단계를 검증한다. 검증되지 않은 현 대리 특징의 차이 최대화를 최종 추천 기준으로 쓰지 않는다.

## 후보 전체 제공: 최종 1개 선택 생략

사용자 요청에 따라 기존 후보·연주 대응을 그대로 재사용하고, 연주마다 최대 4개 후보를 모두 내보내는 경로를 추가했다. 기존 `atepp_representative_v2_full` 선택본과 원본은 변경하지 않는다. 이 경로는 스타일 대리 특징에 의한 최종 선택을 적용하지 않는다.

```powershell
.\.venv\Scripts\python.exe -u classicfy-ai/scripts/export_all_representative_candidates.py --workers 4
.\.venv\Scripts\python.exe classicfy-ai/scripts/verify_all_representative_candidates.py ../outputs/atepp_representative_v2_all_candidates
```

출력은 `outputs/atepp_representative_v2_all_candidates`다. `segments.csv`는 연주×후보마다 한 행이며, `candidate_index`는 기존 0부터 시작하는 번호, `candidate_number`는 공유용 1부터 시작하는 번호다. 각 작품에서 확정된 후보 집합을 모든 연주에 대응시킨 결과다. 후보 번호는 대표성 점수 순서이며 청취 선호나 최종 추천 순위가 아니다.

- `midi/<composition_id>/<perf_id>/candidate_01.mid` 등: 기존 대응 검사를 통과한 후보.
- `midi_review/<composition_id>/<perf_id>/candidate_01.mid` 등: 대응이 불확실해도 유효한 시각과 음표가 있어 출력 가능한 후보. `exported_review`로 구분한다.
- 길이가 0인 구간은 `invalid_interval`로 목록에 남기며 파일은 만들지 않는다. 누락된 매핑이나 내보내기 실패도 후보 행을 삭제하지 않고 사유를 기록한다.
- `clip_relative_path`는 압축 해제 폴더 기준 상대 경로다. `start_seconds`, `end_seconds`는 각 원본 연주 MIDI의 시간이고, 추출 MIDI의 시간은 구간 시작부터 0초다.
- `candidates.csv`는 작품 기준 후보, `works/*.json`은 전체 연주 대응과 후보별 출력 내역, `performances.csv`는 연주별 후보 수·출력 수다. 이 출력에는 단일 `selection`을 저장하지 않는다.
- 모든 출력은 `unvalidated`다. 대응 검사 통과와 음악적 대표성·주제부 판정 정확도는 구분해야 한다. 기존 후보 생성 품질에 관한 한계도 그대로 적용된다.

원본은 연주마다 한 번 파싱해 여러 구간에서 재사용한다. 출력마다 별도 MIDI 디코더로 원본 음표와 음높이·velocity·시간을 대조하고, 음표·페달 종료와 파일 길이를 검사한다. 전체 검증에서는 연주×후보 누락·중복, 원본 해시, 저장된 후보·대응의 불변성, 상대 경로와 파일 해시까지 확인한다. 테스트는 불확실 후보 포함 전체 보존, 빈 구간·누락 대응 기록, 원본 변경 차단, 출력 음표 변조 검출을 확인한다.

전체 내보내기 결과는 1,595개 작품·11,674개 연주, 작품 후보 6,087개, 연주×후보 45,302행이다. 대응 통과 MIDI 44,179개와 대응 불확실 표시 MIDI 1,086개를 모두 저장했다. 길이가 0인 37개만 파일 없이 목록에 보존했다. 내보내기 실패나 원본 해시 불일치는 없었으며, 모든 연주에 적어도 하나의 후보 파일이 있다.

연주별 생성 후보 수는 1개 142연주, 2개 324연주, 3개 320연주, 4개 10,888연주다. 빈 구간으로 파일이 빠진 경우는 `performances.csv`의 실제 출력 수로 구분한다. 검수용까지 포함한 출력 길이는 0.54~371.03초, 중앙값 29.81초이며 60초 초과 220개를 표시했다. 이는 불확실 매핑도 삭제하지 않고 보존한 결과다. 기준 연주에서 생성한 후보 길이는 기존과 같은 약 11.0~37.5초다. 최종 선택이나 확정 주제부 라벨은 포함하지 않는다.
