# 다른 기기에서 16번 작업 이어가기

## 저장소만 받아서 확인하기

```bash
git fetch origin
git switch 'feat/#16-structure-conditioned-style-embedding'
git pull --ff-only origin 'feat/#16-structure-conditioned-style-embedding'
```

[README](README.md) → [score_review.html](score_review.html) → [review_decisions.csv](review_decisions.csv)
순서로 확인한다. HTML을 로컬 브라우저에서 열면 색으로 표시한 7페이지와 14사례를 볼 수 있다.
SVG를 확대해서 볼 수도 있다. 악보 이미지와 55악보 전체 후보 압축 데이터를 저장소에 포함했으므로
이 검수에는 외부 데이터셋·renderer·이전 기기의 70MB 전체 연결 cache가 필요하지 않다.

현재 완료 상태: 악보에서 후보 생성, 기호·성부·문맥의 첫 시각 검수, 원래 beat 방문별 연결,
독립 계산 검증. **실제 프레이즈 시작·끝·계층의 확정과 임베딩 반영은 아직 진행하지 않았다.**

## 다음 작업

1. 검수 CSV의 12개 서로 다른 위치에서 주선율/성부 역할, 앞 문장의 시작, 문장의 크기를 검토한다.
2. 작은 articulation 묶음과 큰 프레이즈를 구분하고, 도착음에서 끝/시작이 겹치는 경우 구간 경계 규칙을 정한다.
3. 확인한 구간을 원본 ID·scope·시작/끝·근거·불확실성·검수자와 함께 별도 annotation으로 저장한다.
4. 반복 방문과 기존 구조 유효 mask를 적용한 뒤 구간 안의 상대 위치 및 시간·공동 변화 반응을 계산한다.
5. 동일 표본에서 전곡/고정 구간/프레이즈 조건을 비교한다. 조성 긴장도는 별도 정의·검증 단계다.

원래 후보 JSON의 pending이나 `eligible_for_embedding=False`를 일괄 true로 바꾸지 않는다.
종결형 화음 힌트에는 반종지·전위·전조·화음 환원이 없으므로 음악적 종지 정답으로 쓰지 않는다.

## 후보를 다시 생성하려면

저장소 루트에서 Python 가상환경과 기존 [requirements.txt](../../../requirements.txt)를 준비한다.
기본 데이터 경로는 저장소 상위 `datasets/ASAP`, `datasets/nASAP`이며 `--data`로 바꿀 수 있다.
필요한 고정 버전:

| 입력 | 저장소 | commit |
|---|---|---|
| ASAP | [fosfrancesco/asap-dataset](https://github.com/fosfrancesco/asap-dataset) | `afc815c75c42e83a79c03feb6da8a35e77d4c6b8` |
| nASAP | [CPJKU/asap-dataset](https://github.com/CPJKU/asap-dataset) | `4097b45757bed854818cf87e77b92323ebf90615` |

55개의 nASAP `xml_score.musicxml`, pilot의 ASAP score MIDI와 nASAP `.match`, 양쪽 metadata 및
ASAP annotations가 필요하다. 정확한 목록/해시는 [protocol.json](protocol.json)의 `inputs`를 사용한다.
기존 입력 목록은 [score_structure_inputs](../score_structure_inputs/README.md)에 있다.
데이터를 sparse checkout으로 가져올 경우 MusicXML 파일도 포함한다.

새 빈 출력 폴더로 실행한다. 기본 `all_scores/` 산출물은 지정한 `--out` 안에 저장된다.

```bash
MPLCONFIGDIR=/tmp/classicfy-matplotlib \
.venv/bin/python classicfy-ai/scripts/extract_phrase_candidates.py \
  --data ../datasets --out /tmp/classicfy-phrase-recheck

MPLCONFIGDIR=/tmp/classicfy-matplotlib \
.venv/bin/python classicfy-ai/scripts/verify_phrase_candidates.py \
  --data ../datasets --out /tmp/classicfy-phrase-recheck
```

`--render`를 생략하면 악보 이미지 없이 동일한 후보·연결·조건·해시를 생성한다.
Gzip 헤더의 실행 시각은 달라질 수 있다. 수치 재현은 독립 검산으로 확인하며, 해시는 각 실행의 입력·코드·파일을 묶는다.
생성 후 코드가 변경되면 기존 실행의 검산은 코드 해시 불일치로 거부하므로 새 출력 폴더에서 다시 생성한다.
새 출력에는 사람이 추가한 검수 CSV/문서가 자동 복사되지 않는다.

## 악보도 다시 조판하려면

```bash
.venv/bin/python -m pip install -r requirements-phrase-render.txt
```

[Verovio 공식 Python 안내](https://book.verovio.org/installing-or-building-from-sources/python.html)를 사용하며
고정 버전은 Verovio 6.3.0·CairoSVG 2.9.1이다. CairoSVG에는 시스템 Cairo가 필요하다.
macOS에서는 Cairo 설치 후 해당 라이브러리 위치를 설정한다. 현재 Apple Silicon 환경에서는 다음과 같이 실행했다.

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/lib \
MPLCONFIGDIR=/tmp/classicfy-matplotlib \
.venv/bin/python classicfy-ai/scripts/extract_phrase_candidates.py \
  --data ../datasets --out /tmp/classicfy-phrase-render-recheck --render
```

다른 OS/설치 경로에서는 시스템 Cairo 위치에 맞춘다. 조판 옵션과 실제 버전은 [renderer.json](renderer.json)에 저장했다.

## 관련 테스트

```bash
PYTHONPATH=classicfy-ai/src MPLCONFIGDIR=/tmp/classicfy-matplotlib \
.venv/bin/python -m unittest discover \
  -s classicfy-ai/tests/unit/preprocessing -p 'test_phrase_candidates.py' -v

PYTHONPATH=classicfy-ai/src MPLCONFIGDIR=/tmp/classicfy-matplotlib \
.venv/bin/python -m unittest discover \
  -s classicfy-ai/tests/integration -p 'test_phrase_candidate_pipeline.py' -v
```

합성 14개는 데이터 없이 실행된다. 실제 pilot 통합 3개는 대상 XML/MIDI/match가 필요하며
데이터가 없으면 skip될 수 있다. 이번 고정 입력 환경에서는 모두 skip 없이 통과했다.
