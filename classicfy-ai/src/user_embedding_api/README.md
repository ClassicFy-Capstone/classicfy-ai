# 좋아요 순서 기반 유저 임베딩 API

백엔드가 전달한 현재 좋아요 전체 목록으로 시대별 유저 임베딩을 계산한다.
DB·카탈로그·체크포인트를 읽지 않는다. 초기 구현은 FastAPI + Pydantic + NumPy이며
모델 학습이나 GPU가 필요하지 않다. 관련 이슈는 GitHub #54, 작업 브랜치는
`feat/#21-user-embedding-api`다. #53 제목 메타데이터 작품 추천과 별개다.

## 실행

저장소 루트에서 Python 3.14 기준:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-api.txt
.venv/bin/python -m uvicorn user_embedding_api.app:app --app-dir classicfy-ai/src --host 127.0.0.1 --port 8000
```

- 계산: `POST /v1/user-embeddings/calculate`
- 상태: `GET /health`
- 대화형 명세: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`
- 백엔드 전달용 정적 명세: [openapi.json](openapi.json). 모델·라우트 변경 시 다시 내보낸다.

```sh
curl --fail-with-body http://127.0.0.1:8000/v1/user-embeddings/calculate \
  -H 'Content-Type: application/json' \
  --data-binary @classicfy-ai/src/user_embedding_api/examples/request.json
```

예제 파일의 벡터는 계산 검증용 단위 기저 벡터이며 실제 연주 임베딩이 아니다.

## 요청 계약

| 필드 | 타입·규칙 |
| --- | --- |
| `userID` | UUID 문자열 |
| `requestVersion` | 0 이상의 정수. 좋아요 상태 변경마다 백엔드에서 증가 |
| `embeddingVersion` | 공백 없는 버전 문자열. 요청 전체가 같은 임베딩 버전임을 백엔드가 보장 |
| `likes` | 현재 유효한 좋아요 전체 목록, 0~10,000개 |
| `likes[].performanceID` | `performance_숫자` 형식, 요청 내 중복 불가 |
| `likes[].eraGroup` | `baroque`, `classical_transition`, `romantic_impressionist` |
| `likes[].performanceEmbedding` | 유한한 JSON 숫자 64개. `e` 또는 동일한 `performanceEmbedding` 단위 벡터 |
| `likes[].createdAt` | 시간대 포함 ISO-8601 시각. 예: `2026-11-26T12:00:00+09:00` |

없는 필드·추가 필드·숫자 문자열·bool 벡터·중복 ID는 오류다.
입력 벡터의 L2 norm은 1에서 0.001 이내여야 하며, 계산 직전 다시 정규화하여
float32 반올림에 따른 작은 크기 차이를 제거한다. 영벡터는 허용하지 않는다.
시각은 순서에만 사용하며 미래 시각인지 확인하지 않는다. 정확한 좋아요 시각은 백엔드가 책임진다.

카탈로그를 조회하지 않으므로 연주 ID의 실제 존재 여부, 시대의 사실 여부,
벡터와 ID의 대응, 버전의 진위는 검증할 수 없다. 백엔드는 같은 버전의 데이터를 JOIN해서 보내야 한다.

## 계산 계약

1. 세 시대를 분리한다. 가중치 순위는 각 시대에서 0부터 다시 시작한다.
2. `createdAt`을 UTC로 비교하여 최신순 정렬한다. 같은 시각이면 `performanceID` 문자열 오름차순이다.
3. 순위 `i`의 가중치는 `alpha ** i`. 기본 `alpha = 0.95`다.
4. `mean = sum(weight * unitEmbedding) / sum(weight)`를 계산한다.
5. `mean / norm(mean)`을 64차원 결과로 반환한다. 임베딩은 두 자리로 반올림하지 않는다.

실제 좋아요 사이의 시간 간격은 가중치에 영향을 주지 않는다.
동일 입력·설정이면 벡터는 동일하고, `computedAt`만 요청 처리 시각에 따라 달라진다.

- 좋아요 0개: 해당 시대에 `status=no_likes`, `embedding=null`, `likeCount=0`.
- 좋아요 1개: 정규화한 해당 연주의 벡터.
- 상쇄되어 가중평균 norm이 `1e-12` 이하: 요청 전체를 422 오류로 반환한다. 임의 벡터나 부분 성공을 만들지 않는다.
- 좋아요 취소: 남아 있는 전체 목록을 다시 전송한다. 서버에 이전 사용자 상태는 저장하지 않는다.

## 응답 계약

최상위는 `userID`, `requestVersion`, `embeddingVersion`, `logicVersion`, `decayAlpha`,
`computedAt`, `eras`다. `eras`는 항상 바로크 → 고전·전환기 → 낭만·인상주의 순서의 3개 항목이다.

각 시대 항목:

| 필드 | 의미 |
| --- | --- |
| `eraGroup` | 시대 코드 |
| `status` | `ready` 또는 `no_likes` |
| `embedding` | 64개 실수 또는 null |
| `likeCount` | 해당 시대 좋아요 개수 |
| `latestLikeAt` | 가장 최근 좋아요 시각(UTC), 없으면 null |

API는 `requestVersion`과 `embeddingVersion`을 그대로 반환한다.
백엔드는 저장 직전에 사용자 좋아요 버전·임베딩 버전이 여전히 같은지 확인하고,
맞는 결과만 `(userID, eraGroup)` 기준으로 저장한다. `no_likes`인 시대는 기존 벡터를 삭제한다.
422 또는 통신 실패에서는 성공 결과처럼 기존 벡터를 덮지 않는다.
`computedAt`은 좋아요 상태 버전의 대체물이 아니다.

## 오류·설정

오류 형식은 `{"error":{"code":"...","message":"...","details":[...]}}`다.
details 항목은 `location`, `message`를 갖고 원본 벡터나 입력 전체를 노출하지 않는다.

| HTTP | code | 의미 |
| --- | --- | --- |
| 401 | `UNAUTHORIZED` | 설정된 API 키가 누락되거나 다름 |
| 422 | `VALIDATION_ERROR` | 요청 형식·값·차원·중복 오류 |
| 422 | `DEGENERATE_USER_EMBEDDING` | 가중평균이 상쇄되어 방향을 정의할 수 없음 |

| 환경 변수 | 기본값 | 의미 |
| --- | --- | --- |
| `USER_EMBEDDING_DECAY_ALPHA` | `0.95` | 유한한 값, `0 < alpha <= 1` |
| `AI_API_KEY` | 미설정 | 설정 시 계산 API에 `X-API-Key` 헤더 필요 |

로컬 기본 실행은 인증 없이 가능하다. 서버 간 배포 시 키를 환경 변수로 주입하고
백엔드가 헤더로 전달한다. `/health`, `/docs`, `/openapi.json`은 이 키 인증의 대상이 아니다.
접근 범위·TLS·요청 크기 제한은 배포 환경에서 설정한다. API는 외부로 배포하지 않았다.

## 코드와 검증

- `schemas.py`: 요청·응답 및 값 검증
- `service.py`: 정렬·가중평균·정규화
- `config.py`: 환경 설정
- `app.py`: HTTP 라우팅·오류·선택적 API 키

```sh
.venv/bin/python -m pip install -r requirements-api-dev.txt
PYTHONPATH=classicfy-ai/src .venv/bin/python -m pytest classicfy-ai/tests/api -q
```

가중치·시대 분리·시간 간격 불변성·동률·취소·결측/잘못된 벡터·인증·상쇄를 검증한다.
기존 MIDI/학습 파이프라인 의존성을 설치하거나 전체 학습 테스트를 실행할 필요는 없다.

2026-11-26 KST 검증: Python 3.14에서 API/계산 테스트 39개 통과.
로컬 Uvicorn 실제 HTTP로 예제 요청과 최종 카탈로그의 시대별 실물 벡터 3개를 확인했다.
외부 백엔드와의 실연동·배포는 아직 수행하지 않았다.

참고: [FastAPI 요청 모델](https://fastapi.tiangolo.com/tutorial/body/),
[응답 모델](https://fastapi.tiangolo.com/tutorial/response-model/),
[API 테스트](https://fastapi.tiangolo.com/tutorial/testing/).
