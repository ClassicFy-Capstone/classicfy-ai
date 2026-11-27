"""Contract and numerical behavior tests; no dataset, DB or network required."""

from datetime import datetime, timedelta, timezone
import math

from fastapi.testclient import TestClient
import pytest

from user_embedding_api.app import createApp
from user_embedding_api.config import Settings


PATH = "/v1/user-embeddings/calculate"


def vector(axis=0, sign=1.0):
    result = [0.0] * 64
    result[axis] = sign
    return result


def like(identifier=1, axis=0, era="baroque", when="2026-11-26T03:00:00Z"):
    return {"performanceID": f"performance_{identifier}", "eraGroup": era,
            "performanceEmbedding": vector(axis), "createdAt": when}


def payload(likes=None):
    return {"userID": "550e8400-e29b-41d4-a716-446655440000", "requestVersion": 12,
            "embeddingVersion": "axy-final-v1", "likes": [] if likes is None else likes}


@pytest.fixture
def client():
    with TestClient(createApp(Settings())) as c:
        yield c


def result(client, request):
    response = client.post(PATH, json=request)
    assert response.status_code == 200, response.text
    return response.json()


def eraResult(response, era="baroque"):
    return next(r for r in response["eras"] if r["eraGroup"] == era)


def testHealthAndOpenAPI(client):
    assert client.get("/health").json()["status"] == "ok"
    schema = client.get("/openapi.json").json()
    assert PATH in schema["paths"]
    vectorSchema = schema["components"]["schemas"]["LikedPerformance"]["properties"]["performanceEmbedding"]
    assert vectorSchema["minItems"] == vectorSchema["maxItems"] == 64
    assert client.get("/docs").status_code == 200


def testEmptyLikesExplicitlyClearEveryEra(client):
    response = result(client, payload())
    assert len(response["eras"]) == 3
    assert all(r["status"] == "no_likes" and r["embedding"] is None
               and r["likeCount"] == 0 and r["latestLikeAt"] is None for r in response["eras"])
    assert response["requestVersion"] == 12
    assert response["embeddingVersion"] == "axy-final-v1"


def testOneLikeEqualsItsVector(client):
    response = result(client, payload([like(axis=9)]))
    assert eraResult(response)["embedding"] == vector(9)
    assert eraResult(response)["likeCount"] == 1


def testRecentRankWeightsAndEraSeparation(client):
    request = payload([like(1, 0, when="2020-01-01T00:00:00Z"),
                       like(2, 1), like(3, 8, era="classical_transition")])
    response = result(client, request)
    expected = [.95 / math.sqrt(1 + .95**2), 1 / math.sqrt(1 + .95**2)] + [0.] * 62
    assert eraResult(response)["embedding"] == pytest.approx(expected)
    assert math.hypot(*eraResult(response)["embedding"]) == pytest.approx(1)
    assert eraResult(response, "classical_transition")["embedding"] == vector(8)
    assert eraResult(response, "romantic_impressionist")["embedding"] is None
    assert response["decayAlpha"] == .95


def testRequestOrderAndTimeGapsDoNotChangeRankWeight(client):
    first = payload([like(1, 0, when="2000-01-01T00:00:00Z"), like(2, 1)])
    second = payload([like(2, 1), like(1, 0, when="2026-11-26T02:59:59Z")])
    assert eraResult(result(client, first))["embedding"] == eraResult(result(client, second))["embedding"]


def testSameInstantOffsetsUseStableIdTiebreak(client):
    a = like(1, 0, when="2026-11-26T03:00:00Z")
    b = like(2, 1, when="2026-11-26T12:00:00+09:00")
    one = eraResult(result(client, payload([b, a])))
    two = eraResult(result(client, payload([a, b])))
    assert one == two
    assert one["embedding"][0] > one["embedding"][1]


def testUnlikeRecomputesWithoutStoredState(client):
    result(client, payload([like(1), like(2, axis=1)]))
    assert eraResult(result(client, payload([like(2, axis=1)])))["embedding"] == vector(1)
    assert eraResult(result(client, payload()))["embedding"] is None


@pytest.mark.parametrize("field,value", [
    ("performanceEmbedding", [0.0] * 63),
    ("performanceEmbedding", [0.0] * 65),
    ("performanceEmbedding", [0.0] * 64),
    ("performanceEmbedding", [2.0] + [0.] * 63),
    ("performanceEmbedding", ["1.0"] + [0.] * 63),
    ("performanceEmbedding", [True] + [0.] * 63),
    ("performanceEmbedding", [1e308] * 64),
    ("eraGroup", "unknown"),
    ("createdAt", "2026-11-26T12:00:00"),
    ("createdAt", 1234567890),
    ("performanceID", "02648"),
])
def testInvalidLikeReturnsStructured422(client, field, value):
    entry = like()
    entry[field] = value
    response = client.post(PATH, json=payload([entry]))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert "input" not in response.json()["error"]["details"][0]


@pytest.mark.parametrize("field,value", [
    ("userID", "not-a-uuid"), ("requestVersion", -1), ("requestVersion", "1"),
    ("requestVersion", True), ("embeddingVersion", ""), ("embeddingVersion", "two versions"),
])
def testInvalidEnvelope(client, field, value):
    request = payload()
    request[field] = value
    assert client.post(PATH, json=request).status_code == 422


def testDuplicatesAndUnexpectedFields(client):
    assert client.post(PATH, json=payload([like(), like()])).status_code == 422
    assert client.post(PATH, json={**payload(), "halfLifeDays": 14}).status_code == 422


@pytest.mark.parametrize("badNumber", ["NaN", "Infinity", "-Infinity"])
def testNonfiniteJSONNumbersReturn422(client, badNumber):
    import json
    body = json.dumps(payload([like()])).replace('1.0,', badNumber + ',', 1)
    response = client.post(PATH, content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def testMalformedJSON(client):
    response = client.post(PATH, content='{"likes":', headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def testConfiguredApiKey():
    with TestClient(createApp(Settings(apiKey="test-only-key"))) as c:
        assert c.get("/health").status_code == 200
        assert c.post(PATH, json=payload()).status_code == 401
        assert c.post(PATH, json=payload(), headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.post(PATH, json=payload(), headers={"X-API-Key": "test-only-key"}).status_code == 200


def testCancellationIsNotMistakenForNoLikes():
    a, b = like(1), like(2)
    b["performanceEmbedding"] = vector(sign=-1)
    with TestClient(createApp(Settings(decayAlpha=1))) as c:
        response = c.post(PATH, json=payload([a, b]))
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "DEGENERATE_USER_EMBEDDING"


def testFloat32RoundingAcceptedAndNormalized(client):
    entry = like()
    entry["performanceEmbedding"] = [1.0000001] + [0.] * 63
    assert eraResult(result(client, payload([entry])))["embedding"] == vector()


def testManyLikesRemainFinite(client):
    likes = [like(i, axis=i % 64,
                  when=(datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=i)).isoformat())
             for i in range(1000)]
    response = eraResult(result(client, payload(likes)))
    assert response["likeCount"] == 1000
    assert math.hypot(*response["embedding"]) == pytest.approx(1)


@pytest.mark.parametrize("alpha", [0, -1, 1.01, float("nan"), float("inf")])
def testInvalidConfig(alpha):
    with pytest.raises(ValueError):
        Settings(decayAlpha=alpha)


def testEnvironmentConfig(monkeypatch):
    monkeypatch.setenv("USER_EMBEDDING_DECAY_ALPHA", "0.98")
    monkeypatch.setenv("AI_API_KEY", "example")
    config = Settings.fromEnv()
    assert config.decayAlpha == .98 and config.apiKey == "example"
    assert "example" not in repr(config)
