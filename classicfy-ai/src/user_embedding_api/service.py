"""Rank-based weighting, independent of HTTP, catalog files and database access."""

from datetime import datetime, timezone
import math

import numpy as np

from . import LOGIC_VERSION
from .schemas import CalculateRequest, CalculateResponse, EraEmbedding, EraGroup


class DegenerateEmbeddingError(ValueError):
    def __init__(self, eraGroup: EraGroup):
        self.eraGroup = eraGroup
        super().__init__(f"Liked vectors cancel out in era {eraGroup.value}")


def calculateUserEmbeddings(request: CalculateRequest, decayAlpha: float = 0.95) -> CalculateResponse:
    if not math.isfinite(decayAlpha) or not 0 < decayAlpha <= 1:
        raise ValueError("decayAlpha must be in (0, 1]")
    results = []
    for era in EraGroup:
        likes = [like for like in request.likes if like.eraGroup == era]
        # Stable secondary key for equal timestamps, followed by newest first.
        # UTC conversion also makes equivalent offsets compare identically.
        likes.sort(key=lambda like: like.performanceID)
        likes.sort(key=lambda like: like.createdAt.astimezone(timezone.utc), reverse=True)
        if not likes:
            results.append(EraEmbedding(eraGroup=era, status="no_likes", embedding=None,
                                        likeCount=0, latestLikeAt=None))
            continue
        vectors = np.asarray([like.performanceEmbedding for like in likes], dtype=np.float64)
        # Accept float32 unit-vector rounding, then remove tiny magnitude variation.
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        weights = np.power(decayAlpha, np.arange(len(likes), dtype=np.float64))
        mean = (weights @ vectors) / weights.sum()
        norm = float(np.linalg.norm(mean))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise DegenerateEmbeddingError(era)
        results.append(EraEmbedding(
            eraGroup=era, status="ready", embedding=(mean / norm).tolist(),
            likeCount=len(likes), latestLikeAt=likes[0].createdAt.astimezone(timezone.utc),
        ))
    return CalculateResponse(
        userID=request.userID, requestVersion=request.requestVersion,
        embeddingVersion=request.embeddingVersion, logicVersion=LOGIC_VERSION,
        decayAlpha=decayAlpha, computedAt=datetime.now(timezone.utc), eras=results,
    )
