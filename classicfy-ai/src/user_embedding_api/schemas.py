"""The backend owns catalog lookup, embedding provenance and like state."""

from datetime import datetime
from enum import Enum
import math
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator


DIMENSIONS = 64
MAX_LIKES = 10000
FiniteNumber = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Vector = Annotated[list[FiniteNumber], Field(min_length=DIMENSIONS, max_length=DIMENSIONS)]
Version = Annotated[str, Field(strict=True, min_length=1, max_length=128, pattern=r"^\S+$")]


class EraGroup(str, Enum):
    BAROQUE = "baroque"
    CLASSICAL_TRANSITION = "classical_transition"
    ROMANTIC_IMPRESSIONIST = "romantic_impressionist"


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LikedPerformance(APIModel):
    performanceID: Annotated[str, Field(pattern=r"^performance_[0-9]+$")]
    eraGroup: EraGroup
    performanceEmbedding: Vector
    createdAt: AwareDatetime

    @field_validator("createdAt", mode="before")
    @classmethod
    def requireTimestamp(cls, value):
        if not isinstance(value, (str, datetime)):
            raise ValueError("createdAt must be an ISO-8601 timestamp with a timezone")
        return value

    @field_validator("performanceEmbedding")
    @classmethod
    def requireUnitVector(cls, value):
        norm = math.hypot(*value)
        if not math.isfinite(norm) or abs(norm - 1.0) > 1e-3:
            raise ValueError("performanceEmbedding must be a nonzero unit vector (norm tolerance 0.001)")
        return value


class CalculateRequest(APIModel):
    userID: UUID
    requestVersion: Annotated[int, Field(strict=True, ge=0)]
    embeddingVersion: Version
    likes: Annotated[list[LikedPerformance], Field(max_length=MAX_LIKES)]

    @model_validator(mode="after")
    def requireUniqueLikes(self):
        ids = [like.performanceID for like in self.likes]
        if len(ids) != len(set(ids)):
            raise ValueError("likes must contain each performanceID at most once")
        return self


class EraEmbedding(APIModel):
    eraGroup: EraGroup
    status: Literal["ready", "no_likes"]
    embedding: Vector | None
    likeCount: Annotated[int, Field(ge=0)]
    latestLikeAt: AwareDatetime | None


class CalculateResponse(APIModel):
    userID: UUID
    requestVersion: int
    embeddingVersion: str
    logicVersion: str
    decayAlpha: float
    computedAt: AwareDatetime
    eras: list[EraEmbedding]


class ErrorDetail(APIModel):
    location: str
    message: str


class ErrorBody(APIModel):
    code: str
    message: str
    details: list[ErrorDetail] = Field(default_factory=list)


class ErrorResponse(APIModel):
    error: ErrorBody


class HealthResponse(APIModel):
    status: Literal["ok"] = "ok"
    service: str = "classicfy-user-embedding"
    logicVersion: str
