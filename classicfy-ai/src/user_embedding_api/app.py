"""Run with uvicorn user_embedding_api.app:app --app-dir classicfy-ai/src."""

import secrets

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from . import LOGIC_VERSION
from .config import Settings
from .schemas import CalculateRequest, CalculateResponse, ErrorResponse, HealthResponse
from .service import DegenerateEmbeddingError, calculateUserEmbeddings


def createApp(settings: Settings | None = None) -> FastAPI:
    settings = settings if settings is not None else Settings.fromEnv()
    application = FastAPI(
        title="ClassicFy User Embedding API", version="0.1.0",
        description="Backend-supplied likes → rank-decayed, era-specific 64D user embeddings.",
    )
    apiKeyHeader = APIKeyHeader(name="X-API-Key", auto_error=False)

    def requireApiKey(apiKey: str | None = Depends(apiKeyHeader)):
        if settings.apiKey is not None and (
            apiKey is None or not secrets.compare_digest(apiKey.encode(), settings.apiKey.encode())
        ):
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

    @application.exception_handler(RequestValidationError)
    async def handleValidationError(request: Request, error: RequestValidationError):
        # Avoid echoing vectors, user input or non-JSON exception contexts.
        return JSONResponse(status_code=422, content={"error": {
            "code": "VALIDATION_ERROR", "message": "Invalid request body",
            "details": [{"location": ".".join(map(str, item["loc"])), "message": item["msg"]}
                        for item in error.errors()],
        }})

    @application.exception_handler(DegenerateEmbeddingError)
    async def handleDegenerateEmbedding(request: Request, error: DegenerateEmbeddingError):
        return JSONResponse(status_code=422, content={"error": {
            "code": "DEGENERATE_USER_EMBEDDING",
            "message": "Weighted vectors cancel out; no user embedding can be normalized",
            "details": [{"location": "eras." + error.eraGroup.value, "message": "Near-zero vector"}],
        }})

    @application.exception_handler(HTTPException)
    async def handleHttpError(request: Request, error: HTTPException):
        return JSONResponse(status_code=error.status_code, headers=error.headers, content={"error": {
            "code": "UNAUTHORIZED" if error.status_code == 401 else "HTTP_ERROR",
            "message": str(error.detail), "details": [],
        }})

    @application.get("/health", response_model=HealthResponse, tags=["health"])
    def health():
        return HealthResponse(logicVersion=LOGIC_VERSION)

    @application.post(
        "/v1/user-embeddings/calculate", response_model=CalculateResponse,
        dependencies=[Depends(requireApiKey)], tags=["user-embeddings"],
        responses={401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    )
    def calculate(request: CalculateRequest):
        return calculateUserEmbeddings(request, settings.decayAlpha)

    return application


app = createApp()
