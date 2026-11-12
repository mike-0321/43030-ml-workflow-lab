"""Local teaching API with authentication, input limits and integrity readiness.

Readiness and prediction fail closed when the on-disk model no longer matches
the trusted local manifest. This is an operational consistency control, not a
signature scheme, rate limiter, TLS gateway or complete authorisation system.
"""

import json
import os
from hmac import compare_digest
from pathlib import Path
from typing import Annotated, Any, Protocol, TypeAlias, TypedDict, cast

import joblib
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from learning_service.common import ARTIFACTS, sha256

MAX_BODY_BYTES = 4096
Pixel: TypeAlias = Annotated[float, Field(strict=True, ge=0, le=16, allow_inf_nan=False)]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pixels: Annotated[list[Pixel], Field(min_length=64, max_length=64)]


class PredictionResult(TypedDict):
    prediction: int
    confidence: float
    model_version: str
    model_sha256: str
    mlflow_run_id: str


class PredictionPipeline(Protocol):
    classes_: NDArray[np.int64]

    def predict_proba(self, pixels: NDArray[np.float64]) -> NDArray[np.float64]: ...


class ModelIntegrityError(RuntimeError):
    """The deployed artifact is absent or differs from the trusted manifest."""


class BodyLimitMiddleware:
    """Bound bytes before JSON parsing, including streamed ASGI request bodies."""

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > self.max_bytes:
                response = JSONResponse(
                    {"detail": "Request body exceeds 4096 bytes"}, status_code=413
                )
                return await response(scope, receive, send)
            body.extend(chunk)
            if not message.get("more_body", False):
                break
        supplied = False

        async def replay() -> Message:
            nonlocal supplied
            if not supplied:
                supplied = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        return await self.app(scope, replay, send)


class Engine:
    def __init__(self, artifact_dir: Path | None = None) -> None:
        artifact_dir = artifact_dir or ARTIFACTS
        self.manifest: dict[str, Any] = json.loads((artifact_dir / "manifest.json").read_text())
        self.model_path = artifact_dir / self.text_field("model_file")
        self.verify_artifact_integrity()
        # Only this exercise's trusted, locally trained artifact is loaded.
        # A checksum does not authenticate a file if both file and manifest are altered.
        self.pipeline = cast(PredictionPipeline, joblib.load(self.model_path))

    def text_field(self, name: str) -> str:
        value = self.manifest[name]
        if not isinstance(value, str):
            raise ValueError(f"Expected a string manifest field: {name}")
        return value

    def verify_artifact_integrity(self) -> None:
        try:
            current_hash = sha256(self.model_path)
        except OSError as exc:
            raise ModelIntegrityError("Model checksum cannot be verified") from exc
        if current_hash != self.text_field("model_sha256"):
            raise ModelIntegrityError("Model checksum does not match manifest")

    def identification(self) -> dict[str, str]:
        return {
            name: self.text_field(name)
            for name in ["model_version", "model_sha256", "mlflow_run_id"]
        }

    def predict(self, pixels: list[float]) -> PredictionResult:
        self.verify_artifact_integrity()
        probabilities = self.pipeline.predict_proba(np.asarray([pixels], dtype=np.float64))[0]
        index = int(np.argmax(probabilities))
        return {
            "prediction": int(self.pipeline.classes_[index]),
            "confidence": float(probabilities[index]),
            "model_version": self.text_field("model_version"),
            "model_sha256": self.text_field("model_sha256"),
            "mlflow_run_id": self.text_field("mlflow_run_id"),
        }


def create_app(api_key: str | None = None, artifact_dir: Path | None = None) -> FastAPI:
    expected_key = api_key if api_key is not None else os.environ.get("LAB_API_KEY")
    if not expected_key:
        raise RuntimeError("Set LAB_API_KEY at runtime; no production key is embedded in source")
    expected_key_bytes = expected_key.encode("utf-8")
    if artifact_dir is None and os.environ.get("LAB_ARTIFACT_DIR"):
        artifact_dir = Path(os.environ["LAB_ARTIFACT_DIR"])
    engine = Engine(artifact_dir=artifact_dir)
    app = FastAPI(title="Supplementary local digits learning API")
    app.add_middleware(BodyLimitMiddleware)
    app.state.engine = engine
    header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def authenticate(supplied: Annotated[str | None, Depends(header)]) -> None:
        if supplied is None or not compare_digest(supplied.encode("utf-8"), expected_key_bytes):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {"location": list(e["loc"]), "message": e["msg"], "type": e["type"]}
                    for e in exc.errors()
                ]
            },
        )

    @app.exception_handler(ModelIntegrityError)
    async def integrity_failure(_request: Request, _exc: ModelIntegrityError) -> JSONResponse:
        return JSONResponse(
            status_code=503, content={"detail": "Model artifact integrity check failed"}
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "scope": "local learning exercise", **engine.identification()}

    @app.get("/ready")
    def ready() -> dict[str, str]:
        engine.verify_artifact_integrity()
        return {"status": "ready", **engine.identification()}

    @app.post("/predict", dependencies=[Depends(authenticate)])
    def predict(request: PredictionRequest) -> PredictionResult:
        return engine.predict(request.pixels)

    return app
