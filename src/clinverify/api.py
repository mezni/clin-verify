"""FastAPI application exposing the ClinVerify HTTP endpoints.

Contracts are defined in ``docs/specifications.md``. Phase 1 implements
``POST /ingest``; ``POST /query`` arrives in Phases 2-3.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from clinverify.config import DOC_ID_PATTERN, Settings, get_settings
from clinverify.errors import InvalidDocumentError
from clinverify.ingestion import (
    IngestionServiceProtocol,
    build_ingestion_service,
)

ALLOWED_CONTENT_TYPES = frozenset({"application/pdf", "application/octet-stream"})
PDF_MAGIC = b"%PDF-"


class IngestResponse(BaseModel):
    """Successful ``POST /ingest`` response."""

    status: str = "success"
    doc_id: str
    pages_processed: int
    chunks_indexed: int


class ErrorResponse(BaseModel):
    """Uniform error envelope returned for all non-2xx responses."""

    error: str
    detail: str
    request_id: str


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", str(uuid.uuid4()))


def _error_response(request: Request, status_code: int, error: str, detail: str) -> JSONResponse:
    payload = ErrorResponse(error=error, detail=detail, request_id=_request_id(request))
    return JSONResponse(status_code=status_code, content=payload.model_dump())


def get_ingestion_service(request: Request) -> IngestionServiceProtocol:
    """Resolve the ingestion service from application state."""

    service = getattr(request.app.state, "ingestion", None)
    if service is None:  # pragma: no cover - defensive; set in lifespan
        raise RuntimeError("ingestion service is not initialized")
    return service  # type: ignore[no-any-return]


IngestionDep = Annotated[IngestionServiceProtocol, Depends(get_ingestion_service)]


def create_app(
    *,
    settings: Settings | None = None,
    ingestion: IngestionServiceProtocol | None = None,
) -> FastAPI:
    """Build the FastAPI application.

    Args:
        settings: Optional settings override (defaults to environment).
        ingestion: Optional pre-built ingestion service. When omitted, the
            service is constructed during application startup.
    """

    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        if application.state.ingestion is None:
            application.state.ingestion = build_ingestion_service(app_settings)
        yield

    app = FastAPI(title="ClinVerify API", version="0.1.0", lifespan=lifespan)
    app.state.settings = app_settings
    app.state.ingestion = ingestion

    @app.middleware("http")
    async def attach_request_id(request: Request, call_next: Any) -> Any:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(InvalidDocumentError)
    async def handle_invalid_document(request: Request, exc: InvalidDocumentError) -> JSONResponse:
        return _error_response(request, 400, "invalid_document", str(exc))

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return _error_response(request, 422, "validation_error", str(exc.errors()))

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        return _error_response(request, 500, "internal_error", str(exc))

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(
        request: Request,
        file: Annotated[UploadFile, File()],
        doc_id: Annotated[str, Query(pattern=DOC_ID_PATTERN)],
        service: IngestionDep,
    ) -> IngestResponse:
        content_type = file.content_type or ""
        if content_type not in ALLOWED_CONTENT_TYPES:
            raise InvalidDocumentError(f"Unsupported content type: {content_type or 'unknown'}")

        data = await file.read()
        if not data:
            raise InvalidDocumentError("Uploaded file is empty")

        max_bytes = app_settings.max_upload_mb * 1024 * 1024
        if len(data) > max_bytes:
            raise InvalidDocumentError(f"File exceeds the {app_settings.max_upload_mb} MB limit")

        if not data.startswith(PDF_MAGIC):
            raise InvalidDocumentError("File is not a valid PDF")

        result = service.ingest(doc_id=doc_id, data=data)
        return IngestResponse(
            status="success",
            doc_id=result.doc_id,
            pages_processed=result.pages_processed,
            chunks_indexed=result.chunks_indexed,
        )

    return app


app = create_app()
