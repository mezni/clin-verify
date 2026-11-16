"""FastAPI application exposing the ClinVerify HTTP endpoints.

Contracts are defined in ``docs/specifications.md``. ``POST /ingest`` is
implemented (Phase 1) and ``POST /query`` performs retrieval and reranking
(Phase 2); grounded generation and citation auditing are added in Phase 3.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from clinverify.config import DOC_ID_PATTERN, Settings, get_settings
from clinverify.errors import EmptyCollectionError, InvalidDocumentError
from clinverify.ingestion import (
    FastEmbedEmbedder,
    IngestionService,
    IngestionServiceProtocol,
)
from clinverify.retrieval import (
    CrossEncoderReranker,
    Mode,
    RetrievalResult,
    RetrievalService,
    RetrievalServiceProtocol,
)

ALLOWED_CONTENT_TYPES = frozenset({"application/pdf", "application/octet-stream"})
PDF_MAGIC = b"%PDF-"


class IngestResponse(BaseModel):
    """Successful ``POST /ingest`` response."""

    status: str = "success"
    doc_id: str
    pages_processed: int
    chunks_indexed: int


class QueryRequest(BaseModel):
    """``POST /query`` request body."""

    query: str = Field(min_length=3, max_length=2000)
    mode: Mode = "hybrid_rerank"
    top_k: int = Field(default=5, ge=1, le=20)


class Source(BaseModel):
    """A retrieved source chunk."""

    text: str
    citation: str
    score: float
    rerank_score: float | None = None


class Telemetry(BaseModel):
    """Per-request stage timings."""

    retrieval_ms: float
    rerank_ms: float
    generation_ms: float = 0.0
    total_ms: float


class QueryResponse(BaseModel):
    """``POST /query`` response.

    ``answer`` and ``citation_verified`` are populated in Phase 3; until then
    they are ``None`` and only retrieval sources are returned.
    """

    query: str
    answer: str | None = None
    sources: list[Source]
    telemetry: Telemetry
    citation_verified: bool | None = None


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


def get_retrieval_service(request: Request) -> RetrievalServiceProtocol:
    """Resolve the retrieval service from application state."""

    service = getattr(request.app.state, "retrieval", None)
    if service is None:  # pragma: no cover - defensive; set in lifespan
        raise RuntimeError("retrieval service is not initialized")
    return service  # type: ignore[no-any-return]


IngestionDep = Annotated[IngestionServiceProtocol, Depends(get_ingestion_service)]
RetrievalDep = Annotated[RetrievalServiceProtocol, Depends(get_retrieval_service)]


def create_app(
    *,
    settings: Settings | None = None,
    ingestion: IngestionServiceProtocol | None = None,
    retrieval: RetrievalServiceProtocol | None = None,
) -> FastAPI:
    """Build the FastAPI application.

    Args:
        settings: Optional settings override (defaults to environment).
        ingestion: Optional pre-built ingestion service. When omitted, the
            service is constructed during application startup.
        retrieval: Optional pre-built retrieval service. When omitted, the
            service is constructed during application startup.
    """

    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        if application.state.ingestion is None or application.state.retrieval is None:
            client = QdrantClient(url=app_settings.qdrant_url, api_key=app_settings.qdrant_api_key)
            embedder = FastEmbedEmbedder(app_settings.dense_model, app_settings.sparse_model)
            if application.state.ingestion is None:
                application.state.ingestion = IngestionService(app_settings, client, embedder)
            if application.state.retrieval is None:
                application.state.retrieval = RetrievalService(
                    app_settings,
                    client,
                    embedder,
                    CrossEncoderReranker(app_settings.reranker_model),
                )
        yield

    app = FastAPI(title="ClinVerify API", version="0.1.0", lifespan=lifespan)
    app.state.settings = app_settings
    app.state.ingestion = ingestion
    app.state.retrieval = retrieval

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

    @app.exception_handler(EmptyCollectionError)
    async def handle_empty_collection(request: Request, exc: EmptyCollectionError) -> JSONResponse:
        return _error_response(request, 400, "empty_collection", str(exc))

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

    @app.post("/query", response_model=QueryResponse)
    async def query(
        request: Request,
        body: QueryRequest,
        service: RetrievalDep,
    ) -> QueryResponse:
        result: RetrievalResult = service.retrieve(
            query=body.query, mode=body.mode, top_k=body.top_k
        )
        total_ms = result.retrieval_ms + result.rerank_ms
        return QueryResponse(
            query=body.query,
            answer=None,
            sources=[
                Source(
                    text=chunk.text,
                    citation=chunk.citation,
                    score=chunk.score,
                    rerank_score=chunk.rerank_score,
                )
                for chunk in result.sources
            ],
            telemetry=Telemetry(
                retrieval_ms=result.retrieval_ms,
                rerank_ms=result.rerank_ms,
                generation_ms=0.0,
                total_ms=total_ms,
            ),
            citation_verified=None,
        )

    return app


app = create_app()
