"""FastAPI application exposing the ClinVerify HTTP endpoints.

Contracts are defined in ``docs/specifications.md``. ``POST /ingest`` ingests
regulatory PDFs (Phase 1) and ``POST /query`` runs hybrid retrieval, reranking,
grounded generation, and citation auditing (Phases 2-3).
"""

from __future__ import annotations

import logging
import time
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
from clinverify.errors import (
    EmptyCollectionError,
    InvalidDocumentError,
    RerankerUnavailableError,
    UpstreamLLMError,
    VectorStoreError,
)
from clinverify.generator import (
    GenerationService,
    GenerationServiceProtocol,
    OpenAIChatClient,
)
from clinverify.ingestion import (
    FastEmbedEmbedder,
    IngestionService,
    IngestionServiceProtocol,
)
from clinverify.logging import REQUEST_LOGGER, configure_logging, log_event
from clinverify.retrieval import (
    CrossEncoderReranker,
    Mode,
    RetrievalResult,
    RetrievalService,
    RetrievalServiceProtocol,
)

ALLOWED_CONTENT_TYPES = frozenset({"application/pdf", "application/octet-stream"})
PDF_MAGIC = b"%PDF-"

logger = logging.getLogger(REQUEST_LOGGER)


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
    """``POST /query`` response with the grounded answer and audit outcome."""

    query: str
    answer: str
    sources: list[Source]
    telemetry: Telemetry
    citation_verified: bool


class ErrorResponse(BaseModel):
    """Uniform error envelope returned for all non-2xx responses."""

    error: str
    detail: str
    request_id: str


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", str(uuid.uuid4()))


def _error_response(request: Request, status_code: int, error: str, detail: str) -> JSONResponse:
    fields = dict(getattr(request.state, "log_fields", {}) or {})
    fields["error_code"] = error
    request.state.log_fields = fields
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


def get_generation_service(request: Request) -> GenerationServiceProtocol:
    """Resolve the generation service from application state."""

    service = getattr(request.app.state, "generation", None)
    if service is None:  # pragma: no cover - defensive; set in lifespan
        raise RuntimeError("generation service is not initialized")
    return service  # type: ignore[no-any-return]


IngestionDep = Annotated[IngestionServiceProtocol, Depends(get_ingestion_service)]
RetrievalDep = Annotated[RetrievalServiceProtocol, Depends(get_retrieval_service)]
GenerationDep = Annotated[GenerationServiceProtocol, Depends(get_generation_service)]


def create_app(
    *,
    settings: Settings | None = None,
    ingestion: IngestionServiceProtocol | None = None,
    retrieval: RetrievalServiceProtocol | None = None,
    generation: GenerationServiceProtocol | None = None,
) -> FastAPI:
    """Build the FastAPI application.

    Args:
        settings: Optional settings override (defaults to environment).
        ingestion: Optional pre-built ingestion service. When omitted, the
            service is constructed during application startup.
        retrieval: Optional pre-built retrieval service. When omitted, the
            service is constructed during application startup.
        generation: Optional pre-built generation service. When omitted, the
            service is constructed during application startup.
    """

    app_settings = settings or get_settings()
    configure_logging()

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
        if application.state.generation is None:
            application.state.generation = GenerationService(
                app_settings,
                OpenAIChatClient(
                    app_settings.generator_model,
                    app_settings.generator_temperature,
                    app_settings.openai_api_key,
                ),
            )
        yield

    app = FastAPI(title="ClinVerify API", version="0.1.0", lifespan=lifespan)
    app.state.settings = app_settings
    app.state.ingestion = ingestion
    app.state.retrieval = retrieval
    app.state.generation = generation

    @app.middleware("http")
    async def attach_request_id(request: Request, call_next: Any) -> Any:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        request.state.log_fields = {}
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        log_event(
            logger,
            "request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round((time.perf_counter() - start) * 1000.0, 2),
            **request.state.log_fields,
        )
        return response

    @app.exception_handler(InvalidDocumentError)
    async def handle_invalid_document(request: Request, exc: InvalidDocumentError) -> JSONResponse:
        return _error_response(request, 400, "invalid_document", str(exc))

    @app.exception_handler(EmptyCollectionError)
    async def handle_empty_collection(request: Request, exc: EmptyCollectionError) -> JSONResponse:
        return _error_response(request, 400, "empty_collection", str(exc))

    @app.exception_handler(UpstreamLLMError)
    async def handle_upstream_llm(request: Request, exc: UpstreamLLMError) -> JSONResponse:
        return _error_response(request, 500, "upstream_llm_error", str(exc))

    @app.exception_handler(VectorStoreError)
    async def handle_vector_store(request: Request, exc: VectorStoreError) -> JSONResponse:
        return _error_response(request, 500, "qdrant_unavailable", str(exc))

    @app.exception_handler(RerankerUnavailableError)
    async def handle_reranker(request: Request, exc: RerankerUnavailableError) -> JSONResponse:
        return _error_response(request, 503, "reranker_unavailable", str(exc))

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
        retrieval: RetrievalDep,
        generation: GenerationDep,
    ) -> QueryResponse:
        result: RetrievalResult = retrieval.retrieve(
            query=body.query, mode=body.mode, top_k=body.top_k
        )
        answer = generation.generate(query=body.query, sources=result.sources)
        total_ms = result.retrieval_ms + result.rerank_ms + answer.generation_ms
        request.state.log_fields = {
            "mode": body.mode,
            "top_k": body.top_k,
            "candidates_prefetch": result.candidates_prefetch,
            "retrieval_ms": round(result.retrieval_ms, 2),
            "rerank_ms": round(result.rerank_ms, 2),
            "generation_ms": round(answer.generation_ms, 2),
            "total_ms": round(total_ms, 2),
            "citation_verified": answer.citation_verified,
            "citation_unresolved_count": answer.unresolved_count,
            "abstained": answer.abstained,
        }
        return QueryResponse(
            query=body.query,
            answer=answer.answer,
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
                generation_ms=answer.generation_ms,
                total_ms=total_ms,
            ),
            citation_verified=answer.citation_verified,
        )

    return app


app = create_app()
