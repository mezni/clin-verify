"""Unit tests for the ``POST /query`` HTTP contract."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient

from clinverify.api import create_app
from clinverify.config import Settings
from clinverify.errors import (
    EmptyCollectionError,
    RerankerUnavailableError,
    UpstreamLLMError,
    VectorStoreError,
)
from clinverify.generator import GenerationResult
from clinverify.ingestion import IngestResult
from clinverify.retrieval import Mode, RetrievalResult, RetrievedChunk

ANSWER = "The reduced dose is 2.5 mg BID [FDA_Apixaban_2025, p.4]."


class FakeIngestionService:
    def ingest(self, doc_id: str, data: bytes) -> IngestResult:
        return IngestResult(doc_id=doc_id, pages_processed=1, chunks_indexed=1)


class FakeRetrievalService:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, Mode, int]] = []

    def retrieve(self, query: str, mode: Mode, top_k: int) -> RetrievalResult:
        self.calls.append((query, mode, top_k))
        if self.error is not None:
            raise self.error
        sources = [
            RetrievedChunk(
                point_id="p1",
                doc_id="FDA_Apixaban_2025",
                page_num=4,
                citation="[FDA_Apixaban_2025, p.4]",
                text="Apixaban 2.5 mg BID",
                score=0.9,
                rerank_score=4.821 if mode == "hybrid_rerank" else None,
            )
        ]
        return RetrievalResult(sources=sources, retrieval_ms=114.2, rerank_ms=342.6)


class FakeGenerationService:
    def __init__(
        self,
        *,
        answer: str = ANSWER,
        citation_verified: bool = True,
        abstained: bool = False,
        generation_ms: float = 312.1,
        error: Exception | None = None,
    ) -> None:
        self._answer = answer
        self._citation_verified = citation_verified
        self._abstained = abstained
        self._generation_ms = generation_ms
        self.error = error
        self.calls: list[tuple[str, list[RetrievedChunk]]] = []

    def generate(self, query: str, sources: Sequence[RetrievedChunk]) -> GenerationResult:
        self.calls.append((query, list(sources)))
        if self.error is not None:
            raise self.error
        return GenerationResult(
            answer=self._answer,
            citation_verified=self._citation_verified,
            abstained=self._abstained,
            unresolved_count=0,
            generation_ms=self._generation_ms,
        )


@contextmanager
def _client(
    retrieval: FakeRetrievalService,
    generation: FakeGenerationService | None = None,
    **settings_overrides: Any,
) -> Iterator[TestClient]:
    settings = Settings(_env_file=None, **settings_overrides)  # type: ignore[call-arg]
    app = create_app(
        settings=settings,
        ingestion=FakeIngestionService(),
        retrieval=retrieval,
        generation=generation or FakeGenerationService(),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def test_query_hybrid_returns_answer_and_sources() -> None:
    retrieval = FakeRetrievalService()
    generation = FakeGenerationService()
    with _client(retrieval, generation) as client:
        response = client.post("/query", json={"query": "What is the dose?", "top_k": 5})

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "What is the dose?"
    assert body["answer"] == ANSWER
    assert body["citation_verified"] is True
    assert body["sources"][0]["citation"] == "[FDA_Apixaban_2025, p.4]"
    assert body["sources"][0]["rerank_score"] == 4.821
    assert body["telemetry"]["retrieval_ms"] == 114.2
    assert body["telemetry"]["generation_ms"] == 312.1
    assert body["telemetry"]["total_ms"] == 114.2 + 342.6 + 312.1
    assert retrieval.calls[0][1] == "hybrid_rerank"
    assert generation.calls[0][1][0].doc_id == "FDA_Apixaban_2025"


def test_query_dense_only_mode() -> None:
    retrieval = FakeRetrievalService()
    with _client(retrieval) as client:
        response = client.post(
            "/query",
            json={"query": "What is the dose?", "mode": "dense_only"},
        )

    assert response.status_code == 200
    assert response.json()["sources"][0]["rerank_score"] is None
    assert retrieval.calls[0][1] == "dense_only"


def test_query_abstention_returns_insufficient_context() -> None:
    generation = FakeGenerationService(
        answer="INSUFFICIENT_CONTEXT", citation_verified=False, abstained=True
    )
    with _client(FakeRetrievalService(), generation) as client:
        response = client.post("/query", json={"query": "What is the dose?"})

    assert response.status_code == 200
    assert response.json()["answer"] == "INSUFFICIENT_CONTEXT"
    assert response.json()["citation_verified"] is False


def test_query_top_k_bounds_return_422() -> None:
    with _client(FakeRetrievalService()) as client:
        too_small = client.post("/query", json={"query": "dose?", "top_k": 0})
        too_large = client.post("/query", json={"query": "dose?", "top_k": 21})

    assert too_small.status_code == 422
    assert too_small.json()["error"] == "validation_error"
    assert too_large.status_code == 422


def test_query_short_text_returns_422() -> None:
    with _client(FakeRetrievalService()) as client:
        response = client.post("/query", json={"query": "hi"})

    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_query_invalid_mode_returns_422() -> None:
    with _client(FakeRetrievalService()) as client:
        response = client.post("/query", json={"query": "dose?", "mode": "invalid"})

    assert response.status_code == 422


def test_query_empty_collection_returns_400() -> None:
    service = FakeRetrievalService(error=EmptyCollectionError("nothing ingested"))
    with _client(service) as client:
        response = client.post("/query", json={"query": "What is the dose?"})

    assert response.status_code == 400
    assert response.json()["error"] == "empty_collection"


def test_query_upstream_llm_error_returns_500() -> None:
    generation = FakeGenerationService(error=UpstreamLLMError("timeout"))
    with _client(FakeRetrievalService(), generation) as client:
        response = client.post("/query", json={"query": "What is the dose?"})

    assert response.status_code == 500
    assert response.json()["error"] == "upstream_llm_error"


def test_query_vector_store_error_returns_500() -> None:
    retrieval = FakeRetrievalService(error=VectorStoreError("connection refused"))
    with _client(retrieval) as client:
        response = client.post("/query", json={"query": "What is the dose?"})

    assert response.status_code == 500
    assert response.json()["error"] == "qdrant_unavailable"


def test_query_reranker_error_returns_503() -> None:
    retrieval = FakeRetrievalService(error=RerankerUnavailableError("model missing"))
    with _client(retrieval) as client:
        response = client.post("/query", json={"query": "What is the dose?"})

    assert response.status_code == 503
    assert response.json()["error"] == "reranker_unavailable"


def test_query_emits_structured_request_log(caplog: Any) -> None:
    retrieval = FakeRetrievalService()
    with caplog.at_level(logging.INFO, logger="clinverify.request"), _client(retrieval) as client:
        response = client.post(
            "/query",
            json={"query": "What is the dose?"},
            headers={"X-Request-ID": "trace-42"},
        )

    records = [r for r in caplog.records if r.name == "clinverify.request"]
    fields = records[-1].extra_fields
    assert response.headers["X-Request-ID"] == "trace-42"
    assert fields["request_id"] == "trace-42"
    assert fields["status_code"] == 200
    assert fields["mode"] == "hybrid_rerank"
    assert fields["top_k"] == 5
    assert fields["citation_verified"] is True
    assert fields["abstained"] is False
    assert fields["total_ms"] == pytest.approx(114.2 + 342.6 + 312.1)


def test_error_emits_error_code_in_log(caplog: Any) -> None:
    retrieval = FakeRetrievalService(error=EmptyCollectionError("nothing ingested"))
    with caplog.at_level(logging.INFO, logger="clinverify.request"), _client(retrieval) as client:
        client.post("/query", json={"query": "What is the dose?"})

    records = [r for r in caplog.records if r.name == "clinverify.request"]
    fields = records[-1].extra_fields
    assert fields["status_code"] == 400
    assert fields["error_code"] == "empty_collection"
