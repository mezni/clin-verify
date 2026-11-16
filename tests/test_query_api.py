"""Unit tests for the ``POST /query`` HTTP contract."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from fastapi.testclient import TestClient

from clinverify.api import create_app
from clinverify.config import Settings
from clinverify.errors import EmptyCollectionError
from clinverify.ingestion import IngestResult
from clinverify.retrieval import Mode, RetrievalResult, RetrievedChunk


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


@contextmanager
def _client(retrieval: FakeRetrievalService, **settings_overrides: Any) -> Iterator[TestClient]:
    settings = Settings(_env_file=None, **settings_overrides)  # type: ignore[call-arg]
    app = create_app(
        settings=settings,
        ingestion=FakeIngestionService(),
        retrieval=retrieval,
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def test_query_hybrid_returns_sources() -> None:
    service = FakeRetrievalService()
    with _client(service) as client:
        response = client.post("/query", json={"query": "What is the dose?", "top_k": 5})

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "What is the dose?"
    assert body["answer"] is None
    assert body["citation_verified"] is None
    assert body["sources"][0]["citation"] == "[FDA_Apixaban_2025, p.4]"
    assert body["sources"][0]["rerank_score"] == 4.821
    assert body["telemetry"]["retrieval_ms"] == 114.2
    assert body["telemetry"]["total_ms"] == 114.2 + 342.6
    assert service.calls[0][1] == "hybrid_rerank"


def test_query_dense_only_mode() -> None:
    service = FakeRetrievalService()
    with _client(service) as client:
        response = client.post(
            "/query",
            json={"query": "What is the dose?", "mode": "dense_only"},
        )

    assert response.status_code == 200
    assert response.json()["sources"][0]["rerank_score"] is None
    assert service.calls[0][1] == "dense_only"


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
