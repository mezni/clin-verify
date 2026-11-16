"""Unit tests for RRF fusion, retrieval modes, and reranking."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from clinverify.config import Settings
from clinverify.errors import EmptyCollectionError, RerankerUnavailableError, VectorStoreError
from clinverify.retrieval import CrossEncoderReranker, RetrievalService, reciprocal_rank_fusion


class FakePoint:
    def __init__(self, point_id: str, score: float, text: str) -> None:
        self.id = point_id
        self.score = score
        self.payload = {
            "doc_id": "FDA_Apixaban_2025",
            "page_num": 4,
            "citation": "[FDA_Apixaban_2025, p.4]",
            "text": text,
        }


class _QueryResponse:
    def __init__(self, points: list[FakePoint]) -> None:
        self.points = points


class _CountResult:
    def __init__(self, count: int) -> None:
        self.count = count


class FakeStore:
    def __init__(self, dense: list[FakePoint], sparse: list[FakePoint], count: int) -> None:
        self._dense = dense
        self._sparse = sparse
        self._count = count

    def collection_exists(self, collection_name: str) -> bool:
        return self._count > 0

    def count(self, collection_name: str, *, exact: bool = True) -> _CountResult:
        return _CountResult(self._count)

    def query_points(
        self,
        *,
        collection_name: str,
        query: Any,
        using: str,
        limit: int,
        with_payload: bool = True,
    ) -> _QueryResponse:
        points = self._dense if using == "dense" else self._sparse
        return _QueryResponse(points[:limit])


class FakeEmbedder:
    def embed(self, texts: Sequence[str]) -> tuple[list[list[float]], list[Any]]:
        dense = [[0.1, 0.2, 0.3] for _ in texts]
        sparse = [object() for _ in texts]
        return dense, sparse


class FakeReranker:
    def __init__(self, scores: list[float] | None = None) -> None:
        self._scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        docs = list(documents)
        self.calls.append((query, docs))
        if self._scores is not None:
            return self._scores[: len(docs)]
        return [float(len(docs) - index) for index in range(len(docs))]


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def test_reciprocal_rank_fusion_scores() -> None:
    scores = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)

    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["c"] == pytest.approx(1 / 62)


def test_hybrid_mode_fuses_and_reranks() -> None:
    dense = [FakePoint("a", 0.9, "alpha"), FakePoint("b", 0.8, "beta")]
    sparse = [FakePoint("b", 5.0, "beta"), FakePoint("c", 4.0, "gamma")]
    store = FakeStore(dense, sparse, count=3)
    reranker = FakeReranker(scores=[0.1, 0.9, 0.5])
    service = RetrievalService(_settings(), store, FakeEmbedder(), reranker)

    result = service.retrieve("query", "hybrid_rerank", top_k=2)

    assert [source.point_id for source in result.sources] == ["a", "c"]
    assert result.sources[0].rerank_score == pytest.approx(0.9)
    assert reranker.calls
    assert reranker.calls[0][1] == ["beta", "alpha", "gamma"]
    assert result.retrieval_ms >= 0
    assert result.rerank_ms >= 0


def test_dense_only_mode_skips_rerank() -> None:
    dense = [FakePoint("a", 0.9, "alpha"), FakePoint("b", 0.8, "beta")]
    store = FakeStore(dense, sparse=[], count=2)
    reranker = FakeReranker()
    service = RetrievalService(_settings(), store, FakeEmbedder(), reranker)

    result = service.retrieve("query", "dense_only", top_k=1)

    assert [source.point_id for source in result.sources] == ["a"]
    assert result.sources[0].rerank_score is None
    assert result.sources[0].score == pytest.approx(0.9)
    assert reranker.calls == []


def test_empty_collection_raises() -> None:
    store = FakeStore(dense=[], sparse=[], count=0)
    service = RetrievalService(_settings(), store, FakeEmbedder(), FakeReranker())

    with pytest.raises(EmptyCollectionError):
        service.retrieve("query", "hybrid_rerank", top_k=5)


class _BrokenStore(FakeStore):
    def collection_exists(self, collection_name: str) -> bool:
        raise RuntimeError("connection refused")


def test_unreachable_store_raises_vector_store_error() -> None:
    store = _BrokenStore(dense=[], sparse=[], count=0)
    service = RetrievalService(_settings(), store, FakeEmbedder(), FakeReranker())

    with pytest.raises(VectorStoreError):
        service.retrieve("query", "hybrid_rerank", top_k=5)


def test_reranker_failure_propagates() -> None:
    dense = [FakePoint("a", 0.9, "alpha")]
    store = FakeStore(dense, sparse=[], count=1)

    class BrokenReranker:
        def score(self, query: str, documents: Sequence[str]) -> list[float]:
            raise RerankerUnavailableError("model missing")

    service = RetrievalService(_settings(), store, FakeEmbedder(), BrokenReranker())

    with pytest.raises(RerankerUnavailableError):
        service.retrieve("query", "hybrid_rerank", top_k=1)


def test_cross_encoder_load_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import sentence_transformers

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("cannot download model")

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", _boom)
    reranker = CrossEncoderReranker("BAAI/bge-reranker-base")

    with pytest.raises(RerankerUnavailableError):
        reranker.score("query", ["document"])
