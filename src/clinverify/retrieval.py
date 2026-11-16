"""Hybrid retrieval pipeline.

Runs parallel dense and sparse (BM25) prefetch queries, fuses the ranked lists
with Reciprocal Rank Fusion (RRF), and reranks the candidates with a
cross-encoder. A ``dense_only`` mode provides a baseline without fusion or
reranking.

See ``docs/plan.md`` (Phase 2), ``docs/adr/0001-hybrid-retrieval-rrf.md``, and
``docs/adr/0002-cross-encoder-reranking.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, Protocol

from clinverify.config import Settings
from clinverify.errors import EmptyCollectionError, RerankerUnavailableError, VectorStoreError
from clinverify.ingestion import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, Embedder

Mode = Literal["hybrid_rerank", "dense_only"]

HYBRID_MODE: Mode = "hybrid_rerank"
DENSE_MODE: Mode = "dense_only"


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved chunk with its provenance and relevance scores."""

    point_id: str
    doc_id: str
    page_num: int
    citation: str
    text: str
    score: float
    rerank_score: float | None


@dataclass(frozen=True)
class RetrievalResult:
    """Sources returned by retrieval along with stage timings."""

    sources: list[RetrievedChunk]
    retrieval_ms: float
    rerank_ms: float
    candidates_prefetch: int = 0


class Reranker(Protocol):
    """Scores query-document pairs jointly."""

    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


class RetrievalStore(Protocol):
    """The subset of the Qdrant client used by the retrieval service."""

    def collection_exists(self, collection_name: str) -> bool: ...

    def count(self, collection_name: str, *, exact: bool = True) -> Any: ...

    def query_points(
        self,
        *,
        collection_name: str,
        query: Any,
        using: str,
        limit: int,
        with_payload: bool = True,
    ) -> Any: ...


class RetrievalServiceProtocol(Protocol):
    """The subset of the retrieval service consumed by the API layer."""

    def retrieve(self, query: str, mode: Mode, top_k: int) -> RetrievalResult: ...


def reciprocal_rank_fusion(ranked_ids: Sequence[Sequence[str]], k: int) -> dict[str, float]:
    """Fuse ranked lists with RRF: ``score = sum(1 / (k + rank))``."""

    scores: dict[str, float] = {}
    for ranked in ranked_ids:
        for rank, point_id in enumerate(ranked, start=1):
            scores[point_id] = scores.get(point_id, 0.0) + 1.0 / (k + rank)
    return scores


class CrossEncoderReranker:
    """Lazily-loaded cross-encoder reranker (``BAAI/bge-reranker-base``)."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self._model_name)
            except Exception as exc:  # noqa: BLE001 - normalize load failures
                raise RerankerUnavailableError(
                    f"Unable to load reranker '{self._model_name}': {exc}"
                ) from exc
        return self._model

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        docs = list(documents)
        if not docs:
            return []
        model = self._load()
        pairs = [(query, doc) for doc in docs]
        scores = model.predict(pairs)
        return [float(score) for score in scores]


class RetrievalService:
    """Executes hybrid or dense-only retrieval against Qdrant."""

    def __init__(
        self,
        settings: Settings,
        client: RetrievalStore,
        embedder: Embedder,
        reranker: Reranker,
    ) -> None:
        self._settings = settings
        self._client = client
        self._embedder = embedder
        self._reranker = reranker

    def _search(self, query: Any, using: str, limit: int) -> list[Any]:
        try:
            response = self._client.query_points(
                collection_name=self._settings.qdrant_collection,
                query=query,
                using=using,
                limit=limit,
                with_payload=True,
            )
        except Exception as exc:  # noqa: BLE001 - normalize store failures
            raise VectorStoreError(f"Vector store query failed: {exc}") from exc
        return list(response.points)

    def _to_chunk(self, point: Any, score: float, rerank_score: float | None) -> RetrievedChunk:
        payload = point.payload or {}
        return RetrievedChunk(
            point_id=str(point.id),
            doc_id=str(payload.get("doc_id", "")),
            page_num=int(payload.get("page_num", 0)),
            citation=str(payload.get("citation", "")),
            text=str(payload.get("text", "")),
            score=float(score),
            rerank_score=rerank_score,
        )

    def retrieve(self, query: str, mode: Mode, top_k: int) -> RetrievalResult:
        collection = self._settings.qdrant_collection
        try:
            exists = self._client.collection_exists(collection)
            is_empty = not exists or self._client.count(collection, exact=True).count == 0
        except Exception as exc:  # noqa: BLE001 - normalize store failures
            raise VectorStoreError(f"Vector store unavailable: {exc}") from exc
        if is_empty:
            raise EmptyCollectionError(
                "No documents have been ingested yet; ingest a PDF before querying"
            )

        start = perf_counter()
        dense_vectors, sparse_vectors = self._embedder.embed([query])
        dense_points = self._search(dense_vectors[0], DENSE_VECTOR_NAME, self._settings.prefetch_k)

        if mode == HYBRID_MODE:
            sparse_points = self._search(
                sparse_vectors[0], SPARSE_VECTOR_NAME, self._settings.prefetch_k
            )
            by_id: dict[str, Any] = {str(point.id): point for point in dense_points}
            for point in sparse_points:
                by_id.setdefault(str(point.id), point)
            fusion = reciprocal_rank_fusion(
                [
                    [str(point.id) for point in dense_points],
                    [str(point.id) for point in sparse_points],
                ],
                self._settings.rrf_k,
            )
            ordered = sorted(fusion.items(), key=lambda item: (-item[1], item[0]))
            candidates: list[tuple[Any, float]] = [
                (by_id[point_id], score) for point_id, score in ordered
            ]
        else:
            candidates = [(point, float(point.score)) for point in dense_points]

        retrieval_ms = (perf_counter() - start) * 1000.0

        rerank_start = perf_counter()
        if mode == HYBRID_MODE:
            pool = candidates[: self._settings.prefetch_k]
            texts = [self._to_chunk(point, score, None).text for point, score in pool]
            rerank_scores = self._reranker.score(query, texts)
            ranked = sorted(
                zip(pool, rerank_scores, strict=True),
                key=lambda item: (-item[1], str(item[0][0].id)),
            )[:top_k]
            sources = [
                self._to_chunk(point, score, rerank_score)
                for (point, score), rerank_score in ranked
            ]
            candidates_prefetch = len(pool)
        else:
            sources = [self._to_chunk(point, score, None) for point, score in candidates[:top_k]]
            candidates_prefetch = 0
        rerank_ms = (perf_counter() - rerank_start) * 1000.0

        return RetrievalResult(
            sources=sources,
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
            candidates_prefetch=candidates_prefetch,
        )
