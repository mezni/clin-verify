"""Unit tests for the ingestion service using fake dependencies."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from qdrant_client import models

from clinverify.config import Settings
from clinverify.ingestion import IngestionService


class FakeEmbedder:
    """Deterministic embedder that avoids downloading ONNX models."""

    def embed(self, texts: Sequence[str]) -> tuple[list[list[float]], list[models.SparseVector]]:
        dense = [[float(len(text) % 7) for _ in range(4)] for text in texts]
        sparse = [
            models.SparseVector(indices=[1, 2], values=[1.0, float(len(text))]) for text in texts
        ]
        return dense, sparse


class FakeQdrantClient:
    """In-memory stand-in for the Qdrant client."""

    def __init__(self) -> None:
        self.collections: set[str] = set()
        self.created: list[str] = []
        self.deleted: list[Any] = []
        self.upserted: list[list[models.PointStruct]] = []

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(
        self,
        *,
        collection_name: str,
        vectors_config: Any,
        sparse_vectors_config: Any,
    ) -> bool:
        self.collections.add(collection_name)
        self.created.append(collection_name)
        return True

    def delete(self, *, collection_name: str, points_selector: Any, wait: bool = True) -> Any:
        self.deleted.append(points_selector)
        return None

    def upsert(self, *, collection_name: str, points: Any, wait: bool = True) -> Any:
        self.upserted.append(list(points))
        return None


def _service(
    **settings_overrides: Any,
) -> tuple[IngestionService, FakeQdrantClient]:
    settings = Settings(_env_file=None, **settings_overrides)  # type: ignore[call-arg]
    client = FakeQdrantClient()
    return IngestionService(settings, client, FakeEmbedder()), client


def test_ingest_returns_counts(
    pdf_factory: Callable[[list[list[str]]], bytes],
) -> None:
    service, client = _service()
    data = pdf_factory([["## Dosing", "| Apixaban | 2.5 mg BID |"]])

    result = service.ingest("FDA_Apixaban_2025", data)

    assert result.doc_id == "FDA_Apixaban_2025"
    assert result.pages_processed == 1
    assert result.chunks_indexed >= 1
    assert client.upserted
    assert len(client.upserted[-1]) == result.chunks_indexed


def test_ingest_deletes_existing_points_for_doc(
    pdf_factory: Callable[[list[list[str]]], bytes],
) -> None:
    service, client = _service()
    data = pdf_factory([["## Dosing", "Apixaban 2.5 mg"]])

    service.ingest("FDA_Apixaban_2025", data)

    assert client.deleted
    condition = client.deleted[-1].filter.must[0]
    assert condition.key == "doc_id"
    assert condition.match.value == "FDA_Apixaban_2025"


def test_reingest_uses_identical_point_ids(
    pdf_factory: Callable[[list[list[str]]], bytes],
) -> None:
    service, client = _service()
    data = pdf_factory([["## Dosing", "| Apixaban | 2.5 mg BID |"]])

    service.ingest("FDA_Apixaban_2025", data)
    first_ids = {point.id for point in client.upserted[-1]}
    service.ingest("FDA_Apixaban_2025", data)
    second_ids = {point.id for point in client.upserted[-1]}

    assert first_ids == second_ids


def test_collection_created_with_hybrid_vectors(
    pdf_factory: Callable[[list[list[str]]], bytes],
) -> None:
    service, client = _service(qdrant_collection="clinical_trials_hybrid")
    data = pdf_factory([["## Dosing", "Apixaban 2.5 mg"]])

    service.ingest("doc", data)

    assert client.created == ["clinical_trials_hybrid"]
