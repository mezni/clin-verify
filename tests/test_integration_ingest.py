"""Integration test exercising the real Qdrant store and FastEmbed models.

Disabled by default; enable with ``CLINVERIFY_INTEGRATION=1`` and a reachable
Qdrant instance (``QDRANT_URL``, default ``http://localhost:6333``).
"""

from __future__ import annotations

import os
from collections.abc import Callable

import pytest
from qdrant_client import QdrantClient

from clinverify.config import Settings
from clinverify.ingestion import FastEmbedEmbedder, IngestionService

pytestmark = pytest.mark.skipif(
    os.getenv("CLINVERIFY_INTEGRATION") != "1",
    reason="set CLINVERIFY_INTEGRATION=1 to run integration tests",
)

COLLECTION = "clinverify_integration"


def test_ingest_roundtrip_is_idempotent(
    pdf_factory: Callable[[list[list[str]]], bytes],
) -> None:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_collection=COLLECTION,
    )
    client = QdrantClient(url=settings.qdrant_url)
    embedder = FastEmbedEmbedder(settings.dense_model, settings.sparse_model)
    service = IngestionService(settings, client, embedder)
    data = pdf_factory(
        [
            [
                "## 2.2 Dosing",
                "| Drug | Dose |",
                "|------|------|",
                "| Apixaban | 2.5 mg BID |",
            ]
        ]
    )

    try:
        first = service.ingest("IT_Apixaban_2025", data)
        assert first.pages_processed >= 1
        assert first.chunks_indexed >= 1

        count_after_first = client.count(COLLECTION, exact=True).count
        assert count_after_first == first.chunks_indexed

        second = service.ingest("IT_Apixaban_2025", data)
        count_after_second = client.count(COLLECTION, exact=True).count

        assert second.chunks_indexed == first.chunks_indexed
        assert count_after_second == count_after_first

        points, _ = client.scroll(COLLECTION, limit=1, with_payload=True)
        assert points[0].payload is not None
        assert points[0].payload["doc_id"] == "IT_Apixaban_2025"
    finally:
        client.delete_collection(COLLECTION)
        client.close()
