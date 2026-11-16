"""Integration test for hybrid retrieval against real Qdrant and models.

Disabled by default; enable with ``CLINVERIFY_INTEGRATION=1`` and a reachable
Qdrant instance. This test loads the FastEmbed models and the cross-encoder
reranker, so the first run downloads model weights.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import pytest
from qdrant_client import QdrantClient

from clinverify.config import Settings
from clinverify.errors import EmptyCollectionError
from clinverify.ingestion import FastEmbedEmbedder, IngestionService
from clinverify.retrieval import CrossEncoderReranker, RetrievalService

pytestmark = pytest.mark.skipif(
    os.getenv("CLINVERIFY_INTEGRATION") != "1",
    reason="set CLINVERIFY_INTEGRATION=1 to run integration tests",
)

COLLECTION = "clinverify_retrieval_it"


def test_hybrid_retrieval_finds_target_page(
    pdf_factory: Callable[[list[list[str]]], bytes],
) -> None:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_collection=COLLECTION,
    )
    client = QdrantClient(url=settings.qdrant_url)
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    embedder = FastEmbedEmbedder(settings.dense_model, settings.sparse_model)
    ingestion = IngestionService(settings, client, embedder)
    retrieval = RetrievalService(
        settings, client, embedder, CrossEncoderReranker(settings.reranker_model)
    )
    data = pdf_factory(
        [
            [
                "## 1 Indications",
                "Apixaban is indicated to reduce the risk of stroke in patients "
                "with nonvalvular atrial fibrillation.",
                "The usual dose is 5 mg twice daily.",
            ],
            [
                "## 2.2 Dose Adjustments",
                "For patients with severe renal impairment, the reduced dose of "
                "apixaban is 2.5 mg twice daily.",
            ],
        ]
    )

    try:
        with pytest.raises(EmptyCollectionError):
            retrieval.retrieve("dose", "hybrid_rerank", top_k=5)

        ingestion.ingest("FDA_Apixaban_2025", data)

        hybrid = retrieval.retrieve(
            "What is the reduced dose of apixaban for severe renal impairment?",
            "hybrid_rerank",
            top_k=5,
        )
        assert hybrid.sources
        assert hybrid.sources[0].doc_id == "FDA_Apixaban_2025"
        assert hybrid.sources[0].page_num == 2
        assert hybrid.sources[0].rerank_score is not None

        dense = retrieval.retrieve(
            "What is the reduced dose of apixaban for severe renal impairment?",
            "dense_only",
            top_k=5,
        )
        assert dense.sources
    finally:
        client.delete_collection(COLLECTION)
        client.close()
