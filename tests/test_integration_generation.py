"""Integration test for grounded generation + citation audit (live OpenAI).

Disabled by default; enable with ``CLINVERIFY_INTEGRATION=1``, a reachable
Qdrant instance, and ``OPENAI_API_KEY``. Loads FastEmbed models, the
cross-encoder reranker, and calls the OpenAI API.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import pytest
from qdrant_client import QdrantClient

from clinverify.config import Settings
from clinverify.generator import ABSTENTION_TOKEN, GenerationService, OpenAIChatClient
from clinverify.ingestion import FastEmbedEmbedder, IngestionService
from clinverify.retrieval import CrossEncoderReranker, RetrievalService

pytestmark = pytest.mark.skipif(
    os.getenv("CLINVERIFY_INTEGRATION") != "1" or not os.getenv("OPENAI_API_KEY"),
    reason="set CLINVERIFY_INTEGRATION=1 and OPENAI_API_KEY to run integration tests",
)

COLLECTION = "clinverify_generation_it"


def test_generation_cites_sources_and_abstains(
    pdf_factory: Callable[[list[list[str]]], bytes],
) -> None:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_collection=COLLECTION,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    client = QdrantClient(url=settings.qdrant_url)
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    embedder = FastEmbedEmbedder(settings.dense_model, settings.sparse_model)
    ingestion = IngestionService(settings, client, embedder)
    retrieval = RetrievalService(
        settings, client, embedder, CrossEncoderReranker(settings.reranker_model)
    )
    generation = GenerationService(
        settings,
        OpenAIChatClient(
            settings.generator_model,
            settings.generator_temperature,
            settings.openai_api_key,
        ),
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
        ingestion.ingest("FDA_Apixaban_2025", data)

        sources = retrieval.retrieve(
            "What is the reduced dose of apixaban for severe renal impairment?",
            "hybrid_rerank",
            top_k=5,
        ).sources
        answer = generation.generate(
            "What is the reduced dose of apixaban for severe renal impairment?", sources
        )

        assert answer.abstained is False
        assert "2.5" in answer.answer
        assert answer.citation_verified is True

        out_of_scope = generation.generate(
            "What is the recommended dose of metformin for type 2 diabetes?", sources
        )
        assert out_of_scope.answer.strip() == ABSTENTION_TOKEN
        assert out_of_scope.abstained is True
    finally:
        client.delete_collection(COLLECTION)
        client.close()
