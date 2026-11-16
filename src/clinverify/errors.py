"""Domain-level exceptions mapped to HTTP error contracts in the API layer."""

from __future__ import annotations


class ClinVerifyError(Exception):
    """Base class for all ClinVerify domain errors."""


class InvalidDocumentError(ClinVerifyError):
    """Raised when an uploaded document is rejected or cannot be parsed.

    Maps to the ``400 invalid_document`` contract in ``docs/specifications.md``.
    """


class EmptyCollectionError(ClinVerifyError):
    """Raised when a query is issued before any document is ingested.

    Maps to the ``400 empty_collection`` contract in ``docs/specifications.md``.
    """


class UpstreamLLMError(ClinVerifyError):
    """Raised when the generation LLM fails or times out.

    Maps to the ``500 upstream_llm_error`` contract in ``docs/specifications.md``.
    """


class VectorStoreError(ClinVerifyError):
    """Raised when the vector store is unreachable or errors.

    Maps to the ``500 qdrant_unavailable`` contract in ``docs/specifications.md``.
    """


class RerankerUnavailableError(ClinVerifyError):
    """Raised when the cross-encoder reranker cannot be loaded.

    Maps to the ``503 reranker_unavailable`` contract in ``docs/specifications.md``.
    """
