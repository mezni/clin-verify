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
