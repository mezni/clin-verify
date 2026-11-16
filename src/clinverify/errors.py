"""Domain-level exceptions mapped to HTTP error contracts in the API layer."""

from __future__ import annotations


class ClinVerifyError(Exception):
    """Base class for all ClinVerify domain errors."""


class InvalidDocumentError(ClinVerifyError):
    """Raised when an uploaded document is rejected or cannot be parsed.

    Maps to the ``400 invalid_document`` contract in ``docs/specifications.md``.
    """
