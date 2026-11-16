"""Shared pytest fixtures and configuration for the ClinVerify test suite."""

from __future__ import annotations

from collections.abc import Callable

import pymupdf
import pytest


def _build_pdf(pages: list[list[str]]) -> bytes:
    """Build an in-memory PDF from a list of pages of text lines."""

    document = pymupdf.open()  # type: ignore[no-untyped-call]
    try:
        for lines in pages:
            page = document.new_page()
            y = 72.0
            for line in lines:
                page.insert_text((72, y), line, fontsize=11)
                y += 16.0
        return document.tobytes()  # type: ignore[no-any-return, no-untyped-call]
    finally:
        document.close()  # type: ignore[no-untyped-call]


@pytest.fixture
def pdf_factory() -> Callable[[list[list[str]]], bytes]:
    """Return a factory that builds PDF bytes from page text lines."""

    return _build_pdf


@pytest.fixture
def sample_pdf(pdf_factory: Callable[[list[list[str]]], bytes]) -> bytes:
    """A small two-page PDF resembling a dosing label."""

    return pdf_factory(
        [
            [
                "## 2.2 Dosing",
                "| Drug | Dose |",
                "|------|------|",
                "| Apixaban | 2.5 mg BID |",
            ],
            [
                "## 5.1 Contraindications",
                "Active pathological bleeding.",
            ],
        ]
    )
