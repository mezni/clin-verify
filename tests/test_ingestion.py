"""Unit tests for parsing, chunking, and deterministic point ids."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from clinverify.config import Settings
from clinverify.errors import InvalidDocumentError
from clinverify.ingestion import chunk_pages, parse_pdf, point_id


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def test_parse_pdf_reports_page_numbers(
    pdf_factory: Callable[[list[list[str]]], bytes],
) -> None:
    data = pdf_factory([["## Section 1", "| Drug | Dose |"], ["## Section 2", "second page"]])

    pages = parse_pdf(data)

    assert [page.page_num for page in pages] == [1, 2]
    assert "Section 1" in pages[0].text
    assert "second page" in pages[1].text


def test_parse_pdf_rejects_non_pdf_bytes() -> None:
    with pytest.raises(InvalidDocumentError):
        parse_pdf(b"this is definitely not a pdf")


def test_chunk_pages_sets_provenance_and_citation(
    pdf_factory: Callable[[list[list[str]]], bytes],
) -> None:
    data = pdf_factory(
        [
            [
                "## Dosing",
                "| Drug | Dose |",
                "|------|------|",
                "| Apixaban | 2.5 mg BID |",
            ]
        ]
    )
    pages = parse_pdf(data)

    chunks = chunk_pages("FDA_Apixaban_2025", pages, _settings())

    assert chunks
    for index, chunk in enumerate(chunks):
        assert chunk.chunk_index == index
        assert chunk.doc_id == "FDA_Apixaban_2025"
        assert chunk.page_num == 1
        assert chunk.citation == "[FDA_Apixaban_2025, p.1]"
    assert any("Apixaban" in chunk.text and "2.5 mg" in chunk.text for chunk in chunks)


def test_chunk_size_is_respected(
    pdf_factory: Callable[[list[list[str]]], bytes],
) -> None:
    sentence = "The quick brown fox jumps over the lazy dog. "
    lines = [(sentence * 2).strip() for _ in range(40)]
    data = pdf_factory([lines])
    pages = parse_pdf(data)

    chunks = chunk_pages("doc", pages, _settings(chunk_size=200, chunk_overlap=20))

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 200 for chunk in chunks)


def test_point_id_is_deterministic_and_unique() -> None:
    assert point_id("doc", 0) == point_id("doc", 0)
    assert point_id("doc", 0) != point_id("doc", 1)
    assert point_id("doc_a", 0) != point_id("doc_b", 0)
