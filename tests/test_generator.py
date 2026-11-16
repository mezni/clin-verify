"""Unit tests for grounded generation and the citation auditor."""

from __future__ import annotations

from typing import Any

import pytest

from clinverify.config import Settings
from clinverify.errors import UpstreamLLMError
from clinverify.generator import (
    ABSTENTION_TOKEN,
    GenerationService,
    OpenAIChatClient,
    audit_citations,
    build_context,
    is_abstention,
)
from clinverify.retrieval import RetrievedChunk


def _source(doc_id: str = "FDA_Apixaban_2025", page_num: int = 4) -> RetrievedChunk:
    return RetrievedChunk(
        point_id=f"{doc_id}-{page_num}",
        doc_id=doc_id,
        page_num=page_num,
        citation=f"[{doc_id}, p.{page_num}]",
        text="The reduced dose is 2.5 mg twice daily.",
        score=0.9,
        rerank_score=4.821,
    )


class FakeLLM:
    def __init__(self, answer: str, *, error: Exception | None = None) -> None:
        self._answer = answer
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if self.error is not None:
            raise self.error
        return self._answer


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def test_audit_valid_citation_is_verified() -> None:
    audit = audit_citations("Dose is 2.5 mg [FDA_Apixaban_2025, p.4].", [_source()])

    assert audit.verified is True
    assert audit.citations == ["[FDA_Apixaban_2025, p.4]"]
    assert audit.unresolved_count == 0


def test_audit_hallucinated_doc_id_is_unresolved() -> None:
    audit = audit_citations("Dose is 2.5 mg [FDA_Notreal_2025, p.4].", [_source()])

    assert audit.verified is False
    assert audit.unresolved == ["[FDA_Notreal_2025, p.4]"]
    assert audit.unresolved_count == 1


def test_audit_hallucinated_page_is_unresolved() -> None:
    audit = audit_citations("Dose is 2.5 mg [FDA_Apixaban_2025, p.99].", [_source()])

    assert audit.verified is False
    assert audit.unresolved_count == 1


def test_audit_without_citations_is_unverified() -> None:
    audit = audit_citations("Dose is 2.5 mg.", [_source()])

    assert audit.verified is False
    assert audit.citations == []


def test_audit_ignores_malformed_tags() -> None:
    audit = audit_citations(
        "See [FDA_Apixaban_2025] and [p.4] and [FDA_Apixaban_2025, page 4].", [_source()]
    )

    assert audit.verified is False
    assert audit.citations == []


def test_audit_multiple_citations_counts_each() -> None:
    sources = [_source(), _source(page_num=5)]
    audit = audit_citations(
        "A [FDA_Apixaban_2025, p.4] and B [FDA_Apixaban_2025, p.5] and C [FDA_X, p.1].",
        sources,
    )

    assert audit.verified is False
    assert audit.unresolved_count == 1
    assert audit.citations == ["[FDA_Apixaban_2025, p.4]", "[FDA_Apixaban_2025, p.5]"]


def test_is_abstention_detection() -> None:
    assert is_abstention("INSUFFICIENT_CONTEXT") is True
    assert is_abstention("insufficient_context") is True
    assert is_abstention("The answer is 2.5 mg [FDA_Apixaban_2025, p.4].") is False


def test_build_context_includes_citations_and_text() -> None:
    context = build_context([_source(), _source(page_num=5)])

    assert "[FDA_Apixaban_2025, p.4]" in context
    assert "[FDA_Apixaban_2025, p.5]" in context
    assert "2.5 mg twice daily" in context


def test_generate_grounded_answer_is_verified() -> None:
    llm = FakeLLM("The reduced dose is 2.5 mg [FDA_Apixaban_2025, p.4].")
    service = GenerationService(_settings(), llm)

    result = service.generate("What is the dose?", [_source()])

    assert result.citation_verified is True
    assert result.abstained is False
    assert result.unresolved_count == 0
    assert result.generation_ms >= 0
    assert llm.calls
    assert "FDA_Apixaban_2025" in llm.calls[0][1]


def test_generate_hallucinated_citation_is_unverified() -> None:
    llm = FakeLLM("The dose is 2.5 mg [FDA_Notreal_2025, p.4].")
    service = GenerationService(_settings(), llm)

    result = service.generate("What is the dose?", [_source()])

    assert result.citation_verified is False
    assert result.unresolved_count == 1


def test_generate_abstention() -> None:
    llm = FakeLLM(ABSTENTION_TOKEN)
    service = GenerationService(_settings(), llm)

    result = service.generate("What is the dose?", [_source()])

    assert result.abstained is True
    assert result.citation_verified is False
    assert result.answer == ABSTENTION_TOKEN


def test_generate_without_sources_abstains_without_llm_call() -> None:
    llm = FakeLLM("should not be called")
    service = GenerationService(_settings(), llm)

    result = service.generate("What is the dose?", [])

    assert result.abstained is True
    assert result.answer == ABSTENTION_TOKEN
    assert llm.calls == []


def test_generate_propagates_upstream_error() -> None:
    llm = FakeLLM("", error=UpstreamLLMError("timeout"))
    service = GenerationService(_settings(), llm)

    with pytest.raises(UpstreamLLMError):
        service.generate("What is the dose?", [_source()])


@pytest.mark.parametrize("api_key", [None, ""])
def test_openai_client_fails_fast_without_api_key(api_key: str | None) -> None:
    client = OpenAIChatClient("gpt-4o-mini", 0.0, api_key)

    with pytest.raises(UpstreamLLMError, match="OPENAI_API_KEY is not configured"):
        client.complete("system", "user")
