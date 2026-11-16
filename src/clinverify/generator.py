"""Grounded generation and programmatic citation auditing.

Builds a context-only prompt from the reranked sources, calls the generation
LLM at ``temperature=0.0``, and verifies every emitted ``[doc_id, p.X]`` tag
against the retrieved chunks. When the context lacks the answer the model
returns a deterministic ``INSUFFICIENT_CONTEXT`` abstention.

See ``docs/plan.md`` (Phase 3) and ``docs/adr/0004-deterministic-generation-citation-audit.md``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

from clinverify.config import Settings
from clinverify.errors import UpstreamLLMError
from clinverify.retrieval import RetrievedChunk

ABSTENTION_TOKEN = "INSUFFICIENT_CONTEXT"

CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9_\-]+),\s*p\.(\d+)\]")
ABSTENTION_PATTERN = re.compile(r"\bINSUFFICIENT_CONTEXT\b", re.IGNORECASE)

SYSTEM_PROMPT = (
    "You are ClinVerify, a clinical evidence assistant. Answer the user's "
    "question using ONLY the retrieved context provided in the user message. "
    "Do not rely on outside knowledge and do not infer or interpolate any value "
    "that is not explicitly stated in the context.\n\n"
    "Rules:\n"
    "1. Ground every factual claim in the retrieved context.\n"
    "2. After each claim, add an inline citation in the exact form [doc_id, p.X] "
    "using the citation shown above the supporting context block. Cite only the "
    "blocks that actually support the claim.\n"
    "3. Never invent doc_ids, page numbers, or citations.\n"
    "4. If the context does not contain the answer, reply with exactly "
    f"{ABSTENTION_TOKEN} and nothing else."
)


@dataclass(frozen=True)
class CitationAudit:
    """Outcome of verifying citation tags against retrieved chunks."""

    verified: bool
    citations: list[str]
    unresolved: list[str]
    unresolved_count: int


@dataclass(frozen=True)
class GenerationResult:
    """The grounded answer plus its audit outcome and timing."""

    answer: str
    citation_verified: bool
    abstained: bool
    unresolved_count: int
    generation_ms: float


class LLMClient(Protocol):
    """Minimal chat-completion interface used by the generator."""

    def complete(self, system: str, user: str) -> str: ...


class GenerationServiceProtocol(Protocol):
    """The subset of the generation service consumed by the API layer."""

    def generate(self, query: str, sources: Sequence[RetrievedChunk]) -> GenerationResult: ...


def is_abstention(answer: str) -> bool:
    """Return ``True`` when the answer is the deterministic abstention token."""

    return bool(ABSTENTION_PATTERN.search(answer))


def audit_citations(answer: str, sources: Sequence[RetrievedChunk]) -> CitationAudit:
    """Verify every ``[doc_id, p.X]`` tag resolves to a retrieved chunk.

    A response is verified only when it contains at least one citation and every
    citation maps to a retrieved chunk. Unresolved tags (hallucinated doc_ids or
    page numbers) make the response unverified and are reported.
    """

    allowed = {f"[{source.doc_id}, p.{source.page_num}]" for source in sources}
    tags = [
        f"[{match.group(1)}, p.{match.group(2)}]" for match in CITATION_PATTERN.finditer(answer)
    ]
    resolved = [tag for tag in tags if tag in allowed]
    unresolved = [tag for tag in tags if tag not in allowed]
    verified = bool(tags) and not unresolved
    return CitationAudit(
        verified=verified,
        citations=resolved,
        unresolved=unresolved,
        unresolved_count=len(unresolved),
    )


def build_context(sources: Sequence[RetrievedChunk]) -> str:
    """Render retrieved chunks into a citation-annotated context block."""

    blocks = [f"{source.citation}\n{source.text}" for source in sources]
    return "\n\n---\n\n".join(blocks)


class OpenAIChatClient:
    """Lazily-initialized OpenAI chat-completion client."""

    def __init__(
        self,
        model: str,
        temperature: float,
        api_key: str | None,
        *,
        max_retries: int = 2,
        timeout: float = 30.0,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._api_key = api_key
        self._max_retries = max_retries
        self._timeout = timeout
        self._client: Any | None = None

    def _load(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._api_key,
                max_retries=self._max_retries,
                timeout=self._timeout,
            )
        return self._client

    def complete(self, system: str, user: str) -> str:
        try:
            client = self._load()
            response = client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as exc:  # noqa: BLE001 - normalize upstream failures
            raise UpstreamLLMError(f"Generation request failed: {exc}") from exc
        return response.choices[0].message.content or ""


class GenerationService:
    """Generates grounded answers and audits their citations."""

    def __init__(self, settings: Settings, llm: LLMClient) -> None:
        self._settings = settings
        self._llm = llm

    def generate(self, query: str, sources: Sequence[RetrievedChunk]) -> GenerationResult:
        start = perf_counter()

        if not sources:
            return GenerationResult(
                answer=ABSTENTION_TOKEN,
                citation_verified=False,
                abstained=True,
                unresolved_count=0,
                generation_ms=(perf_counter() - start) * 1000.0,
            )

        user = f"Retrieved context:\n\n{build_context(sources)}\n\nQuestion: {query}"
        answer = self._llm.complete(SYSTEM_PROMPT, user)
        generation_ms = (perf_counter() - start) * 1000.0

        abstained = is_abstention(answer)
        audit = audit_citations(answer, sources)
        return GenerationResult(
            answer=answer,
            citation_verified=audit.verified and not abstained,
            abstained=abstained,
            unresolved_count=audit.unresolved_count,
            generation_ms=generation_ms,
        )
