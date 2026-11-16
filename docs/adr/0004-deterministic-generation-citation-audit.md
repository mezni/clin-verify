# ADR-0004: Deterministic generation + citation audit

- **Status:** Accepted
- **Date:** 2026-11-15
- **Deciders:** Architecture team
- **Related:** [sdd.md §4.3](../sdd.md), [architecture.md §2.3](../architecture.md), [product.md §4.3](../product.md)

## Context

In a regulated clinical workflow, an unverifiable or hallucinated dosage is worse than no answer. LLMs can fabricate claims and emit citation tags that do not correspond to any retrieved source. Two guarantees are required: (1) answers must be grounded solely in retrieved context, and (2) every emitted citation must be traceable to a real chunk.

## Decision

- Generate with `gpt-4o-mini` at **`temperature=0.0`** with a system prompt instructing the model to answer **solely** from provided context and to include inline citations in the form `[doc_id, p.X]`.
- When evidence is absent, return a deterministic **`INSUFFICIENT_CONTEXT`** response rather than interpolating.
- Post-process every response with a **regex citation auditor** that extracts `[doc_id, p.X]` tags and verifies each maps to a retrieved chunk's `citation`; unresolved tags set `citation_verified=false`.

## Consequences

### Positive
- Reproducible, auditable outputs suitable for regulated review.
- Programmatic verification is independent of the LLM's self-report.
- Deterministic abstention prevents hallucinated clinical claims.

### Negative / Trade-offs
- `temperature=0.0` reduces stylistic variation and may reduce recall on ambiguous phrasing.
- Abstention can frustrate users when evidence exists but retrieval missed it — surfaced via the UI/telemetry for tuning.
- Regex auditing only validates citation *form and mapping*, not the factual entailment of the sentence citing it (Ragas faithfulness covers that).

## Alternatives Considered

- **Higher temperature** — rejected: non-determinism unacceptable for clinical claims.
- **LLM-judge for citation verification** — rejected: adds cost/latency and reintroduces non-determinism.
- **No abstention (always answer)** — rejected: violates the core trust requirement.
