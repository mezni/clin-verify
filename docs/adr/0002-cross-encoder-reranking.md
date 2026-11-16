# ADR-0002: Cross-encoder reranking stage

- **Status:** Accepted
- **Date:** 2026-11-15
- **Deciders:** Architecture team
- **Related:** [sdd.md §4.2](../sdd.md), [architecture.md §2.2](../architecture.md), [product.md §4.2](../product.md)

## Context

After RRF fusion, the top candidates still contain noise and near-duplicates. Clinical correctness requires the exact target evidence (correct dosage, correct condition) to rank in the top-`k` context window. Bi-encoders (dense embeddings) score query and document independently, so they cannot model token-level interactions like negation.

## Decision

Apply a **cross-encoder reranker** (`BAAI/bge-reranker-base`) to the fused top candidates (`k=20`), scoring each query-document pair jointly. Forward the top `k=5` to the generator. Reranking runs locally via Sentence-Transformers.

## Consequences

### Positive
- Joint query-document scoring resolves clinical negation and promotes exact numeric matches.
- Materially improves context precision, the metric the Ragas gate tracks (`≥ 0.88`).
- Keeps the most expensive scoring to a small candidate set (20), bounding cost.

### Negative / Trade-offs
- Adds ~350 ms latency (see telemetry `rerank_ms`), the largest single stage; P95 total budget is `< 800 ms`.
- Requires loading and running a transformer model in-process (memory footprint).
- Model availability is a runtime dependency — mitigated by the `503 reranker_unavailable` error contract.

## Alternatives Considered

- **No reranking** — rejected: insufficient precision on numeric/negation cases.
- **LLM-as-reranker** — rejected: far higher latency and cost per query.
- **Larger cross-encoder (e.g., `bge-reranker-large`)** — deferred: latency budget; revisit if precision falls below SLA.
