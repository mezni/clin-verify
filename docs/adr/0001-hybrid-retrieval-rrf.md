# ADR-0001: Hybrid retrieval with Reciprocal Rank Fusion

- **Status:** Accepted
- **Date:** 2026-11-15
- **Deciders:** Architecture team
- **Related:** [sdd.md §4.2](../sdd.md), [architecture.md §2.2](../architecture.md), [specifications.md §2](../specifications.md)

## Context

Clinical queries hinge on exact numeric values and negation (`serum creatinine >= 1.5 mg/dL`, `CrCl < 30 mL/min`, `2.5 mg` vs `25 mg`). Dense-only embedding retrieval maps these distinct tokens into nearly identical vector neighborhoods, causing dangerous retrieval misses. Conversely, sparse-only (BM25) retrieval misses semantic synonymy (e.g., "renal impairment" ↔ "kidney dysfunction").

## Decision

Run **dense vector KNN** and **sparse BM25** prefetch queries in parallel against Qdrant (`k=20` each), then merge the ranked lists using **Reciprocal Rank Fusion (RRF)** with rank constant `k=60`:

```
score(d) = Σ_over_lists  1 / (k + rank(d))
```

## Consequences

### Positive
- Captures both exact lexical/numeric matches and semantic synonymy.
- RRF is score-scale-agnostic — no normalization tuning between cosine and BM25 scores.
- Improves recall of the prefetch stage before reranking.

### Negative / Trade-offs
- Two indexes per document (dense + sparse) increases storage and ingest cost.
- RRF requires both lists to be non-empty to be meaningful; sparse may return few results for very short queries.
- Adds a fusion step to the retrieval latency budget.

## Alternatives Considered

- **Dense-only retrieval** — rejected: numerical/negation blindness.
- **Sparse-only (BM25)** — rejected: no semantic recall.
- **Weighted score linear combination** — rejected: requires per-corpus score normalization and is brittle.
- **ColBERT / late interaction** — deferred: higher operational complexity than warranted for the current corpus size.
