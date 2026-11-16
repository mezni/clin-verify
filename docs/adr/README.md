# Architecture Decision Records (ADRs)

This directory records the significant architectural decisions for ClinVerify, the context in which they were made, and their consequences. ADRs are immutable once `Accepted`; superseding decisions are captured in new ADRs that reference the old ones.

Format based on [Michael Nygard's ADR pattern](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions).

## Index

| ID | Title | Status | Date |
| :-- | :-- | :-- | :-- |
| [0001](0001-hybrid-retrieval-rrf.md) | Hybrid retrieval with Reciprocal Rank Fusion | Accepted | 2026-11-15 |
| [0002](0002-cross-encoder-reranking.md) | Cross-encoder reranking stage | Accepted | 2026-11-15 |
| [0003](0003-local-onnx-embeddings.md) | Local ONNX embeddings via FastEmbed | Accepted | 2026-11-15 |
| [0004](0004-deterministic-generation-citation-audit.md) | Deterministic generation + citation audit | Accepted | 2026-11-15 |
| [0005](0005-qdrant-hybrid-store.md) | Qdrant as the hybrid vector store | Accepted | 2026-11-15 |
| [0006](0006-table-preserving-chunking.md) | Table-preserving parsing and header-aware chunking | Accepted | 2026-11-15 |
| [0007](0007-containerized-microservices.md) | Containerized microservices via Docker Compose | Accepted | 2026-11-15 |
| [0008](0008-ragas-ci-eval-gate.md) | Ragas evaluation gate in CI | Accepted | 2026-11-15 |

## Statuses

- **Proposed** — under discussion.
- **Accepted** — agreed and in effect.
- **Deprecated** — no longer recommended.
- **Superseded** — replaced by a later ADR.
