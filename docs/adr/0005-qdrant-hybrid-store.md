# ADR-0005: Qdrant as the hybrid vector store

- **Status:** Accepted
- **Date:** 2026-11-15
- **Deciders:** Architecture team
- **Related:** [sdd.md §4.4](../sdd.md), [specifications.md §2](../specifications.md), [architecture.md §2.1](../architecture.md)

## Context

The retrieval design requires a store that can hold **both dense and sparse vectors per point**, execute concurrent prefetch queries, support server-side IDF for BM25, and carry rich payload metadata (`doc_id`, `page_num`, `citation`, `text`) for citation mapping and filtering.

## Decision

Use **Qdrant** as the vector database, deployed as its own container, with a single collection `clinical_trials_hybrid`:
- Named vector `dense`: size 384, cosine.
- Named vector `bm25`: `SparseVectorParams` with `models.Modifier.IDF`.
- Payload metadata per point as specified in `specifications.md §2`.

## Consequences

### Positive
- Native multi-vector (dense + sparse) support in one collection simplifies the hybrid design.
- Server-side IDF modifier for BM25 without a separate search engine.
- Payload filtering enables future scoping by `doc_id` or date.
- Official Docker image fits the Compose topology.

### Negative / Trade-offs
- Adds a stateful service to the deployment; requires volume persistence and backup strategy.
- Server-side IDF means the collection must be populated before queries are meaningful (cold-start → `empty_collection` guard).

## Alternatives Considered

- **pgvector + Postgres FTS** — considered: viable but requires custom fusion and no native sparse+IDF in one query.
- **Elasticsearch/OpenSearch (BM25) + separate dense store** — rejected: two systems to operate and keep in sync.
- **FAISS in-process** — rejected: no sparse support, no persistence/payload model.
