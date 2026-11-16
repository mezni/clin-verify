# ADR-0003: Local ONNX embeddings via FastEmbed

- **Status:** Accepted
- **Date:** 2026-11-15
- **Deciders:** Architecture team
- **Related:** [sdd.md §4.1](../sdd.md), [architecture.md §2.1](../architecture.md), [specifications.md §3](../specifications.md)

## Context

Embedding generation for both dense (`BAAI/bge-small-en-v1.5`, 384-dim) and sparse (`Qdrant/bm25`) representations is on the critical path for ingestion and, for query embedding, on the critical path for retrieval. Hosted embedding APIs add per-call cost, network latency, and send document text off-premise — a concern for regulatory/PHI-adjacent content.

## Decision

Generate all embeddings **locally** using **FastEmbed** with ONNX Runtime:
- Dense: `BAAI/bge-small-en-v1.5` (384 dims, cosine).
- Sparse: `Qdrant/bm25` with server-side IDF modifier (`models.Modifier.IDF`).

## Consequences

### Positive
- No per-query embedding cost; predictable cost profile.
- Document text stays within the deployment boundary.
- Removes network variance from the retrieval latency path.
- Small model (384-dim) keeps memory and cold-start modest.

### Negative / Trade-offs
- Model weights must be baked into or cached by the container image (image size, cold-start).
- Model upgrades require a full re-index of the corpus.
- CPU inference competes with the reranker for resources; may need resource limits.

## Alternatives Considered

- **OpenAI `text-embedding-3-small`** — rejected: cost, egress of document text, network latency.
- **Larger local model (`bge-base`/`bge-large`)** — deferred: precision gain not justified vs. latency/memory at current corpus scale.
