# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Phase 3 generation & citation audit: grounded `gpt-4o-mini` generation
  (`temperature=0.0`) over reranked context, deterministic
  `INSUFFICIENT_CONTEXT` abstention, and a regex citation auditor that sets
  `citation_verified` and counts unresolved `[doc_id, p.X]` tags.
- Complete `/query` telemetry (`generation_ms`, `total_ms`) and error contracts
  (`500 upstream_llm_error`, `500 qdrant_unavailable`, `503 reranker_unavailable`).
- Phase 2 retrieval slice: `POST /query` with parallel dense + sparse prefetch
  (`k=20`), Reciprocal Rank Fusion (`k=60`), and cross-encoder reranking
  (`bge-reranker-base`, top `k=5`), returning `sources` with
  `text`/`citation`/`score`/`rerank_score` and `retrieval_ms`/`rerank_ms`.
- `dense_only` retrieval mode for baseline comparison.
- `400 empty_collection` guard when querying before any document is ingested.
- Phase 1 ingestion slice: `POST /ingest` with layout-aware PDF parsing
  (`pymupdf4llm`), header-aware chunking, FastEmbed dense + sparse embeddings,
  and idempotent Qdrant upserts (`clinical_trials_hybrid`).
- Ingestion error contracts (`400 invalid_document`, `422 validation_error`)
  with a uniform error envelope and `X-Request-ID` correlation.
- Unit tests (parsing, chunking, deterministic point ids, service, API contract)
  and an opt-in live integration test.
- Documentation set: product brief, architecture, technical specifications,
  specification-driven design (SDD), implementation plan, test strategy, and
  Architecture Decision Records (`docs/`).
- Project README with setup, usage, and evaluation instructions.
- Phase 0 scaffold: `src/clinverify` package, `pyproject.toml` with pinned `uv.lock`,
  `ruff`/`mypy`/`pytest` configuration, `.env.example`, and a GitHub Actions CI
  workflow (lint → typecheck → test).

### Changed
- Repository cleanup and restructuring.
- Corrected cross-document inconsistencies in citation format, latency example,
  dosage example, RRF parameters, and metric naming.

## [0.3.0] - 2026-11-07

### Added
- Excel backup/restore support.
- `uv`-based dependency workflow.
- FastAPI + SQLite backend rewritten with DDD layers.

## [0.2.0] - 2026-10-30

### Added
- Anthropic SDK integration.
- LLM architecture foundations (Day 1).

## [0.1.0] - 2026-10-24

### Added
- Ragas evaluation suite.
- Structured output support.
- Multi-agent LangChain workflows.
- MCP (Model Context Protocol) integration.
- Short-term memory and tool support.
- Initial agent implementation.
- Claude API single-turn call support.
