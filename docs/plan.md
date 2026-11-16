# Implementation Plan & Roadmap

> **Status:** Draft for review
> **Scope:** Build sequence, milestones, and exit criteria for ClinVerify
> **Related:** [sdd.md](sdd.md) · [architecture.md](architecture.md) · [specifications.md](specifications.md) · [testing.md](testing.md) · [adr/](adr/README.md)

## 1. Guiding Principles

- **Contract-first.** API contracts (`specifications.md`) and Pydantic models are frozen before implementation; code conforms to the contract, not vice versa.
- **Vertical slices.** Each phase ends with a runnable, testable increment — never a half-wired pipeline.
- **Gate early.** The Ragas evaluation gate and citation auditor are wired in as soon as retrieval and generation exist, not retrofitted.
- **Observability from Phase 1.** Telemetry fields are emitted from the first endpoint, not added later.

## 2. Phase Overview

| Phase | Milestone | Primary Deliverable | Depends on |
| :-- | :-- | :-- | :-- |
| 0 | Repository & tooling | Dev environment, lint/type/test commands, CI skeleton | — |
| 1 | Ingestion slice | `POST /ingest` end-to-end into Qdrant | Phase 0 |
| 2 | Retrieval slice | `POST /query` returning ranked sources (no generation) | Phase 1 |
| 3 | Generation & audit | Grounded answer + citation audit | Phase 2 |
| 4 | UI & telemetry | Streamlit inspector with mode toggle and telemetry | Phase 3 |
| 5 | Evaluation gate | Ragas CI gate + golden dataset | Phase 3 |
| 6 | Hardening & release | Failure-mode coverage, docs, `docker compose up` | Phases 1–5 |

## 3. Phase Detail

### Phase 0 — Repository & Tooling
**Tasks**
- Initialize Python project (dependency management via `uv`), `pyproject.toml`, pinned lockfile.
- Establish layout: `src/` (ingestion, retrieval, generator, api), `tests/`, `docs/`.
- Configure `ruff` (lint), `mypy` (types), `pytest`.
- Add `.env.example` (`OPENAI_API_KEY`, Qdrant URL, model names) and `.gitignore`.
- Scaffold GitHub Actions workflow (lint → type → test).

**Exit criteria**
- `ruff`, `mypy`, `pytest` all green on an empty suite.
- CI runs on a trivial PR.

### Phase 1 — Ingestion Slice (`POST /ingest`)
**Tasks**
- Implement `pymupdf4llm` parsing → Markdown with preserved tables/headers.
- Implement header-aware `RecursiveCharacterTextSplitter` (600/80) per [ADR-0006](adr/0006-table-preserving-chunking.md).
- Wire FastEmbed dense (`bge-small-en-v1.5`) + sparse (`Qdrant/bm25`) per [ADR-0003](adr/0003-local-onnx-embeddings.md).
- Create Qdrant collection `clinical_trials_hybrid` (dense 384/cosine + sparse IDF) per [ADR-0005](adr/0005-qdrant-hybrid-store.md).
- Deterministic point IDs (`uuid5(doc_id, chunk_index)`) for idempotent re-ingest.
- Implement `POST /ingest` contract + validation (`doc_id` pattern, MIME, size cap).

**Exit criteria**
- Uploading a sample FDA label returns `{status, doc_id, pages_processed, chunks_indexed}`.
- Re-ingesting the same `doc_id` does not duplicate points.
- Invalid uploads return `400 invalid_document`; bad `doc_id` returns `422`.

### Phase 2 — Retrieval Slice (`POST /query`, no generation)
**Tasks**
- Parallel dense + sparse prefetch (`k=20`) in Qdrant.
- RRF fusion (`k=60`) per [ADR-0001](adr/0001-hybrid-retrieval-rrf.md).
- Cross-encoder reranking (`bge-reranker-base`, top `k=5`) per [ADR-0002](adr/0002-cross-encoder-reranking.md).
- Implement `dense_only` mode for baseline comparison.
- Return `sources` with `text`, `citation`, `score`, `rerank_score`; emit `retrieval_ms`/`rerank_ms`.
- `empty_collection` guard (`400`).

**Exit criteria**
- Golden-dataset questions return the `expected_doc_id`/`expected_page` chunk within top-`k`.
- Hybrid mode outperforms dense-only on the numeric/negation subset.
- `top_k` boundary validation returns `422` outside `1..20`.

### Phase 3 — Generation & Citation Audit
**Tasks**
- System prompt enforcing context-only answers + inline `[doc_id, p.X]` citations.
- `gpt-4o-mini` at `temperature=0.0` per [ADR-0004](adr/0004-deterministic-generation-citation-audit.md).
- `INSUFFICIENT_CONTEXT` deterministic abstention path.
- Regex citation auditor; set `citation_verified`; count unresolved tags.
- Complete `telemetry` (`generation_ms`, `total_ms`).
- Error contracts: `500 upstream_llm_error`, `500 qdrant_unavailable`, `503 reranker_unavailable`.

**Exit criteria**
- A golden-dataset answer cites only retrieved chunks; `citation_verified=true`.
- Injected hallucinated citation sets `citation_verified=false`.
- Out-of-scope query returns `INSUFFICIENT_CONTEXT`.

### Phase 4 — UI & Telemetry
**Tasks**
- Streamlit inspector: query box, answer pane, expandable source drawers (raw Markdown, RRF rank, rerank score).
- Pipeline toggle: `hybrid_rerank` ↔ `dense_only`.
- Stage-by-stage telemetry display.
- Structured JSON request logging with `request_id` correlation.

**Exit criteria**
- Toggling modes visibly changes retrieval results and telemetry.
- `request_id` appears in both success and error responses/logs.

### Phase 5 — Evaluation Gate
**Tasks**
- Author `golden_dataset.json` with coverage matrix (dosage, contraindication, negation, eligibility, abstention).
- Implement `tests/test_ragas_eval.py` (`Faithfulness ≥ 0.95`, `LLMContextPrecisionWithReference ≥ 0.88`) per [ADR-0008](adr/0008-ragas-ci-eval-gate.md).
- Wire gate into CI as a **required** check.

**Exit criteria**
- Gate passes on the golden dataset and blocks merge on simulated regression.

### Phase 6 — Hardening & Release
**Tasks**
- Execute failure-mode mitigations from [sdd.md §5](sdd.md): OOM/batch ingest, divergence regression, hallucination flagging, fragmentation, upstream outages, cold collection.
- Finalize `docker compose up` topology with healthchecks and volumes per [ADR-0007](adr/0007-containerized-microservices.md).
- Complete remaining docs (`security.md`, `observability.md`, `deployment.md`, `CONTRIBUTING.md`).
- Verify SLAs: P95 latency `< 800 ms`, faithfulness `≥ 0.95`, context precision `≥ 0.88`.

**Exit criteria**
- Clean `docker compose up` → ingest → query → inspected answer with verified citations.
- All CI checks green; SLAs met on the golden dataset.

## 4. Critical Path & Risks

- **Critical path:** Phase 0 → 1 → 2 → 3 → 5. Phases 4 and 6 can overlap with 5.
- **Highest-risk item:** retrieval precision on numeric/negation cases (Phases 2–3) — mitigated by hybrid + rerank + the Phase 5 gate.
- **Latency risk:** reranker + generation against the `< 800 ms` P95 target — measure from Phase 2 and tune `k`/batch size if needed.
- **CI cost risk:** Ragas LLM calls — mitigate with response caching or a gating subset if runtime becomes prohibitive.

## 5. Definition of Done (per phase)

1. Code conforms to the frozen contracts in `specifications.md`.
2. Unit + integration tests for the phase pass; lint and type checks clean.
3. Telemetry fields for the phase are emitted and documented.
4. Relevant failure modes have a test or an explicit, documented deferral.
5. Docs updated where behavior diverged from spec.
