# Test Strategy

> **Status:** Draft for review
> **Scope:** Unit, integration, and evaluation testing for ClinVerify
> **Related:** [sdd.md §6](sdd.md) · [plan.md](plan.md) · [specifications.md](specifications.md) · [adr/0008](adr/0008-ragas-ci-eval-gate.md)

## 1. Objectives

1. **Contract conformance** — every endpoint matches `specifications.md` exactly (schemas, status codes, error envelopes).
2. **Correctness of mechanics** — chunking preserves tables, RRF math is correct, citation auditing is sound.
3. **Quality non-regression** — retrieval precision and generation faithfulness do not regress below SLA on PR.
4. **Failure resilience** — the mitigations in [sdd.md §5](sdd.md) are exercised, not assumed.

## 2. Test Pyramid & Tooling

| Layer | Scope | Tooling | Runs |
| :-- | :-- | :-- | :-- |
| **Unit** | Pure functions: chunking, RRF, regex audit, Pydantic validation | `pytest` | Every commit / CI |
| **Integration** | Endpoints against a live Qdrant container; mocked LLM | `pytest`, `testcontainers` (or Compose service), `respx`/`vcr` for OpenAI | Every PR / CI |
| **Evaluation** | Answer quality on golden dataset | `pytest` + `ragas` | Every PR / CI (required gate) |
| **Manual / UI** | Streamlit inspector, mode toggle, source drawers | Streamlit run-through | Pre-release |

**Determinism rules**
- All LLM calls are **mocked** in unit and integration tests (fixed responses) — no network in CI except the evaluation gate.
- Embeddings are deterministic; where a real model is needed, use a tiny fixture corpus with a stable expected ordering.
- Ragas is the **only** test tier permitted to call a real LLM, and its results use threshold tolerance, not exact match.

## 3. Unit Tests

### 3.1 Chunking (`src/ingestion.py`)
- Table rows stay bound to their header within a chunk (no condition/dose split).
- Chunk length respects `chunk_size=600` with `chunk_overlap=80`.
- Separator priority splits at `##`/`###` boundaries before falling back to `\n`.
- `page_num` and `citation` metadata are attached correctly.
- Edge cases: empty PDF, single-page PDF, table wider than `chunk_size`.

### 3.2 Fusion & Retrieval (`src/retrieval.py`)
- RRF score matches the hand-computed `Σ 1/(k+rank)` (`k=60`) on synthetic ranked lists.
- Ties and unequal-length lists are handled deterministically.
- `dense_only` mode bypasses sparse prefetch and reranking.
- `top_k` is clamped/validated to `1..20`.

### 3.3 Citation Auditor (`src/generator.py`)
- Valid `[doc_id, p.X]` tags resolving to retrieved chunks → `citation_verified=true`.
- Unresolved tag (hallucinated `doc_id` or `page_num`) → `citation_verified=false`; unresolved count correct.
- Multiple citations, malformed tags, and tags without a page number are handled per contract.
- `INSUFFICIENT_CONTEXT` detection and flagging.

### 3.4 API Validation
- `query` shorter than 3 chars → `422`.
- `top_k` of `0` and `21` → `422`.
- Invalid `mode` → `422`.
- Invalid `doc_id` pattern → `422`; bad MIME → `400 invalid_document`.
- Error envelope always contains `error`, `detail`, `request_id`.

## 4. Integration Tests

- **Round-trip:** `POST /ingest` a fixture PDF → `POST /query` → assert `expected_doc_id`/`expected_page` appears in `sources`.
- **Idempotent ingest:** ingesting the same `doc_id` twice yields no duplicate points.
- **Citation audit end-to-end:** force a hallucinated citation via the mocked LLM → `citation_verified=false`.
- **Abstention:** query with no matching context → `INSUFFICIENT_CONTEXT`.
- **Cold collection:** query before any ingest → `400 empty_collection`.
- **Upstream failures:** mocked OpenAI timeout → `500 upstream_llm_error`; Qdrant unreachable → `500 qdrant_unavailable`; reranker load failure → `503 reranker_unavailable`.
- **Telemetry completeness:** every response contains `retrieval_ms`, `rerank_ms`, `generation_ms`, `total_ms`, and `total_ms` ≈ sum of stages.

## 5. Evaluation Suite (`tests/test_ragas_eval.py`)

- **Framework:** PyTest + Ragas v0.4+.
- **Dataset:** `golden_dataset.json` — version-controlled, with a coverage matrix:

| Category | Example |
| :-- | :-- |
| Exact dosage | Reduced-dose criteria for Apixaban |
| Contraindication | Boxed warnings / CYP450 interactions |
| Negation | Exclusion criteria ("must NOT have prior therapy") |
| Eligibility thresholds | Renal/hepatic impairment cutoffs |
| Abstention | Query with no supporting evidence → `INSUFFICIENT_CONTEXT` |

- **Metrics & thresholds (merge-blocking):**
  - `Faithfulness ≥ 0.95`
  - `LLMContextPrecisionWithReference ≥ 0.88`
- **Retrieval assertion:** `expected_doc_id` + `expected_page` present within top-`k` sources.
- **Tolerance:** LLM-judge metrics are non-deterministic; use threshold margins and periodic re-baselining rather than exact equality.

## 6. Non-Functional Tests

| Dimension | Test | Target |
| :-- | :-- | :-- |
| Latency | Time a full `/query` on the golden set | P95 `< 800 ms` |
| Memory | Ingest a large PDF under a memory cap | No OOM; batch upsert |
| Hybrid vs. dense-only | Compare metrics on numeric/negation subset | Hybrid strictly better |
| Idempotency | Re-ingest `doc_id` | Zero duplicate points |

## 7. CI Pipeline & Gates

```
lint (ruff) → typecheck (mypy) → unit tests → integration tests → ragas eval gate
```

- **Blocking:** any lint/type error, failing unit/integration test, or Ragas threshold breach.
- **Non-blocking (report):** latency and memory trends.
- Runs on every pull request via GitHub Actions; the Ragas gate is a required status check per [ADR-0008](adr/0008-ragas-ci-eval-gate.md).

## 8. Coverage & Exit Criteria

- **Line coverage target:** `≥ 85%` on `src/` (excluding model-loading boilerplate).
- **Contract coverage:** 100% of documented endpoints, status codes, and error envelopes exercised.
- **Failure-mode coverage:** every entry in [sdd.md §5](sdd.md) has at least one test or a documented deferral.
- **Release gate:** all CI checks green, SLAs met, and a manual Streamlit run-through passes.

### Failure-Mode Coverage

| # | Mitigation | Coverage |
| :-- | :-- | :-- |
| FM-1 | Page-count guard (`MAX_PAGES`) + batched embed/upsert | `tests/test_ingestion_service.py::test_ingest_rejects_documents_over_max_pages`, `::test_ingest_batches_embeddings_and_upserts`; upload caps in `tests/test_api.py` |
| FM-2 | Hybrid BM25 + RRF + rerank; golden regression gate | `tests/test_retrieval.py`; Ragas gate (`tests/test_ragas_eval.py`, `-m eval`) |
| FM-3 | `temperature=0.0`, abstention, citation audit | `tests/test_generator.py`; `tests/test_query_api.py` |
| FM-4 | Header-aware separators + 80-token overlap | `tests/test_ingestion.py` |
| FM-5 | Typed `500`/`503` envelopes, bounded retries | `tests/test_query_api.py`, `tests/test_generator.py`; degraded-mode (return sources on generation failure) is a **documented deferral** — the frozen error contract mandates `500 upstream_llm_error` |
| FM-6 | `empty_collection` guard | `tests/test_retrieval.py`, `tests/test_query_api.py` |
