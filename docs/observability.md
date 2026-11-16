# Observability

ClinVerify emits structured telemetry on every request so answers can be
audited and SLAs monitored. See `docs/sdd.md` §6.1 for the design.

## 1. Structured JSON Logging

- `src/clinverify/logging.py` installs a single `JsonFormatter` handler on the
  `clinverify` logger (stdout, one JSON object per line).
- `configure_logging()` is idempotent and called at API startup.
- `log_event(logger, message, **fields)` merges arbitrary structured fields into
  the record; `REQUEST_LOGGER` (`clinverify.request`) carries request logs.

Example line:

```json
{"level": "INFO", "logger": "clinverify.request", "message": "request", "request_id": "8f2c…", "method": "POST", "path": "/query", "status_code": 200, "duration_ms": 412.7, "mode": "hybrid_rerank", "top_k": 5, "candidates_prefetch": 20, "retrieval_ms": 88.1, "rerank_ms": 240.3, "generation_ms": 84.3, "total_ms": 412.7, "citation_verified": true, "citation_unresolved_count": 0, "abstained": false}
```

## 2. Request Correlation

- Every request gets a `request_id` (from the `X-Request-ID` header or a new
  UUID), stored on `request.state` and echoed in the `X-Request-ID` response
  header and in every error envelope.
- Use `request_id` to join a client-visible error to its server-side log line.

## 3. Query Telemetry

`POST /query` records stage timings and quality flags both in the log line and in
the response `telemetry` object:

| Field | Meaning |
| :-- | :-- |
| `retrieval_ms` | Dense+sparse prefetch and RRF fusion time |
| `rerank_ms` | Cross-encoder reranking time |
| `generation_ms` | LLM generation time |
| `total_ms` | Sum of the three stages |
| `candidates_prefetch` | Candidate pool size before reranking |
| `citation_verified` | Whether every citation mapped to a retrieved chunk |
| `citation_unresolved_count` | Number of unmapped citation tags |
| `abstained` | Whether the model returned `INSUFFICIENT_CONTEXT` |

## 4. Error Taxonomy

Errors are mapped to typed envelopes (`error`, `detail`, `request_id`) by
exception handlers in `src/clinverify/api.py`:

| Condition | Status | `error` |
| :-- | :-- | :-- |
| Invalid/oversized/unreadable PDF | `400` | `invalid_document` |
| Query before any ingest | `400` | `empty_collection` |
| Request schema violation | `422` | `validation_error` |
| LLM failure/timeout | `500` | `upstream_llm_error` |
| Qdrant unreachable | `500` | `qdrant_unavailable` |
| Reranker unavailable | `503` | `reranker_unavailable` |
| Unexpected | `500` | `internal_error` |

## 5. Health Checks

- `GET /health` returns `{"status": "ok"}`.
- Docker Compose healthchecks probe this endpoint (`api`), the Qdrant TCP port
  (`qdrant`), and Streamlit's `/_stcore/health` (`ui`) — see
  `docs/deployment.md` §6.

## 6. SLA Measurement

| SLA | Source |
| :-- | :-- |
| Faithfulness `≥ 0.95`, Context precision `≥ 0.88` | Ragas gate (`tests/test_ragas_eval.py`, `-m eval`) |
| End-to-end P95 latency `< 800 ms` | `latency_p95()` over `total_ms` samples (`src/clinverify/evaluation.py`) |

Latency is a **non-blocking report**: sample `total_ms` across the golden set
(from the log lines or the API `telemetry`), compute `latency_p95`, and compare
against `LATENCY_P95_TARGET_MS`. Trend regressions are investigated, not
merge-blocking.
