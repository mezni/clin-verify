# Security & Safety Model

ClinVerify serves regulated clinical/pharmacological content, so correctness and
grounding are treated as security properties. This document describes the threat
model, the controls in place, and known limitations.

## 1. Threat Model & Scope

| Asset | Threat | Control |
| :-- | :-- | :-- |
| Answer integrity | Fabricated dosage / citation (FM-3) | Grounded prompt, `temperature=0.0`, deterministic abstention, regex citation audit |
| Host resources | OOM via oversized PDF (FM-1) | Upload cap, content-type/magic-byte checks, page-count guard, batched embedding |
| Vector store | Unauthenticated access to Qdrant | Single service boundary; UI never talks to Qdrant; optional API key |
| Secrets | Leaked `OPENAI_API_KEY` | Env-only injection, never logged or committed |
| Supply chain | Malicious/compromised dependency | Lockfile pinning, CI gates, review |

Out of scope: multi-tenant isolation, end-user authentication/authorization, and
network-level hardening (handled by the deployment platform).

## 2. Secrets Management

- `OPENAI_API_KEY` is supplied via environment variables (`.env` locally, CI
  secret or orchestrator secret in deployment). It is **never** written to logs,
  responses, or the repository.
- `.env` is git-ignored; only `.env.example` (no secrets) is committed.
- Qdrant runs without auth in the local Compose topology; set
  `QDRANT_API_KEY` for any shared/remote deployment.

## 3. Input Validation & Resource Limits

- **Upload size cap** (`MAX_UPLOAD_MB`, default 50) enforced before parsing.
- **Content type** restricted to `application/pdf` and **magic bytes** must be
  `%PDF`.
- **Page-count guard** (`MAX_PAGES`, default 300) rejects oversized documents
  with `400 invalid_document` before parsing (FM-1).
- **`doc_id`** must match `^[A-Za-z0-9_\-]{3,128}$` (blocks path/injection
  attempts).
- **Batched embedding/upsert** (`EMBED_BATCH_SIZE`) bounds peak memory on large
  documents (FM-1).

## 4. Grounding & Prompt-Injection Containment

- The generation system prompt restricts the model to the **provided context
  only** and mandates abstention when evidence is missing (FM-3).
- `temperature=0.0` makes outputs deterministic.
- Every citation is validated by a regex auditor against the retrieved chunk set;
  unresolved citations flag the answer (`citation_verified=false`) rather than
  being silently accepted.
- Document text is passed as **context data**, not instructions; the model is
  told not to follow instructions embedded in retrieved content.
- The Ragas gate (`docs/adr/0008-ragas-ci-eval-gate.md`) blocks merges that
  regress faithfulness or context precision, catching drift in grounding.

## 5. Data Handling & Privacy

- The system stores document chunks and embeddings only; no user accounts or PII
  are collected.
- Structured logs contain request metadata and telemetry — never document text,
  prompts, answers, or secrets.
- Model weights are cached in a dedicated volume (`model_cache`); the Qdrant
  volume (`qdrant_storage`) holds the index. Neither is committed.

## 6. Network & Service Boundaries

- The Streamlit UI talks **only** to the API (`API_BASE_URL`); it never connects
  to Qdrant directly (ADR-0007).
- All failures return typed error envelopes with a `request_id` and no internal
  stack traces (see `docs/observability.md` §4).

## 7. Supply Chain & Container

- Dependencies are pinned in `uv.lock`; CI runs `uv sync --locked` so builds are
  reproducible.
- CI blocks on lint, typecheck, tests, and the Ragas gate before merge.
- The runtime image installs only production dependencies (`uv sync --no-dev`).

### Known Limitations

- The container currently runs as `root`; a non-root `USER` is a recommended
  hardening follow-up.
- No rate limiting or per-client quotas in the local topology; add at the
  gateway for public deployments.

## 8. Reporting

Report suspected vulnerabilities privately to the maintainers rather than via
public issues.
