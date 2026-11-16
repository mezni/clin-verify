# ClinVerify

**Hybrid Pharmacological & Clinical Trial RAG Engine**

ClinVerify is a high-precision Retrieval-Augmented Generation (RAG) engine and interactive clinical evidence inspector. It extracts exact drug dosing regimens, contraindications, and patient cohort eligibility criteria from dense regulatory documents such as FDA drug labels and Phase III clinical trial protocols.

Unlike general-purpose semantic RAG systems, ClinVerify combines **lexical BM25 keyword retrieval**, **dense semantic embeddings**, and **cross-encoder reranking** inside a containerized microservice — enforcing strict inline source citations and automated faithfulness gates.

## Features

- **Layout-Aware Ingestion** — Table-preserving PDF parsing with `pymupdf4llm`, chunked into structured Markdown with page-level provenance metadata.
- **Two-Stage Hybrid Retrieval** — Parallel dense + sparse (BM25) prefetch in Qdrant, fused via Reciprocal Rank Fusion (RRF), then refined with a cross-encoder reranker.
- **Citation-Enforced Generation** — `gpt-4o-mini` at `temperature=0.0` with programmatic regex citation verification and deterministic `INSUFFICIENT_CONTEXT` abstention.
- **Interactive Evidence Inspector** — Streamlit UI with pipeline topology toggle, per-stage latency telemetry, and expandable source drawers.
- **CI/CD Evaluation Gate** — Ragas-based `Faithfulness` and `LLMContextPrecisionWithReference` metrics on every pull request.

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full system topology and component breakdown. Product context lives in [docs/product.md](docs/product.md).

```
[ Regulatory PDFs ] → [ PyMuPDF4LLM Parser ] → [ FastEmbed (Dense & Sparse) ] → [ Qdrant ]
[ Streamlit UI ] → [ FastAPI ] → [ Cross-Encoder Rerank ] → [ GPT-4o-mini ] → [ Citation Auditor ]
```

## Tech Stack

| Layer | Technologies |
| :--- | :--- |
| API Framework | FastAPI, Uvicorn, Pydantic v2 |
| Vector Database | Qdrant (dense + sparse collections) |
| Embeddings & Reranking | FastEmbed (`bge-small-en-v1.5`, `bm25`), Sentence-Transformers (`bge-reranker-base`) |
| PDF Parsing | PyMuPDF4LLM |
| LLM Generation | OpenAI API (`gpt-4o-mini`) |
| Evaluation | Ragas, PyTest, Datasets |
| Frontend UI | Streamlit |
| Containerization | Docker, Docker Compose, Makefile |

## Documentation

| Document | Description |
| :--- | :--- |
| [docs/product.md](docs/product.md) | Product brief, users, requirements, and SLAs |
| [docs/architecture.md](docs/architecture.md) | System topology and component breakdown |
| [docs/specifications.md](docs/specifications.md) | API contracts, Qdrant schema, and hyperparameters |
| [docs/sdd.md](docs/sdd.md) | Specification-driven design (contracts, failure modes, telemetry) |
| [docs/plan.md](docs/plan.md) | Implementation roadmap (Phases 0–6) |
| [docs/testing.md](docs/testing.md) | Test strategy and CI quality gates |
| [docs/adr/](docs/adr/README.md) | Architecture Decision Records |
| [CHANGELOG.md](CHANGELOG.md) | Release history |

## Project Layout

```
src/clinverify/    # application package (ingestion, retrieval, generator, api, client, ui)
tests/             # unit, integration, and evaluation tests
docs/              # product, architecture, specs, SDD, plan, ADRs
.github/workflows/ # CI (lint → typecheck → test)
```

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
git clone <repo-url> clin-verify
cd clin-verify
uv sync                # create the venv and install dependencies
cp .env.example .env   # set OPENAI_API_KEY and other settings
```

Run the quality gates (mirrored in CI):

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

## Deployment

Container orchestration (FastAPI + Qdrant + Streamlit via Docker Compose) is
delivered in Phase 6 — see [docs/plan.md](docs/plan.md). Once available:

```bash
docker compose up --build
```

## Usage

Run the API locally (requires a Qdrant instance reachable at `QDRANT_URL`):

```bash
uv run uvicorn clinverify.api:app --reload
```

Ingest a regulatory PDF:

```bash
curl -X POST "http://localhost:8000/ingest?doc_id=FDA_Apixaban_2025" \
  -F "file=@label.pdf;type=application/pdf"
# {"status":"success","doc_id":"FDA_Apixaban_2025","pages_processed":24,"chunks_indexed":185}
```

`POST /ingest` parses the PDF, chunks it at table-header boundaries, embeds each
chunk (dense + sparse), and upserts points into Qdrant. Re-ingesting the same
`doc_id` replaces the prior version (idempotent).

Query the indexed evidence with the full pipeline (dense + BM25 prefetch → RRF
fusion → cross-encoder reranking → grounded generation → citation audit):

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the reduced dose of apixaban?", "mode": "hybrid_rerank", "top_k": 5}'
# {"query":"...","answer":"The reduced dose is 2.5 mg twice daily [FDA_Apixaban_2025, p.4].","sources":[...],"telemetry":{...},"citation_verified":true}
```

Generation runs `gpt-4o-mini` at `temperature=0.0` (set `OPENAI_API_KEY`) and
answers solely from the retrieved context, emitting inline `[doc_id, p.X]`
citations that are programmatically verified — unresolved tags set
`citation_verified=false`. When the context lacks the answer, `/query` returns
`INSUFFICIENT_CONTEXT` instead of interpolating. Set `"mode": "dense_only"` for
the baseline retrieval path. Querying before any document is ingested returns
`400 empty_collection`.

### Evidence inspector (Streamlit)

Launch the inspector against a running API (defaults to `API_BASE_URL`):

```bash
uv run streamlit run src/clinverify/ui.py
```

The UI provides a query box, an answer pane with citation-audit status, a
`hybrid_rerank` ↔ `dense_only` pipeline toggle, stage-by-stage telemetry
(`retrieval_ms`, `rerank_ms`, `generation_ms`, `total_ms`), and expandable
source drawers showing each chunk's rank, reranker score, retrieval score, and
raw Markdown.

Every request emits one structured JSON log line to stdout (correlated by
`request_id`, also returned in the `X-Request-ID` header and error envelopes).

### Running tests

```bash
uv run pytest                                  # unit tests (+ deterministic eval-gate checks)
CLINVERIFY_INTEGRATION=1 uv run pytest         # + live Qdrant/FastEmbed tests
CLINVERIFY_EVAL=1 uv run pytest -m eval        # + Ragas gate (calls a real LLM; needs OPENAI_API_KEY)
```

## Evaluation

The merge-blocking Ragas gate evaluates answer quality against the
version-controlled [`golden_dataset.json`](golden_dataset.json), covering
dosage, contraindication, negation, eligibility, and abstention cases
(`docs/adr/0008-ragas-ci-eval-gate.md`):

- `Faithfulness ≥ 0.95`
- `LLMContextPrecisionWithReference ≥ 0.88`

The dataset schema, coverage checks, retrieval-hit assertions, and the
threshold logic are deterministic and run on every commit; a simulated
regression below threshold fails the gate. The Ragas-backed run calls a real
LLM and is wired as a required CI check (`.github/workflows/ci.yml` →
`eval-gate`); set the `OPENAI_API_KEY` repository secret to enable it.

```bash
CLINVERIFY_EVAL=1 OPENAI_API_KEY=sk-... uv run pytest tests/test_ragas_eval.py -m eval
```

## License

Proprietary. All rights reserved.
