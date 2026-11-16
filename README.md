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
src/clinverify/    # application package (ingestion, retrieval, generator, api)
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

1. **Ingest** regulatory PDFs via the `/ingest` endpoint.
2. **Query** the engine via `/query` or through the Streamlit evidence inspector.
3. Inspect retrieval telemetry, RRF ranks, and reranker scores for each answer.

## Evaluation

The Ragas faithfulness and context-precision gates run against a curated golden
dataset. The suite is introduced in Phase 5 — see [docs/testing.md](docs/testing.md).

```bash
uv run pytest tests/test_ragas_eval.py
```

## License

Proprietary. All rights reserved.
