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

## Getting Started

### Prerequisites

- Docker and Docker Compose
- An OpenAI API key

### Setup

```bash
git clone <repo-url> clin-verify
cd clin-verify
cp .env.example .env   # add your OPENAI_API_KEY
docker compose up --build
```

The FastAPI backend and Streamlit UI will be available once the stack is healthy.

## Usage

1. **Ingest** regulatory PDFs via the `/ingest` endpoint.
2. **Query** the engine via `/query` or through the Streamlit evidence inspector.
3. Inspect retrieval telemetry, RRF ranks, and reranker scores for each answer.

## Evaluation

```bash
pytest tests/test_ragas_eval.py
```

Runs the Ragas faithfulness and context-precision gates against the curated golden dataset.

## License

See repository for license details.
