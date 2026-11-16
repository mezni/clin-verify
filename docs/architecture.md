# System Architecture: ClinVerify Hybrid RAG Engine

## 1. System Topology & Data Flow

ClinVerify is structured as a containerized microservices application composed of a **FastAPI Backend**, a **Qdrant Vector Database**, and a **Streamlit Frontend**, orchestrated via Docker Compose.

```
[ Regulatory PDFs ]
        │
        ▼
[ PyMuPDF4LLM Parser ] ──> [ Markdown Tables & Chunks ]
        │
        ▼
[ FastEmbed (Dense & Sparse) ]
        │
        ▼
[ Qdrant Storage ]
        ▲
        │ (Hybrid RRF Prefetch)
[ User / Streamlit UI ] ──> [ FastAPI Service ] ──> [ Cross-Encoder Rerank ]
                                        │
                                        ▼
                              [ OpenAI GPT-4o-mini (temp=0.0) ]
                                        │
                                        ▼
                              [ Regex Citation Auditor ] ──> [ Verified Response ]
```

---

## 2. Component Breakdown

### 2.1 Ingestion & Parsing Pipeline (`src/clinverify/ingestion.py`)
* **Tooling:** `pymupdf4llm` + LangChain `RecursiveCharacterTextSplitter`.
* **Process:** PDFs are parsed into structured Markdown preserving multi-column tables and section headers (`##`, `###`). Chunks are split at 600 tokens with an 80-token overlap.
* **Embedding Generation:** 
  * *Dense Vector:* Generated via `BAAI/bge-small-en-v1.5` (384 dimensions, cosine distance).
  * *Sparse Vector:* Generated via server-side IDF-modified BM25 (`Qdrant/bm25`).
* **Payload Metadata:** Each vector point stores `doc_id`, `page_num`, `citation` tag (e.g., `[FDA_Apixaban_2025, p.4]`), and chunk text.

### 2.2 Hybrid Retrieval & Cross-Encoder Reranking (`src/clinverify/retrieval.py`)
* **Step 1 (Parallel Prefetch):** Qdrant executes concurrent dense vector search and sparse lexical BM25 search.
* **Step 2 (RRF Fusion):** Results are merged using Reciprocal Rank Fusion (RRF) to capture both semantic synonymy and exact token matches (vital for numerical dosages).
* **Step 3 (Cross-Encoder Reranking):** The top prefetch candidates (e.g., $k=20$) are passed to `BAAI/bge-reranker-base` for joint query-document scoring, filtering out noise and resolving negation. The top-$k$ ($k=5$) are forwarded to the LLM.

### 2.3 Grounded Generation & Citation Auditor (`src/clinverify/generator.py`)
* **LLM:** `gpt-4o-mini` configured at `temperature=0.0` (fails fast with `500 upstream_llm_error` when `OPENAI_API_KEY` is unset).
* **System Constraints:** Instructed to use *only* provided context, include inline citations, and refuse unanswerable queries (`INSUFFICIENT_CONTEXT`).
* **Auditor:** Post-processes the output to verify that every citation tag strictly maps to an ingested chunk in the retrieved payload.

### 2.4 API, Config & Telemetry (`src/clinverify/api.py`, `config.py`, `errors.py`, `logging.py`)
* **API:** FastAPI app exposing `POST /ingest`, `POST /query`, and `GET /health`; shared Qdrant client + FastEmbed embedder are created in the lifespan.
* **Config:** Pydantic `Settings` (env-driven) for models, hyperparameters, upload/page guards, and the UI's `API_BASE_URL`.
* **Errors:** Domain exceptions mapped to typed envelopes (`invalid_document`, `empty_collection`, `upstream_llm_error`, `qdrant_unavailable`, `reranker_unavailable`).
* **Telemetry:** One structured JSON log line per request (correlated by `request_id`), carrying stage timings and citation/abstention flags.

### 2.5 Evidence Inspector (`src/clinverify/ui.py`, `client.py`)
* **UI:** Streamlit inspector with a query box, citation-audit status, `hybrid_rerank` ↔ `dense_only` toggle, per-stage telemetry, and expandable source drawers.
* **Client:** Thin HTTP client (`ClinVerifyClient`) that talks only to the API — never to Qdrant directly.

### 2.6 CI/CD Evaluation Gate (`src/clinverify/evaluation.py`, `tests/test_ragas_eval.py`)
* **Framework:** PyTest + Ragas (dev-only dependency).
* **Metrics:** Evaluates `Faithfulness` (`≥ 0.95`) and `LLMContextPrecisionWithReference` (`≥ 0.88`) against `golden_dataset.json` on every pull request via GitHub Actions.
* **Gate logic:** Dataset schema/coverage/threshold checks are deterministic and run on every commit; the Ragas-backed run is a required CI check (ADR-0008).

---

## 3. Deployment Topology

Three containers orchestrated by Docker Compose (ADR-0007); see [deployment.md](deployment.md):

```
[ Streamlit UI :8501 ] ──HTTP──> [ FastAPI :8000 ] ──> [ Qdrant :6333 ]
                                      │                    (volume)
                                      └──> OpenAI API
```

- Healthcheck-gated startup (`qdrant` → `api` → `ui`).
- Persistent volumes for the index (`qdrant_storage`) and model cache (`model_cache`).
- CPU-only PyTorch keeps the runtime image lean; models download on first run.

---

## 4. Technology Stack

| Layer | Technologies & Libraries |
| :--- | :--- |
| **API Framework** | FastAPI, Uvicorn, Pydantic v2 |
| **Vector Database** | Qdrant (Docker deployment, dense + sparse collections) |
| **Embeddings & Reranking** | FastEmbed (`bge-small-en-v1.5`, `bm25`), Sentence-Transformers (`bge-reranker-base`) on CPU-only PyTorch |
| **PDF Parsing** | PyMuPDF4LLM |
| **LLM Generation** | OpenAI API (`gpt-4o-mini`) |
| **Evaluation** | Ragas, PyTest, Datasets |
| **Frontend UI** | Streamlit |
| **Observability** | Structured JSON logging, request-scoped telemetry |
| **Containerization** | Docker, Docker Compose, Makefile |
