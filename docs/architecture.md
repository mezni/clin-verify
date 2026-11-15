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

### 2.1 Ingestion & Parsing Pipeline (`src/ingestion.py`)
* **Tooling:** `pymupdf4llm` + LangChain `RecursiveCharacterTextSplitter`.
* **Process:** PDFs are parsed into structured Markdown preserving multi-column tables and section headers (`##`, `###`). Chunks are split at 600 tokens with an 80-token overlap.
* **Embedding Generation:** 
  * *Dense Vector:* Generated via `BAAI/bge-small-en-v1.5` (384 dimensions, cosine distance).
  * *Sparse Vector:* Generated via server-side IDF-modified BM25 (`Qdrant/bm25`).
* **Payload Metadata:** Each vector point stores `doc_id`, `page_num`, `citation` tag (e.g., `[FDA_Apixaban_2025, p.4]`), and chunk text.

### 2.2 Hybrid Retrieval & Cross-Encoder Reranking (`src/retrieval.py`)
* **Step 1 (Parallel Prefetch):** Qdrant executes concurrent dense vector search and sparse lexical BM25 search.
* **Step 2 (RRF Fusion):** Results are merged using Reciprocal Rank Fusion (RRF) to capture both semantic synonymy and exact token matches (vital for numerical dosages).
* **Step 3 (Cross-Encoder Reranking):** The top prefetch candidates (e.g., $k=20$) are passed to `BAAI/bge-reranker-base` for joint query-document scoring, filtering out noise and resolving negation. The top-$k$ ($k=5$) are forwarded to the LLM.

### 2.3 Grounded Generation & Citation Auditor (`src/generator.py`)
* **LLM:** `gpt-4o-mini` configured at `temperature=0.0`.
* **System Constraints:** Instructed to use *only* provided context, include inline citations, and refuse unanswerable queries (`INSUFFICIENT_CONTEXT`).
* **Auditor:** Post-processes the output to verify that every citation tag strictly maps to an ingested chunk in the retrieved payload.

### 2.4 CI/CD Evaluation Gate (`tests/test_ragas_eval.py`)
* **Framework:** PyTest + Ragas v0.4+.
* **Metrics:** Evaluates `Faithfulness` and `LLMContextPrecisionWithReference` against a golden dataset of curated pharmacological Q&A pairs on every pull request via GitHub Actions.

---

## 3. Technology Stack

| Layer | Technologies & Libraries |
| :--- | :--- |
| **API Framework** | FastAPI, Uvicorn, Pydantic v2 |
| **Vector Database** | Qdrant (Docker deployment, dense + sparse collections) |
| **Embeddings & Reranking** | FastEmbed (`bge-small-en-v1.5`, `bm25`), Sentence-Transformers (`bge-reranker-base`) |
| **PDF Parsing** | PyMuPDF4LLM |
| **LLM Generation** | OpenAI API (`gpt-4o-mini`) |
| **Evaluation** | Ragas, PyTest, Datasets |
| **Frontend UI** | Streamlit |
| **Containerization** | Docker, Docker Compose, Makefile |
