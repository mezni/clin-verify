# Technical Specifications: ClinVerify Hybrid RAG Engine

## 1. API Endpoints & Request/Response Contracts

### 1.1 Ingest Regulatory Document (`POST /ingest`)

**Description:** Parses an FDA drug label or clinical trial PDF using `pymupdf4llm`, chunks the text while preserving markdown tables, computes dense and sparse embeddings via FastEmbed, and upserts points to Qdrant.

- **Content-Type:** `multipart/form-data`
- **Query Parameters:**
  - `doc_id` (string, required): Unique identifier for provenance (e.g., `FDA_Apixaban_2025`).
- **Request Payload:** Binary PDF file under form field `file`.

**Response JSON:**

```json
{
  "status": "success",
  "doc_id": "FDA_Apixaban_2025",
  "pages_processed": 24,
  "chunks_indexed": 185
}
```

### 1.2 Hybrid Retrieve & Generate (`POST /query`)

**Description:** Executes hybrid retrieval (dense + BM25 sparse prefetch with RRF fusion), cross-encoder reranking, grounded LLM generation, and automated citation verification.

- **Content-Type:** `application/json`

**Request Body Schema:**

```json
{
  "query": "What is the reduced dose of Apixaban for patients with serum creatinine >= 1.5 mg/dL?",
  "mode": "hybrid_rerank",
  "top_k": 5
}
```

**Response JSON Schema:**

```json
{
  "query": "What is the reduced dose of Apixaban for patients with serum creatinine >= 1.5 mg/dL?",
  "answer": "For patients with serum creatinine >= 1.5 mg/dL accompanied by age >= 80 years or body weight <= 60 kg, the recommended dose of apixaban is 2.5 mg orally twice daily [FDA_Apixaban_2025, p.4].",
  "sources": [
    {
      "text": "Table 1: Dose Adjustments... In patients with any 2 of the following: age >= 80, body weight <= 60 kg, or serum creatinine >= 1.5 mg/dL, the dose is 2.5 mg BID.",
      "citation": "[FDA_Apixaban_2025, p.4]",
      "rerank_score": 4.821,
      "score": 0.892
    }
  ],
  "telemetry": {
    "retrieval_ms": 114.2,
    "rerank_ms": 342.6,
    "generation_ms": 312.1,
    "total_ms": 768.9
  },
  "citation_verified": true
}
```

### 1.3 Health Check (`GET /health`)

Liveness probe used by Docker Compose healthchecks.

```json
{ "status": "ok" }
```

### 1.4 Error Contracts

All non-2xx responses share a single envelope:

```json
{
  "error": "validation_error",
  "detail": "top_k must be between 1 and 20",
  "request_id": "b3f1c2e4-9d7a-4c1e-8f2a-1a2b3c4d5e6f"
}
```

| Status | `error` | Trigger |
| :--- | :--- | :--- |
| **400** | `invalid_document` | Non-PDF MIME, corrupt/encrypted PDF, unreadable bytes, empty file, or over the upload/page limits. |
| **400** | `empty_collection` | `/query` issued before any document is ingested. |
| **422** | `validation_error` | Pydantic schema violation (`top_k` out of range, `query` too short, bad `mode`). |
| **500** | `upstream_llm_error` | OpenAI API failure/timeout, or `OPENAI_API_KEY` not configured. |
| **500** | `qdrant_unavailable` | Vector store connection failure. |
| **503** | `reranker_unavailable` | Cross-encoder model failed to load. |
| **500** | `internal_error` | Any other unexpected server error. |

---

## 2. Qdrant Collection & Payload Schema

**Collection Name:** `clinical_trials_hybrid`

### Vector Configurations

| Vector | Configuration |
| :--- | :--- |
| **Dense** (`dense`) | Size: 384 dimensions · Distance Metric: Cosine · Model: `BAAI/bge-small-en-v1.5` |
| **Sparse** (`bm25`) | Parameter Type: `SparseVectorParams` with server-side IDF modifier (`models.Modifier.IDF`) · Model: `Qdrant/bm25` (FastEmbed local ONNX) |

### Payload Metadata Structure per Point

```json
{
  "doc_id": "FDA_Apixaban_2025",
  "page_num": 4,
  "citation": "[FDA_Apixaban_2025, p.4]",
  "text": "Table 1: Dose Adjustments for Apixaban..."
}
```

---

## 3. Pipeline Hyperparameters & Model Specifications

| Component | Specification / Model | Operational Purpose |
| :--- | :--- | :--- |
| **PDF Parser** | `pymupdf4llm` | Preserves markdown tables and hierarchical headers (`##`, `###`) to prevent tabular fragmentation. |
| **Chunking Strategy** | LangChain `RecursiveCharacterTextSplitter` | `chunk_size = 600` tokens, `chunk_overlap = 80` tokens. Separators ordered to split at table headers (`\n## `, `\n### `, `\n\n`, `\n`, ` `). |
| **Ingestion Guards** | `MAX_UPLOAD_MB=50`, `MAX_PAGES=300`, `EMBED_BATCH_SIZE=64` | Reject oversized documents before parsing and bound peak memory by embedding/upserting in batches (FM-1). |
| **Dense Embedder** | `BAAI/bge-small-en-v1.5` | Generates 384-dim semantic embeddings locally via ONNX runtime. |
| **Sparse Embedder** | `Qdrant/bm25` (FastEmbed) | Computes token frequency and IDF for exact lexical and numerical matching. |
| **Retrieval Fusion** | Reciprocal Rank Fusion (RRF) | Merges the top-20 results from dense and sparse prefetch queries using RRF (rank constant `k=60`). |
| **Reranker** | `BAAI/bge-reranker-base` | Cross-encoder model performing joint query-document scoring to evaluate clinical negation and context relevance ($k=5$). |
| **Generator LLM** | `gpt-4o-mini` (`temperature=0.0`) | Enforces deterministic outputs, zero hallucination, and strict inline citation grounding. |

---

## 4. Golden Dataset & Evaluation Schema (`golden_dataset.json`)

The evaluation test suite validates retrieval precision and generation faithfulness (`Faithfulness`, `LLMContextPrecisionWithReference`) against a structured JSON schema:

```json
[
  {
    "id": "q_001",
    "question": "What is the reduced dose of Apixaban for patients with serum creatinine >= 1.5 mg/dL?",
    "ground_truth": "Patients meeting at least two criteria including serum creatinine >= 1.5 mg/dL receive 2.5 mg twice daily.",
    "expected_doc_id": "FDA_Apixaban_2025",
    "expected_page": 4
  }
]
```
