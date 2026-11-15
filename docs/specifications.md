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
    "generation_ms": 412.1,
    "total_ms": 868.9
  },
  "citation_verified": true
}
```

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
| **Chunking Strategy** | LangChain `RecursiveCharacterTextSplitter` | `chunk_size = 600` tokens, `chunk_overlap = 80` tokens. Separators ordered by table headers (`\n## `, `\n`). |
| **Dense Embedder** | `BAAI/bge-small-en-v1.5` | Generates 384-dim semantic embeddings locally via ONNX runtime. |
| **Sparse Embedder** | `Qdrant/bm25` (FastEmbed) | Computes token frequency and IDF for exact lexical and numerical matching. |
| **Retrieval Fusion** | Reciprocal Rank Fusion (RRF) | Merges top-$k$ results from dense and sparse prefetch queries ($k=20$). |
| **Reranker** | `BAAI/bge-reranker-base` | Cross-encoder model performing joint query-document scoring to evaluate clinical negation and context relevance ($k=5$). |
| **Generator LLM** | `gpt-4o-mini` (`temperature=0.0`) | Enforces deterministic outputs, zero hallucination, and strict inline citation grounding. |

---

## 4. Golden Dataset & Evaluation Schema (`golden_dataset.json`)

The evaluation test suite validates retrieval recall and generation faithfulness against a structured JSON schema:

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
