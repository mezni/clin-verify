# Product Brief: ClinVerify — Hybrid Pharmacological & Clinical Trial RAG Engine

## 1. Executive Summary
**ClinVerify** is a high-precision Retrieval-Augmented Generation (RAG) engine and interactive clinical evidence inspector designed to extract exact drug dosing regimens, contraindications, and patient cohort eligibility criteria from dense regulatory documents (FDA drug labels and Phase III clinical trial protocols). 

Unlike general-purpose semantic RAG systems—which frequently conflate similar numerical values (e.g., `2.5 mg` vs. `25 mg`) or miss negation in exclusion tables—ClinVerify combines **lexical BM25 keyword retrieval**, **dense semantic embeddings**, and **cross-encoder reranking** inside a containerized microservice, enforcing strict inline source citations, automated CI/CD faithfulness gates, and a real-time Streamlit inspection UI.

---

## 2. Problem Statement
Clinical research coordinators, pharmacovigilance analysts, and medical writers must routinely cross-reference hundreds of pages of unstructured PDFs to verify patient eligibility and drug-drug interactions. Standard vector-only RAG pipelines fail in this domain due to three critical vulnerabilities:
1. **Numerical & Lexical Blindness:** Dense embedding models map distinct dosages (`2.5 mg` vs. `25 mg`) and biochemical markers (`CrCl < 30 mL/min`) into nearly identical vector spaces, causing dangerous retrieval misses.
2. **Table Fragmentation:** Naive character-based chunking splits multi-row FDA dosing tables across boundaries, separating conditions from their corresponding dose adjustments.
3. **Unverifiable Hallucinations:** Without deterministic citation verification and automated regression testing, LLM outputs cannot be trusted in regulated healthcare workflows.

---

## 3. Target Users & Core Use Cases
* **Clinical Trial Investigators & Coordinators:** Rapidly screen complex Inclusion/Exclusion criteria (e.g., hepatic impairment thresholds, prior therapy washout windows) across multi-arm Phase III protocols.
* **Pharmacovigilance & Medical Information Teams:** Query FDA prescribing information for CYP450 drug-drug interactions, boxed warnings, and renal/hepatic dose modifications with page-level traceability.
* **AI Platform & MLOps Engineers:** Evaluate and benchmark retrieval topologies (Dense vs. Sparse vs. Hybrid + Rerank) with automated quality gates prior to model or chunking changes.

---

## 4. Functional Requirements & System Capabilities

### 4.1 Layout-Aware Document Ingestion (`/ingest`)
* **Table-Preserving Parsing:** Converts multi-column regulatory PDFs into structured Markdown using `pymupdf4llm` to keep dosing and interaction matrices intact.
* **Granular Provenance Metadata:** Tags every chunk with `doc_id`, `page_num`, and `citation` strings for deterministic citation mapping.
* **Dual-Vector Indexing:** Generates both dense semantic vectors (`BAAI/bge-small-en-v1.5`) and sparse lexical representations (`Qdrant/bm25`) locally via `FastEmbed`.

### 4.2 Two-Stage Hybrid Retrieval & Reranking (`/query`)
* **Stage 1 — Parallel Prefetch & Fusion:** Executes simultaneous dense vector search and sparse BM25 keyword search in Qdrant, merging top candidates via **Reciprocal Rank Fusion (RRF)**.
* **Stage 2 — Cross-Encoder Reranking:** Scores query-document pairs jointly using a cross-encoder model (`BAAI/bge-reranker-base`) to resolve clinical negation and promote exact dosage matches to the top-$k$ context window.

### 4.3 Citation-Enforced Generation & Guardrails
* **Strict Grounding:** Operates at `temperature=0.0` with explicit instructions to answer *solely* from retrieved context chunks.
* **Programmatic Citation Verification:** Post-processes LLM responses using regex validation to confirm every emitted `[doc_id, p.X]` tag matches an actual retrieved chunk ID; flags unverified claims in the API response.
* **Deterministic Abstention:** Returns a structured `INSUFFICIENT_CONTEXT` response when regulatory evidence is absent rather than interpolating from pre-training weights.

### 4.4 Interactive Evidence Inspector (Streamlit UI)
* **Pipeline Topology Toggle:** Switch between `Hybrid + Cross-Encoder Rerank` and `Dense-Only Baseline` in real time to visualize retrieval gaps.
* **Stage-by-Stage Telemetry:** Monitor latency breakdowns (`retrieval_ms`, `rerank_ms`, `generation_ms`) and citation audit status per request.
* **Expandable Source Drawers:** Inspect raw extracted Markdown tables, RRF fusion ranks, and reranker scores alongside generated answers.

---

## 5. Non-Functional Requirements & Target SLAs

| Metric / Dimension | Target Threshold | Engineering Rationale |
| :--- | :--- | :--- |
| **Faithfulness (Ragas)** | `≥ 0.95` | Prevents unsupported clinical claims or dosage extrapolation. |
| **LLMContextPrecisionWithReference (Ragas)** | `≥ 0.88` | Ensures top-$k$ reranked chunks contain the exact target evidence. |
| **End-to-End Retrieval Latency (P95)** | `< 800 ms` | Balances cross-encoder accuracy with interactive clinical search UX. |
| **Deployment Footprint** | Single `docker compose up` | Self-contained FastAPI + Qdrant + Streamlit stack with local ONNX embeddings. |
