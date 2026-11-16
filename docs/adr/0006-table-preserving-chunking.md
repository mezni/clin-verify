# ADR-0006: Table-preserving parsing and header-aware chunking

- **Status:** Accepted
- **Date:** 2026-11-15
- **Deciders:** Architecture team
- **Related:** [sdd.md §4.1](../sdd.md), [architecture.md §2.1](../architecture.md), [specifications.md §3](../specifications.md)

## Context

FDA dosing tables and clinical inclusion/exclusion matrices are multi-row structures where a condition (e.g., "serum creatinine >= 1.5 mg/dL") must stay bound to its corresponding dose adjustment (e.g., "2.5 mg BID"). Naive character-based chunking splits these across boundaries, separating a condition from its value and producing dangerously misleading context. Column-major PDF extraction similarly scrambles table cell order.

## Decision

- Parse PDFs with **`pymupdf4llm`**, which preserves tables as Markdown and headers as `##`/`###`.
- Chunk with LangChain **`RecursiveCharacterTextSplitter`** at `chunk_size=600`, `chunk_overlap=80`, using a separator priority that splits first at table/section headers (`\n## `, `\n### `, `\n\n`, `\n`, ` `) so rows stay grouped with their section.
- Store `page_num` per chunk for page-level provenance.

## Consequences

### Positive
- Table rows remain intact with their headers within a chunk.
- Markdown structure is human-inspectable in the UI source drawers.
- Overlap mitigates residual boundary loss.

### Negative / Trade-offs
- Very wide tables may still exceed `chunk_size`; splitting within a table is a known edge case to monitor.
- Markdown tables consume more tokens than raw text, reducing effective content per chunk.
- Note: `RecursiveCharacterTextSplitter` measures **characters**, not tokens; the "600 tokens" target is an approximation to be validated in testing.

## Alternatives Considered

- **PyPDF2 / raw text extraction** — rejected: loses table structure.
- **Semantic/embedding-based chunking** — deferred: higher complexity; header-based splitting is sufficient and deterministic.
- **Fixed-size character chunking** — rejected: fragments tables.
