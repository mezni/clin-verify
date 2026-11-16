"""Document ingestion and parsing pipeline.

Phase 1 implements layout-aware PDF parsing (``pymupdf4llm``), header-aware
chunking (LangChain ``RecursiveCharacterTextSplitter``), and dense/sparse
embedding generation (FastEmbed) upserted into Qdrant.

See ``docs/plan.md`` (Phase 1) and ``docs/adr/0006-table-preserving-chunking.md``.
"""

__all__: list[str] = []
