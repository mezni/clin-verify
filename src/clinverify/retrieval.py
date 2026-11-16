"""Hybrid retrieval pipeline.

Phase 2 implements parallel dense + sparse (BM25) prefetch, Reciprocal Rank
Fusion (RRF), and cross-encoder reranking.

See ``docs/plan.md`` (Phase 2), ``docs/adr/0001-hybrid-retrieval-rrf.md``, and
``docs/adr/0002-cross-encoder-reranking.md``.
"""

__all__: list[str] = []
