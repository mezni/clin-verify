"""Grounded generation and citation auditing.

Phase 3 implements deterministic generation (``gpt-4o-mini``, temperature 0.0),
deterministic ``INSUFFICIENT_CONTEXT`` abstention, and regex-based citation
verification.

See ``docs/plan.md`` (Phase 3) and
``docs/adr/0004-deterministic-generation-citation-audit.md``.
"""

__all__: list[str] = []
