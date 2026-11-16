"""Golden-dataset loading and the Ragas evaluation gate.

Defines the version-controlled golden dataset schema, the merge-blocking
quality thresholds, and the deterministic gate logic. The Ragas orchestration
itself lives in ``tests/test_ragas_eval.py`` (ragas is a dev-only dependency).

See ``docs/plan.md`` (Phase 5), ``docs/adr/0008-ragas-ci-eval-gate.md``, and
``docs/testing.md`` §5.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel

Category = Literal["dosage", "contraindication", "negation", "eligibility", "abstention"]

REQUIRED_CATEGORIES: tuple[Category, ...] = (
    "dosage",
    "contraindication",
    "negation",
    "eligibility",
    "abstention",
)

DEFAULT_GOLDEN_DATASET = Path(__file__).resolve().parents[2] / "golden_dataset.json"


class GoldenEntry(BaseModel):
    """A single golden-dataset question with its expected provenance."""

    id: str
    category: Category
    question: str
    ground_truth: str
    expected_doc_id: str | None = None
    expected_page: int | None = None
    contexts: list[str] = []
    expects_abstention: bool = False


@dataclass(frozen=True)
class EvalThresholds:
    """Merge-blocking quality thresholds."""

    faithfulness: float = 0.95
    context_precision: float = 0.88


@dataclass(frozen=True)
class EvalScores:
    """Measured Ragas metrics for a run."""

    faithfulness: float
    context_precision: float


@dataclass(frozen=True)
class GateResult:
    """Outcome of comparing scores against thresholds."""

    passed: bool
    failures: list[str]
    scores: EvalScores
    thresholds: EvalThresholds


class SourceLike(Protocol):
    """A retrieved source with provenance, for retrieval-hit assertions."""

    @property
    def doc_id(self) -> str: ...

    @property
    def page_num(self) -> int: ...


def load_golden_dataset(path: Path | str = DEFAULT_GOLDEN_DATASET) -> list[GoldenEntry]:
    """Load and validate the golden dataset from ``path``."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [GoldenEntry.model_validate(item) for item in raw]


def evaluate_gate(scores: EvalScores, thresholds: EvalThresholds | None = None) -> GateResult:
    """Compare ``scores`` against ``thresholds`` and report any breaches.

    A threshold breach yields ``passed=False`` and a human-readable failure,
    which the CI gate turns into a merge block.
    """

    limits = thresholds or EvalThresholds()
    failures: list[str] = []
    if scores.faithfulness < limits.faithfulness:
        failures.append(f"faithfulness {scores.faithfulness:.3f} < {limits.faithfulness:.2f}")
    if scores.context_precision < limits.context_precision:
        failures.append(
            f"context_precision {scores.context_precision:.3f} < {limits.context_precision:.2f}"
        )
    return GateResult(
        passed=not failures,
        failures=failures,
        scores=scores,
        thresholds=limits,
    )


def retrieval_hit(sources: Sequence[SourceLike], doc_id: str | None, page: int | None) -> bool:
    """Return ``True`` when ``doc_id``/``page`` appears within the sources."""

    if doc_id is None or page is None:
        return False
    return any(source.doc_id == doc_id and source.page_num == page for source in sources)
