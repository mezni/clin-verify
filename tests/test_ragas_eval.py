"""Ragas evaluation gate for the golden dataset.

The deterministic tests (schema, coverage, threshold logic, retrieval-hit) run
on every commit and back the merge-blocking gate mechanism. The Ragas-backed
test calls a real LLM and only runs with ``CLINVERIFY_EVAL=1`` and
``OPENAI_API_KEY`` set (see ``docs/adr/0008-ragas-ci-eval-gate.md``).
"""

from __future__ import annotations

import os

import pytest

from clinverify.config import Settings
from clinverify.evaluation import (
    REQUIRED_CATEGORIES,
    EvalScores,
    EvalThresholds,
    GoldenEntry,
    evaluate_gate,
    load_golden_dataset,
    retrieval_hit,
)
from clinverify.generator import ABSTENTION_TOKEN, GenerationService, OpenAIChatClient
from clinverify.retrieval import RetrievedChunk

EVAL_ENABLED = os.getenv("CLINVERIFY_EVAL") == "1" and bool(os.getenv("OPENAI_API_KEY"))


def _sources_for(entry: GoldenEntry) -> list[RetrievedChunk]:
    doc_id = entry.expected_doc_id or entry.id
    page = entry.expected_page or 1
    return [
        RetrievedChunk(
            point_id=f"{entry.id}-{index}",
            doc_id=doc_id,
            page_num=page,
            citation=f"[{doc_id}, p.{page}]",
            text=text,
            score=1.0,
            rerank_score=None,
        )
        for index, text in enumerate(entry.contexts)
    ]


def test_golden_dataset_schema_is_valid() -> None:
    entries = load_golden_dataset()

    assert entries
    assert len({entry.id for entry in entries}) == len(entries)
    for entry in entries:
        assert entry.question
        assert entry.ground_truth
        if entry.expects_abstention:
            assert entry.expected_doc_id is None
        else:
            assert entry.expected_doc_id is not None
            assert entry.expected_page is not None


def test_golden_dataset_covers_all_categories() -> None:
    categories = {entry.category for entry in load_golden_dataset()}

    assert set(REQUIRED_CATEGORIES) <= categories


def test_gate_passes_at_thresholds() -> None:
    gate = evaluate_gate(EvalScores(faithfulness=0.95, context_precision=0.88))

    assert gate.passed is True
    assert gate.failures == []


def test_gate_blocks_on_faithfulness_regression() -> None:
    gate = evaluate_gate(EvalScores(faithfulness=0.90, context_precision=0.90))

    assert gate.passed is False
    assert any("faithfulness" in failure for failure in gate.failures)


def test_gate_blocks_on_context_precision_regression() -> None:
    gate = evaluate_gate(EvalScores(faithfulness=0.99, context_precision=0.80))

    assert gate.passed is False
    assert any("context_precision" in failure for failure in gate.failures)


def test_gate_honours_custom_thresholds() -> None:
    thresholds = EvalThresholds(faithfulness=0.5, context_precision=0.5)
    gate = evaluate_gate(EvalScores(faithfulness=0.6, context_precision=0.6), thresholds)

    assert gate.passed is True


def test_retrieval_hit_detects_expected_provenance() -> None:
    sources = _sources_for(load_golden_dataset()[0])

    assert retrieval_hit(sources, "FDA_Apixaban_2025", 4) is True
    assert retrieval_hit(sources, "FDA_Apixaban_2025", 99) is False
    assert retrieval_hit(sources, None, None) is False


@pytest.mark.eval
@pytest.mark.skipif(
    not EVAL_ENABLED,
    reason="set CLINVERIFY_EVAL=1 and OPENAI_API_KEY to run the Ragas evaluation gate",
)
def test_ragas_gate_passes_on_golden_dataset() -> None:
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.llms import llm_factory
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference

    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    generation = GenerationService(
        settings,
        OpenAIChatClient(
            settings.generator_model,
            settings.generator_temperature,
            settings.openai_api_key,
        ),
    )

    samples: list[SingleTurnSample] = []
    for entry in load_golden_dataset():
        sources = _sources_for(entry)
        result = generation.generate(entry.question, sources)
        if entry.expects_abstention:
            assert result.abstained is True
            assert result.answer.strip() == ABSTENTION_TOKEN
            continue
        assert result.citation_verified is True, entry.id
        samples.append(
            SingleTurnSample(
                user_input=entry.question,
                response=result.answer,
                retrieved_contexts=entry.contexts,
                reference=entry.ground_truth,
            )
        )

    dataset = EvaluationDataset(samples=samples)
    report = evaluate(
        dataset,
        metrics=[Faithfulness(), LLMContextPrecisionWithReference()],
        llm=llm_factory(settings.generator_model),
        show_progress=False,
    )
    frame = report.to_pandas()
    scores = EvalScores(
        faithfulness=float(frame["faithfulness"].mean()),
        context_precision=float(frame["llm_context_precision_with_reference"].mean()),
    )

    gate = evaluate_gate(scores)
    assert gate.passed, f"Ragas gate failed: {gate.failures}"
