"""Headless tests for the Streamlit evidence inspector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from clinverify.client import ClinVerifyClient, QueryResult, Source, Telemetry

UI_PATH = str(Path(__file__).resolve().parents[1] / "src" / "clinverify" / "ui.py")


def _fake_query(
    self: ClinVerifyClient,
    query: str,
    mode: str = "hybrid_rerank",
    top_k: int = 5,
) -> QueryResult:
    return QueryResult(
        query=query,
        answer="The reduced dose is 2.5 mg twice daily [FDA_Apixaban_2025, p.4].",
        sources=[
            Source(
                text="The reduced dose is 2.5 mg twice daily.",
                citation="[FDA_Apixaban_2025, p.4]",
                score=0.892,
                rerank_score=4.821 if mode == "hybrid_rerank" else None,
            )
        ],
        telemetry=Telemetry(
            retrieval_ms=114.2, rerank_ms=342.6, generation_ms=312.1, total_ms=768.9
        ),
        citation_verified=True,
        request_id="rid-ui",
    )


def _fake_abstain(
    self: ClinVerifyClient,
    query: str,
    mode: str = "hybrid_rerank",
    top_k: int = 5,
) -> QueryResult:
    return QueryResult(
        query=query,
        answer="INSUFFICIENT_CONTEXT",
        sources=[],
        telemetry=Telemetry(retrieval_ms=10.0, rerank_ms=0.0, generation_ms=20.0, total_ms=30.0),
        citation_verified=False,
        request_id="rid-abstain",
    )


def _run_query(at: AppTest, query: str) -> None:
    at.text_area[0].set_value(query)
    at.button[0].click()
    at.run()


def test_ui_renders_without_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ClinVerifyClient, "query", _fake_query)
    at = AppTest.from_file(UI_PATH, default_timeout=30).run()

    assert not at.exception


def test_ui_renders_answer_and_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ClinVerifyClient, "query", _fake_query)
    at = AppTest.from_file(UI_PATH, default_timeout=30).run()

    _run_query(at, "What is the reduced dose of apixaban?")

    assert not at.exception
    rendered = " ".join(block.value for block in at.markdown)
    assert "2.5 mg" in rendered
    assert "[FDA_Apixaban_2025, p.4]" in rendered
    assert at.session_state is not None


def test_ui_mode_toggle_changes_result(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def _record(
        self: ClinVerifyClient, query: str, mode: str = "hybrid_rerank", top_k: int = 5
    ) -> Any:
        calls.append(mode)
        return _fake_query(self, query, mode=mode, top_k=top_k)

    monkeypatch.setattr(ClinVerifyClient, "query", _record)
    at = AppTest.from_file(UI_PATH, default_timeout=30).run()

    _run_query(at, "What is the reduced dose of apixaban?")
    at.radio[0].set_value("dense_only")
    _run_query(at, "What is the reduced dose of apixaban?")

    assert calls == ["hybrid_rerank", "dense_only"]


def test_ui_short_query_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ClinVerifyClient, "query", _fake_query)
    at = AppTest.from_file(UI_PATH, default_timeout=30).run()

    _run_query(at, "hi")

    assert not at.exception
    assert any("at least 3 characters" in warning.value for warning in at.warning)


def test_ui_renders_abstention(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ClinVerifyClient, "query", _fake_abstain)
    at = AppTest.from_file(UI_PATH, default_timeout=30).run()

    _run_query(at, "What is the dose of metformin?")

    assert not at.exception
    assert any("INSUFFICIENT_CONTEXT" in warning.value for warning in at.warning)
