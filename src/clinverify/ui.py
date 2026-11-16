"""Streamlit evidence inspector for ClinVerify.

Run with ``streamlit run src/clinverify/ui.py`` (or via Docker Compose). The UI
talks only to the FastAPI service — never to Qdrant directly.

See ``docs/plan.md`` (Phase 4) and ``docs/product.md`` §4.4.
"""

from __future__ import annotations

import streamlit as st

from clinverify.client import ApiError, ClinVerifyClient, QueryResult
from clinverify.config import get_settings

MODE_LABELS = {
    "hybrid_rerank": "Hybrid + Cross-Encoder Rerank",
    "dense_only": "Dense-Only Baseline",
}
ABSTENTION_TOKEN = "INSUFFICIENT_CONTEXT"


def render_answer(result: QueryResult) -> None:
    """Render the answer pane with its citation-audit status."""

    st.subheader("Answer")
    if result.answer.strip() == ABSTENTION_TOKEN:
        st.warning(
            "INSUFFICIENT_CONTEXT — the retrieved evidence does not contain an "
            "answer to this question."
        )
    else:
        st.markdown(result.answer)

    if result.citation_verified:
        st.success("Citations verified against retrieved chunks.")
    else:
        st.error("Citation audit failed — the answer contains unverified citations.")


def render_telemetry(result: QueryResult) -> None:
    """Render stage-by-stage latency telemetry."""

    st.subheader("Telemetry")
    telemetry = result.telemetry
    columns = st.columns(4)
    columns[0].metric("Retrieval (ms)", f"{telemetry.retrieval_ms:.1f}")
    columns[1].metric("Rerank (ms)", f"{telemetry.rerank_ms:.1f}")
    columns[2].metric("Generation (ms)", f"{telemetry.generation_ms:.1f}")
    columns[3].metric("Total (ms)", f"{telemetry.total_ms:.1f}")
    if result.request_id:
        st.caption(f"request_id: {result.request_id}")


def render_sources(result: QueryResult) -> None:
    """Render expandable source drawers with scores and raw Markdown."""

    st.subheader(f"Sources ({len(result.sources)})")
    for rank, source in enumerate(result.sources, start=1):
        rerank = "n/a" if source.rerank_score is None else f"{source.rerank_score:.3f}"
        label = f"#{rank} · {source.citation} · rerank {rerank} · score {source.score:.3f}"
        with st.expander(label):
            st.caption(f"Rank #{rank} · rerank_score {rerank} · retrieval score {source.score:.3f}")
            st.markdown(source.text)


def main() -> None:
    """Entry point for the Streamlit app."""

    settings = get_settings()
    st.set_page_config(page_title="ClinVerify", page_icon="🔎", layout="wide")
    st.title("ClinVerify — Clinical Evidence Inspector")

    with st.sidebar:
        st.header("Pipeline")
        base_url = st.text_input("API base URL", value=settings.api_base_url)
        mode = st.radio(
            "Retrieval mode",
            options=list(MODE_LABELS),
            format_func=lambda value: MODE_LABELS[value],
        )
        top_k = st.slider("Top-k sources", min_value=1, max_value=20, value=settings.rerank_top_k)

    query = st.text_area(
        "Question",
        height=100,
        placeholder="e.g. What is the reduced dose of apixaban for severe renal impairment?",
    )
    run = st.button("Run query", type="primary")

    if not run:
        return

    if len(query.strip()) < 3:
        st.warning("Enter a question of at least 3 characters.")
        return

    client = ClinVerifyClient(base_url)
    try:
        with st.spinner("Retrieving and generating…"):
            result = client.query(query.strip(), mode=mode, top_k=top_k)
    except ApiError as exc:
        st.error(f"**{exc.error}** — {exc.detail}")
        if exc.request_id:
            st.caption(f"request_id: {exc.request_id}")
        return

    render_answer(result)
    render_telemetry(result)
    render_sources(result)


if __name__ == "__main__":
    main()
