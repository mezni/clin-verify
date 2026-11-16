"""Unit tests for the ClinVerify HTTP client used by the Streamlit UI."""

from __future__ import annotations

import httpx
import pytest
import respx

from clinverify.client import ApiError, ClinVerifyClient

BASE_URL = "http://api.test"

QUERY_BODY = {
    "query": "What is the dose?",
    "answer": "2.5 mg BID [FDA_Apixaban_2025, p.4].",
    "sources": [
        {
            "text": "Apixaban 2.5 mg BID",
            "citation": "[FDA_Apixaban_2025, p.4]",
            "score": 0.9,
            "rerank_score": 4.821,
        }
    ],
    "telemetry": {
        "retrieval_ms": 114.2,
        "rerank_ms": 342.6,
        "generation_ms": 312.1,
        "total_ms": 768.9,
    },
    "citation_verified": True,
}


@respx.mock
def test_query_success() -> None:
    respx.post(f"{BASE_URL}/query").mock(
        return_value=httpx.Response(200, json=QUERY_BODY, headers={"X-Request-ID": "rid-1"})
    )
    client = ClinVerifyClient(BASE_URL)

    result = client.query("What is the dose?", mode="hybrid_rerank", top_k=5)

    assert result.answer == QUERY_BODY["answer"]
    assert result.citation_verified is True
    assert result.sources[0].rerank_score == 4.821
    assert result.telemetry.total_ms == 768.9
    assert result.request_id == "rid-1"


@respx.mock
def test_query_error_envelope_raises_api_error() -> None:
    respx.post(f"{BASE_URL}/query").mock(
        return_value=httpx.Response(
            400,
            json={"error": "empty_collection", "detail": "nothing ingested", "request_id": "rid-2"},
        )
    )
    client = ClinVerifyClient(BASE_URL)

    with pytest.raises(ApiError) as excinfo:
        client.query("What is the dose?")

    assert excinfo.value.status_code == 400
    assert excinfo.value.error == "empty_collection"
    assert excinfo.value.detail == "nothing ingested"
    assert excinfo.value.request_id == "rid-2"


@respx.mock
def test_query_non_json_error_uses_http_error() -> None:
    respx.post(f"{BASE_URL}/query").mock(return_value=httpx.Response(502, text="bad gateway"))
    client = ClinVerifyClient(BASE_URL)

    with pytest.raises(ApiError) as excinfo:
        client.query("What is the dose?")

    assert excinfo.value.error == "http_error"
    assert excinfo.value.status_code == 502


@respx.mock
def test_connection_error_raises_api_error() -> None:
    respx.post(f"{BASE_URL}/query").mock(side_effect=httpx.ConnectError("refused"))
    client = ClinVerifyClient(BASE_URL)

    with pytest.raises(ApiError) as excinfo:
        client.query("What is the dose?")

    assert excinfo.value.status_code == 0
    assert excinfo.value.error == "connection_error"


@respx.mock
def test_ingest_success() -> None:
    respx.post(f"{BASE_URL}/ingest").mock(
        return_value=httpx.Response(
            200,
            json={
                "status": "success",
                "doc_id": "FDA_Apixaban_2025",
                "pages_processed": 24,
                "chunks_indexed": 185,
            },
        )
    )
    client = ClinVerifyClient(BASE_URL)

    result = client.ingest("FDA_Apixaban_2025", "label.pdf", b"%PDF-1.4")

    assert result.chunks_indexed == 185
    assert result.doc_id == "FDA_Apixaban_2025"


@respx.mock
def test_health_true_and_false() -> None:
    route = respx.get(f"{BASE_URL}/health")
    route.mock(return_value=httpx.Response(200, json={"status": "ok"}))
    client = ClinVerifyClient(BASE_URL)
    assert client.health() is True

    route.mock(return_value=httpx.Response(503, json={"error": "unavailable"}))
    assert client.health() is False
