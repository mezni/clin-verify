"""Unit tests for the ``POST /ingest`` HTTP contract and error handling."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from fastapi.testclient import TestClient

from clinverify.api import create_app
from clinverify.config import Settings
from clinverify.errors import InvalidDocumentError
from clinverify.ingestion import IngestResult

PDF_BYTES = b"%PDF-1.4\n%%EOF\n"
DOC_ID = "FDA_Apixaban_2025"


class FakeService:
    """Fake ingestion service recording calls and returning fixed results."""

    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, bytes]] = []

    def ingest(self, doc_id: str, data: bytes) -> IngestResult:
        self.calls.append((doc_id, data))
        if self.error is not None:
            raise self.error
        return IngestResult(doc_id=doc_id, pages_processed=24, chunks_indexed=185)


@contextmanager
def _client(service: FakeService, **settings_overrides: Any) -> Iterator[TestClient]:
    settings = Settings(_env_file=None, **settings_overrides)  # type: ignore[call-arg]
    app = create_app(settings=settings, ingestion=service)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def _upload(
    client: TestClient,
    *,
    data: bytes = PDF_BYTES,
    content_type: str = "application/pdf",
    doc_id: str = DOC_ID,
) -> Any:
    return client.post(
        "/ingest",
        params={"doc_id": doc_id},
        files={"file": ("label.pdf", data, content_type)},
    )


def test_health() -> None:
    with _client(FakeService()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ingest_success_contract() -> None:
    service = FakeService()
    with _client(service) as client:
        response = _upload(client)

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "doc_id": DOC_ID,
        "pages_processed": 24,
        "chunks_indexed": 185,
    }
    assert service.calls[0][0] == DOC_ID


def test_ingest_invalid_doc_id_returns_422() -> None:
    with _client(FakeService()) as client:
        response = _upload(client, doc_id="ab")

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_error"
    assert "request_id" in body


def test_ingest_unsupported_content_type_returns_400() -> None:
    with _client(FakeService()) as client:
        response = _upload(client, content_type="text/plain")

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_document"


def test_ingest_non_pdf_bytes_returns_400() -> None:
    with _client(FakeService()) as client:
        response = _upload(client, data=b"not a pdf")

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_document"


def test_ingest_empty_file_returns_400() -> None:
    with _client(FakeService()) as client:
        response = _upload(client, data=b"")

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_document"


def test_ingest_oversized_file_returns_400() -> None:
    oversized = b"%PDF-" + b"0" * (2 * 1024 * 1024)
    with _client(FakeService(), max_upload_mb=1) as client:
        response = _upload(client, data=oversized)

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_document"


def test_service_error_maps_to_400() -> None:
    service = FakeService(error=InvalidDocumentError("PDF is encrypted"))
    with _client(service) as client:
        response = _upload(client)

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_document"
    assert "encrypted" in response.json()["detail"]


def test_request_id_header_is_returned() -> None:
    with _client(FakeService()) as client:
        response = client.get("/health", headers={"X-Request-ID": "trace-123"})

    assert response.headers["X-Request-ID"] == "trace-123"


def test_request_id_generated_when_absent() -> None:
    with _client(FakeService()) as client:
        response = client.get("/health")

    assert response.headers["X-Request-ID"]
