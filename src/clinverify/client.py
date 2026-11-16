"""HTTP client for the ClinVerify API.

Consumed by the Streamlit evidence inspector (``clinverify.ui``). Keeps the UI
free of transport concerns and gives the request/response contracts a single
typed home outside the server package.
"""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel


class Source(BaseModel):
    """A retrieved source chunk returned by ``POST /query``."""

    text: str
    citation: str
    score: float
    rerank_score: float | None = None


class Telemetry(BaseModel):
    """Per-request stage timings returned by ``POST /query``."""

    retrieval_ms: float
    rerank_ms: float
    generation_ms: float
    total_ms: float


class QueryResult(BaseModel):
    """Parsed ``POST /query`` response."""

    query: str
    answer: str
    sources: list[Source]
    telemetry: Telemetry
    citation_verified: bool
    request_id: str | None = None


class IngestResult(BaseModel):
    """Parsed ``POST /ingest`` response."""

    status: str
    doc_id: str
    pages_processed: int
    chunks_indexed: int
    request_id: str | None = None


class ApiError(Exception):
    """Raised when the API returns a non-2xx response or is unreachable."""

    def __init__(
        self,
        status_code: int,
        error: str,
        detail: str,
        request_id: str | None = None,
    ) -> None:
        super().__init__(f"{status_code} {error}: {detail}")
        self.status_code = status_code
        self.error = error
        self.detail = detail
        self.request_id = request_id


class ClinVerifyClient:
    """Thin synchronous client over the ClinVerify HTTP API."""

    def __init__(self, base_url: str, *, timeout: float = 120.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def query(self, query: str, mode: str = "hybrid_rerank", top_k: int = 5) -> QueryResult:
        """Run ``POST /query`` and return the parsed result."""

        response = self._request(
            "POST",
            "/query",
            json={"query": query, "mode": mode, "top_k": top_k},
        )
        return QueryResult(**response.json(), request_id=response.headers.get("X-Request-ID"))

    def ingest(
        self,
        doc_id: str,
        filename: str,
        data: bytes,
        content_type: str = "application/pdf",
    ) -> IngestResult:
        """Run ``POST /ingest`` for a PDF document."""

        response = self._request(
            "POST",
            "/ingest",
            params={"doc_id": doc_id},
            files={"file": (filename, data, content_type)},
        )
        return IngestResult(**response.json(), request_id=response.headers.get("X-Request-ID"))

    def health(self) -> bool:
        """Return ``True`` when the API health endpoint responds."""

        try:
            response = self._request("GET", "/health")
        except ApiError:
            return False
        return response.status_code == 200

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = httpx.request(
                method,
                f"{self._base_url}{path}",
                timeout=self._timeout,
                **kwargs,
            )
        except httpx.HTTPError as exc:
            raise ApiError(0, "connection_error", str(exc)) from exc
        if response.status_code >= 400:
            self._raise_api_error(response)
        return response

    @staticmethod
    def _raise_api_error(response: httpx.Response) -> None:
        error = "http_error"
        detail = response.text[:500]
        request_id = response.headers.get("X-Request-ID")
        try:
            body = response.json()
        except ValueError:
            body = {}
        if isinstance(body, dict):
            error = str(body.get("error", error))
            detail = str(body.get("detail", detail))
            request_id = body.get("request_id", request_id)
        raise ApiError(response.status_code, error, detail, request_id)
