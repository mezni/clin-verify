"""Layout-aware document ingestion pipeline.

Parses regulatory PDFs into structured Markdown (``pymupdf4llm``), chunks them
with a header-aware splitter, embeds each chunk with FastEmbed (dense + sparse),
and upserts points into Qdrant.

See ``docs/plan.md`` (Phase 1), ``docs/adr/0003-local-onnx-embeddings.md``,
``docs/adr/0005-qdrant-hybrid-store.md``, and
``docs/adr/0006-table-preserving-chunking.md``.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import pymupdf
import pymupdf4llm
from fastembed import SparseTextEmbedding, TextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import models

from clinverify.config import Settings
from clinverify.errors import InvalidDocumentError

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"

# Separator priority: split at table/section headers first so table rows stay
# grouped with their section header (ADR-0006).
SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", " "]

_POINT_NAMESPACE = uuid.UUID("6f6a3f1e-2b0f-4b6a-9d0e-1c2a3b4c5d6e")


@dataclass(frozen=True)
class PageMarkdown:
    """Structured Markdown for a single PDF page."""

    page_num: int
    text: str


@dataclass(frozen=True)
class Chunk:
    """A chunk of document text with provenance metadata."""

    doc_id: str
    chunk_index: int
    page_num: int
    text: str
    citation: str


@dataclass(frozen=True)
class IngestResult:
    """Summary returned by a successful ingestion."""

    doc_id: str
    pages_processed: int
    chunks_indexed: int


class Embedder(Protocol):
    """Produces dense and sparse vectors for a batch of texts."""

    def embed(
        self, texts: Sequence[str]
    ) -> tuple[list[list[float]], list[models.SparseVector]]: ...


class IngestionServiceProtocol(Protocol):
    """The subset of the ingestion service consumed by the API layer."""

    def ingest(self, doc_id: str, data: bytes) -> IngestResult: ...


class VectorStore(Protocol):
    """The subset of the Qdrant client used by the ingestion service."""

    def collection_exists(self, collection_name: str) -> bool: ...

    def create_collection(
        self,
        *,
        collection_name: str,
        vectors_config: Any,
        sparse_vectors_config: Any,
    ) -> Any: ...

    def delete(self, *, collection_name: str, points_selector: Any, wait: bool = True) -> Any: ...

    def upsert(self, *, collection_name: str, points: Any, wait: bool = True) -> Any: ...


def parse_pdf(data: bytes) -> list[PageMarkdown]:
    """Parse raw PDF bytes into per-page structured Markdown.

    Raises:
        InvalidDocumentError: if the bytes are not a readable PDF.
    """

    try:
        document = pymupdf.open(stream=data, filetype="pdf")  # type: ignore[no-untyped-call]
    except Exception as exc:  # noqa: BLE001 - normalize all parse failures
        raise InvalidDocumentError(f"Unable to open PDF: {exc}") from exc

    try:
        if document.needs_pass:
            raise InvalidDocumentError("PDF is encrypted and cannot be read")
        try:
            pages = pymupdf4llm.to_markdown(document, page_chunks=True)
        except Exception as exc:  # noqa: BLE001 - normalize all parse failures
            raise InvalidDocumentError(f"Unable to parse PDF: {exc}") from exc
    finally:
        document.close()  # type: ignore[no-untyped-call]

    parsed: list[PageMarkdown] = []
    for index, page in enumerate(pages, start=1):
        metadata = page.get("metadata", {}) or {}
        page_num = int(metadata.get("page_number") or index)
        text = str(page.get("text", ""))
        parsed.append(PageMarkdown(page_num=page_num, text=text))
    return parsed


def chunk_pages(doc_id: str, pages: Sequence[PageMarkdown], settings: Settings) -> list[Chunk]:
    """Split per-page Markdown into chunks, preserving page provenance."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=SEPARATORS,
    )

    chunks: list[Chunk] = []
    for page in pages:
        for piece in splitter.split_text(page.text):
            text = piece.strip()
            if not text:
                continue
            chunk_index = len(chunks)
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    page_num=page.page_num,
                    text=text,
                    citation=f"[{doc_id}, p.{page.page_num}]",
                )
            )
    return chunks


def point_id(doc_id: str, chunk_index: int) -> str:
    """Deterministic point id enabling idempotent re-ingestion."""

    return str(uuid.uuid5(_POINT_NAMESPACE, f"{doc_id}:{chunk_index}"))


class FastEmbedEmbedder:
    """FastEmbed-backed dense + sparse embedder (local ONNX)."""

    def __init__(self, dense_model: str, sparse_model: str) -> None:
        self._dense = TextEmbedding(model_name=dense_model, lazy_load=True)
        self._sparse = SparseTextEmbedding(model_name=sparse_model, lazy_load=True)

    def embed(self, texts: Sequence[str]) -> tuple[list[list[float]], list[models.SparseVector]]:
        documents = list(texts)
        dense_vectors = [vector.tolist() for vector in self._dense.embed(documents)]
        sparse_vectors = [
            models.SparseVector(
                indices=[int(i) for i in embedding.indices],
                values=[float(v) for v in embedding.values],
            )
            for embedding in self._sparse.embed(documents)
        ]
        return dense_vectors, sparse_vectors


class IngestionService:
    """Parses, chunks, embeds, and indexes documents into Qdrant."""

    def __init__(
        self,
        settings: Settings,
        client: VectorStore,
        embedder: Embedder,
    ) -> None:
        self._settings = settings
        self._client = client
        self._embedder = embedder

    def ensure_collection(self) -> None:
        """Create the hybrid collection if it does not already exist."""

        collection = self._settings.qdrant_collection
        if self._client.collection_exists(collection):
            return
        self._client.create_collection(
            collection_name=collection,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=self._settings.dense_dim,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )

    def _delete_document(self, doc_id: str) -> None:
        self._client.delete(
            collection_name=self._settings.qdrant_collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=doc_id),
                        )
                    ]
                )
            ),
            wait=True,
        )

    def ingest(self, doc_id: str, data: bytes) -> IngestResult:
        """Ingest a PDF for ``doc_id``, replacing any prior version."""

        pages = parse_pdf(data)
        chunks = chunk_pages(doc_id, pages, self._settings)

        self.ensure_collection()
        self._delete_document(doc_id)

        if chunks:
            dense_vectors, sparse_vectors = self._embedder.embed([chunk.text for chunk in chunks])
            points = [
                models.PointStruct(
                    id=point_id(doc_id, chunk.chunk_index),
                    vector={
                        DENSE_VECTOR_NAME: dense_vectors[i],
                        SPARSE_VECTOR_NAME: sparse_vectors[i],
                    },
                    payload={
                        "doc_id": doc_id,
                        "page_num": chunk.page_num,
                        "citation": chunk.citation,
                        "text": chunk.text,
                    },
                )
                for i, chunk in enumerate(chunks)
            ]
            self._client.upsert(
                collection_name=self._settings.qdrant_collection,
                points=points,
                wait=True,
            )

        return IngestResult(
            doc_id=doc_id,
            pages_processed=len(pages),
            chunks_indexed=len(chunks),
        )
