"""Application configuration loaded from the environment.

Field names map to upper-case environment variables (e.g. ``openai_api_key``
reads ``OPENAI_API_KEY``). Values are defined in ``.env.example``.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

DOC_ID_PATTERN = r"^[A-Za-z0-9_\-]{3,128}$"


class Settings(BaseSettings):
    """Runtime settings for the ClinVerify service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str | None = None
    generator_model: str = "gpt-4o-mini"
    generator_temperature: float = 0.0

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "clinical_trials_hybrid"

    dense_model: str = "BAAI/bge-small-en-v1.5"
    dense_dim: int = 384
    sparse_model: str = "Qdrant/bm25"
    reranker_model: str = "BAAI/bge-reranker-base"

    prefetch_k: int = 20
    rerank_top_k: int = 5
    rrf_k: int = 60

    chunk_size: int = 600
    chunk_overlap: int = 80
    max_upload_mb: int = 50


@lru_cache
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance."""

    return Settings()
