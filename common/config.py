"""Application settings loaded from .env via pydantic-settings."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Resolve the .env file relative to the project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Centralised configuration for all services."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- API keys ---
    OPENAI_API_KEY: str = ""
    YOUTUBE_API_KEY: str = ""
    PINECONE_API_KEY: str = ""

    # --- Seeded Agent Card discovery URLs ---
    YOUTUBE_AGENT_CARD_URL: str = "http://localhost:5001/.well-known/agent-card.json"
    PRODUCT_AGENT_CARD_URL: str = "http://localhost:5002/.well-known/agent-card.json"

    # --- Optional tuning ---
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_EMBEDDING_DIMENSIONS: int = 1536
    PINECONE_INDEX_NAME: str = "fmas-youtube-transcripts"
    PINECONE_ENVIRONMENT: str = ""
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"
    ENABLE_PINECONE: bool = True
    ENABLE_LEGACY_SHOPPING_FLOW: bool = True
    ENABLE_MOCK_PRODUCT_CATALOG: bool = True
    A2A_CLIENT_TIMEOUT_SECONDS: int = 30
    YOUTUBE_REVIEW_CONCURRENCY: int = 3
    YOUTUBE_TRANSCRIPT_CHUNK_SIZE: int = 900
    YOUTUBE_TRANSCRIPT_CHUNK_OVERLAP: int = 120
    LOG_LEVEL: str = "INFO"

    @model_validator(mode="after")
    def _normalize_pinecone_environment(self) -> "Settings":
        """Support both cloud/region and legacy PINECONE_ENVIRONMENT formats."""
        if self.PINECONE_ENVIRONMENT:
            parts = [part for part in self.PINECONE_ENVIRONMENT.split("-") if part]
            if len(parts) >= 4:
                self.PINECONE_CLOUD = parts[-1]
                self.PINECONE_REGION = "-".join(parts[:-1])
        return self


@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


def configure_logging(settings: Settings | None = None) -> None:
    """Set up basic logging based on the configured LOG_LEVEL."""
    s = settings or get_settings()
    logging.basicConfig(
        level=getattr(logging, s.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
