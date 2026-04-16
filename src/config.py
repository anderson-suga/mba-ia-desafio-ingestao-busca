"""Application settings — single source of truth for environment variables.

All modules must import ``settings`` from here instead of calling
``os.getenv`` or ``load_dotenv`` directly.  ``pydantic-settings`` reads the
``.env`` file at the project root automatically via ``SettingsConfigDict``,
eliminating duplicated ``load_dotenv`` calls and scattered ``os.getenv``
defaults throughout the codebase.

Usage::

    from config import settings

    db_url = settings.database_url
    provider = settings.llm_provider
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve the project root relative to this file's location (src/config.py → root).
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Immutable application settings loaded from the environment and ``.env`` file.

    Field names map directly to environment variables by uppercasing them
    (e.g. ``llm_provider`` → ``LLM_PROVIDER``).  Default values mirror those
    defined in ``.env.example``.
    """

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        # Ignore extra keys present in .env but not declared here.
        extra="ignore",
    )

    # ── LLM Provider ──────────────────────────────────────────────────────────
    llm_provider: str = "google"

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # ── Google Gemini ─────────────────────────────────────────────────────────
    google_api_key: str = ""
    google_chat_model: str = "gemini-2.5-flash-lite"
    google_embedding_model: str = "models/gemini-embedding-001"

    # ── Database (PostgreSQL + pgvector) ──────────────────────────────────────
    database_url: str = ""
    pg_vector_collection_name: str = "pdf_chunks"

    # ── PDF Source ────────────────────────────────────────────────────────────
    pdf_path: str = "document.pdf"


# Module-level singleton — imported by all other modules.
settings: Settings = Settings()
