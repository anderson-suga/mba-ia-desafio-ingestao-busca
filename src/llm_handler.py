"""LLM Handler — Factory for provider-aware ChatModel and Embeddings.

Centralizes the instantiation of LangChain ChatModels and Embedding drivers
based on the ``LLM_PROVIDER`` environment variable.  Every module that needs
an LLM or embeddings client should obtain it from here, avoiding duplicated
credential handling and provider-specific setup logic.

Supported providers:
  - ``openai``  → ChatOpenAI + OpenAIEmbeddings
  - ``google``  → ChatGoogleGenerativeAI + GoogleGenerativeAIEmbeddings

⚠️  If the embedding provider is changed *without* re-running ingestion the
vector dimensions will be incompatible and search results will be incorrect.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator

from config import settings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LLMHandlerError(Exception):
    """Base exception for LLM handler errors."""


class ProviderNotSupportedError(LLMHandlerError):
    """Raised when an unsupported LLM provider is configured."""


class MissingAPIKeyError(LLMHandlerError):
    """Raised when the required API key for the selected provider is not set."""


class AuthenticationError(LLMHandlerError):
    """Raised when the provider rejects the API key at runtime."""


class RateLimitError(LLMHandlerError):
    """Raised when the provider rate-limits the current API key."""


class ProviderConnectionError(LLMHandlerError):
    """Raised when a network or connectivity error prevents reaching the provider."""

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """Immutable configuration for a single LLM provider.

    Attributes:
        provider: Provider identifier (``"openai"`` or ``"google"``).
        api_key: The API key for the provider.
        chat_model: Model name used for chat completions.
        embedding_model: Model name used for embeddings.
    """

    provider: str
    api_key: str = field(repr=False)  # Never include the key in repr/logs.
    chat_model: str
    embedding_model: str


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LLMHandler:
    """Factory that creates ChatModel and Embeddings instances.

    Reads the ``LLM_PROVIDER`` environment variable and lazily instantiates
    the corresponding LangChain objects.  API keys are validated at
    construction time (fail-fast).

    Usage::

        handler = LLMHandler()
        model = handler.get_chat_model()
        embeddings = handler.get_embeddings()

    Raises:
        ProviderNotSupportedError: If ``LLM_PROVIDER`` is not ``openai`` or ``google``.
        MissingAPIKeyError: If the required API key is absent or empty.
    """

    _chat_model: BaseChatModel | None = field(default=None, init=False, repr=False)
    _embeddings: Embeddings | None = field(default=None, init=False, repr=False)
    _config: ProviderConfig | None = field(default=None, init=False, repr=False)

    # ---- Initialisation ----------------------------------------------------

    def __post_init__(self) -> None:
        """Validate configuration eagerly.

        Called automatically by the dataclass machinery right after
        ``__init__``.  Configuration is read from the module-level ``settings``
        object (``src/config.py``), which handles ``.env`` loading via
        ``pydantic-settings``.

        Raises:
            ProviderNotSupportedError: If ``LLM_PROVIDER`` is not ``openai`` or ``google``.
            MissingAPIKeyError: If the required API key is absent or empty.
        """
        # Validate and cache the configuration immediately (fail-fast).
        self._config = self._build_config()

    # ---- Configuration -----------------------------------------------------

    def _build_config(self) -> ProviderConfig:
        """Assemble a :class:`ProviderConfig` from ``config.settings``.

        Returns:
            A fully populated configuration object.

        Raises:
            ProviderNotSupportedError: If the provider string is unknown.
            MissingAPIKeyError: If the required API key is missing.
        """
        provider = settings.llm_provider.strip().lower()

        match provider:
            case "openai":
                api_key = settings.openai_api_key.strip()
                if not api_key:
                    raise MissingAPIKeyError(
                        "OPENAI_API_KEY is not set. "
                        "Please add it to your .env file."
                    )
                chat_model = settings.openai_chat_model
                embedding_model = settings.openai_embedding_model
            case "google":
                api_key = settings.google_api_key.strip()
                if not api_key:
                    raise MissingAPIKeyError(
                        "GOOGLE_API_KEY is not set. "
                        "Please add it to your .env file."
                    )
                chat_model = settings.google_chat_model
                embedding_model = settings.google_embedding_model
            case _:
                raise ProviderNotSupportedError(
                    f"Unsupported LLM_PROVIDER: {provider!r}. "
                    f"Supported values: 'openai', 'google'."
                )

        return ProviderConfig(
            provider=provider,
            api_key=api_key,
            chat_model=chat_model,
            embedding_model=embedding_model,
        )

    @property
    def config(self) -> ProviderConfig:
        """Provider configuration, validated and cached at construction time."""
        # _config is guaranteed non-None after __post_init__.
        return self._config  # type: ignore[return-value]

    # ---- Runtime error handling --------------------------------------------

    @contextmanager
    def _wrap_api_errors(self, provider: str) -> Generator[None, None, None]:
        """Context manager that translates provider SDK exceptions into typed errors.

        Args:
            provider: The provider identifier (``"openai"`` or ``"google"``),
                used to build informative error messages.

        Yields:
            None — wraps the body of the ``with`` block.

        Raises:
            AuthenticationError: If the provider rejects the API key.
            RateLimitError: If the provider rate-limits the request.
            ProviderConnectionError: If a network error occurs.
        """
        try:
            yield
        except Exception as exc:  # noqa: BLE001
            exc_type = type(exc).__name__
            exc_module = type(exc).__module__

            # Map well-known SDK exception names to our typed errors.
            if "AuthenticationError" in exc_type or "InvalidApiKey" in exc_type:
                raise AuthenticationError(
                    f"[{provider}] API key was rejected by the provider. "
                    "Check the key in your .env file."
                ) from exc
            if "RateLimitError" in exc_type or "ResourceExhausted" in exc_type:
                raise RateLimitError(
                    f"[{provider}] Rate limit reached. Wait and retry."
                ) from exc
            if any(t in exc_type for t in ("ConnectionError", "ConnectError", "ServiceUnavailable")):
                raise ProviderConnectionError(
                    f"[{provider}] Could not connect to the provider API. "
                    "Check your internet connection."
                ) from exc
            # Unknown error — re-raise as-is so the caller sees the original.
            raise

    # ---- Chat Model --------------------------------------------------------

    def get_chat_model(self) -> BaseChatModel:
        """Return a ChatModel instance for the configured provider.

        The model is created with ``temperature=0`` to ensure deterministic,
        factual responses in a RAG pipeline.

        Returns:
            A :class:`BaseChatModel` instance (``ChatOpenAI`` or
            ``ChatGoogleGenerativeAI``).

        Raises:
            ProviderNotSupportedError: If the provider is unknown.
            MissingAPIKeyError: If the API key is missing.
        """
        if self._chat_model is not None:
            return self._chat_model

        cfg = self.config
        logger.info("Initializing ChatModel for provider=%s model=%s", cfg.provider, cfg.chat_model)

        with self._wrap_api_errors(cfg.provider):
            match cfg.provider:
                case "openai":
                    from langchain_openai import ChatOpenAI

                    self._chat_model = ChatOpenAI(
                        model=cfg.chat_model,
                        api_key=cfg.api_key,
                        temperature=0,
                    )
                case "google":
                    from langchain_google_genai import ChatGoogleGenerativeAI

                    self._chat_model = ChatGoogleGenerativeAI(
                        model=cfg.chat_model,
                        google_api_key=cfg.api_key,
                        temperature=0,
                    )
                case _:
                    raise ProviderNotSupportedError(
                        f"Unsupported provider: {cfg.provider!r}"
                    )

        return self._chat_model  # type: ignore[return-value]

    # ---- Embeddings --------------------------------------------------------

    def get_embeddings(self) -> Embeddings:
        """Return an Embeddings instance for the configured provider.

        Uses the same provider as the ChatModel to ensure embedding
        dimensionality is consistent between ingestion and retrieval.

        Returns:
            An :class:`Embeddings` instance (``OpenAIEmbeddings`` or
            ``GoogleGenerativeAIEmbeddings``).

        Raises:
            ProviderNotSupportedError: If the provider is unknown.
            MissingAPIKeyError: If the API key is missing.
        """
        if self._embeddings is not None:
            return self._embeddings

        cfg = self.config
        logger.info(
            "Initializing Embeddings for provider=%s model=%s",
            cfg.provider,
            cfg.embedding_model,
        )

        with self._wrap_api_errors(cfg.provider):
            match cfg.provider:
                case "openai":
                    from langchain_openai import OpenAIEmbeddings

                    self._embeddings = OpenAIEmbeddings(
                        model=cfg.embedding_model,
                        api_key=cfg.api_key,
                    )
                case "google":
                    from langchain_google_genai import GoogleGenerativeAIEmbeddings

                    self._embeddings = GoogleGenerativeAIEmbeddings(
                        model=cfg.embedding_model,
                        google_api_key=cfg.api_key,
                    )
                case _:
                    raise ProviderNotSupportedError(
                        f"Unsupported provider: {cfg.provider!r}"
                    )

        return self._embeddings  # type: ignore[return-value]

    # ---- Helpers -----------------------------------------------------------

    @property
    def provider_name(self) -> str:
        """Return the current provider identifier (e.g. ``"openai"``)."""
        return self.config.provider


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_handler: LLMHandler | None = None


def get_llm_handler() -> LLMHandler:
    """Return a module-level singleton :class:`LLMHandler`.

    Useful for scripts that only need one handler throughout their lifetime
    (e.g. ``ingest.py`` and ``chat.py``).

    Returns:
        The shared :class:`LLMHandler` instance.

    Note:
        Not thread-safe. Suitable for single-threaded CLI usage only.
        Do not use in async servers without adding a lock.
    """
    global _default_handler
    if _default_handler is None:
        _default_handler = LLMHandler()
    return _default_handler
