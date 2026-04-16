"""Vector Store — Factory for PGVector instances.

Centralizes the creation of ``PGVector`` objects so that ``ingest.py`` and
``search.py`` always use the same backend, package, and configuration.

Design
------
- ``get_vector_store(embeddings)`` is the **primary factory**: it accepts an
  already-constructed :class:`~langchain_core.embeddings.Embeddings` instance
  and returns a configured ``PGVector``.  This makes the function easy to test
  in isolation (inject any embeddings object) and keeps ``ingest.py`` in
  control of its own ``embeddings`` lifecycle.

- ``get_vector_store_from_handler()`` is a **convenience wrapper** that
  obtains the singleton embeddings from :func:`~llm_handler.get_llm_handler`
  and delegates to ``get_vector_store``.  Used by ``search.py``, which has no
  other need to hold a reference to the embeddings object.

Both functions produce a ``PGVector`` configured with:
- ``langchain_postgres`` (modern, psycopg v3 driver) — same package as the
  original ``ingest.py``.
- ``use_jsonb=True`` — required for ``langchain_postgres ≥ 0.0.15``.
- ``connection=settings.database_url`` — psycopg v3 DSN format.
"""

from __future__ import annotations

import logging

from config import settings
from langchain_core.embeddings import Embeddings
from langchain_postgres.vectorstores import PGVector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primary factory
# ---------------------------------------------------------------------------


def get_vector_store(embeddings: Embeddings) -> PGVector:
    """Return a configured ``PGVector`` instance for the given embeddings.

    This is the single source of truth for ``PGVector`` construction in this
    project.  Both ``ingest.py`` and ``search.py`` delegate here, guaranteeing
    they use the same package (``langchain_postgres``), connection format
    (psycopg v3 DSN), and ``use_jsonb`` flag.

    Args:
        embeddings: A LangChain :class:`~langchain_core.embeddings.Embeddings`
            instance, typically obtained from
            :meth:`~llm_handler.LLMHandler.get_embeddings`.

    Returns:
        A ``PGVector`` instance ready for ``add_documents``, ``similarity_search``,
        and related operations.

    Raises:
        Exception: Propagates any connection or configuration error raised by
            ``PGVector`` so callers can handle or log them appropriately.
    """
    logger.debug(
        "Building PGVector store: collection=%r url=%r",
        settings.pg_vector_collection_name,
        settings.database_url[:30] + "...",  # truncate to avoid leaking credentials
    )
    return PGVector(
        embeddings=embeddings,
        collection_name=settings.pg_vector_collection_name,
        connection=settings.database_url,
        use_jsonb=True,
    )


# ---------------------------------------------------------------------------
# Convenience wrapper (used by search.py)
# ---------------------------------------------------------------------------


def get_vector_store_from_handler() -> PGVector:
    """Return a ``PGVector`` instance using the singleton LLM handler's embeddings.

    Convenience wrapper around :func:`get_vector_store` for callers that do not
    need to manage the embeddings object themselves (e.g. ``search.py``).
    Obtains embeddings from the module-level singleton via
    :func:`~llm_handler.get_llm_handler`.

    Returns:
        A ``PGVector`` instance with embeddings from the currently configured
        LLM provider (``LLM_PROVIDER`` env var).

    Raises:
        Exception: Propagates errors from the LLM handler or ``PGVector``
            construction.
    """
    # Import here to avoid circular imports at module load time.
    from llm_handler import get_llm_handler

    embeddings = get_llm_handler().get_embeddings()
    return get_vector_store(embeddings)
