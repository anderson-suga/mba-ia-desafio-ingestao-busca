"""Search Engine — Retrieval layer for the RAG pipeline.

Responsible for:
  1. Connecting to PGVector and running ``similarity_search_with_score(query, k=10)``.
  2. Concatenating the retrieved ``page_content`` chunks into a context string.
  3. Building a ``PromptTemplate`` with ``{contexto}`` partially filled via ``.partial()``.

This module does NOT instantiate the ChatModel, invoke the chain, or format
terminal output — those responsibilities belong to ``chat.py``.
"""

from __future__ import annotations

import logging

from langchain.prompts import PromptTemplate
from vector_store import get_vector_store_from_handler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt Template — identical to the challenge specification.
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def retrieve_context(query: str, k: int = 10) -> str:
    """Retrieve the k most relevant chunks and return the concatenated context.

    Connects to PGVector using the singleton embeddings from ``get_llm_handler()``
    to guarantee embedding consistency with the ingestion step.  Uses
    ``similarity_search_with_score`` as explicitly required by the challenge
    specification.  Scores are logged at DEBUG level and then discarded.

    Args:
        query: The user's question.
        k: Number of results to retrieve (default: 10, as specified by the challenge).

    Returns:
        String with the concatenated ``page_content`` of the retrieved chunks,
        separated by double newlines.  Returns an empty string if no results
        are found.

    Raises:
        Exception: Re-raises any PGVector or embedding provider errors after
            logging them, so ``build_chain`` callers receive a typed failure.
    """
    vector_store = get_vector_store_from_handler()

    logger.debug("Running similarity_search_with_score: query=%r k=%d", query, k)
    results_with_scores = vector_store.similarity_search_with_score(query, k=k)

    if not results_with_scores:
        logger.warning("No chunks retrieved for query: %r", query)
        return ""

    for doc, score in results_with_scores:
        logger.debug("Score=%.4f | chunk preview: %r", score, doc.page_content[:80])

    context_text = "\n\n".join(doc.page_content for doc, _ in results_with_scores)
    logger.info("Retrieved %d chunks for context.", len(results_with_scores))
    return context_text


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def build_chain(context: str) -> PromptTemplate:
    """Build a PromptTemplate partially filled with the retrieved context.

    Compiles ``PROMPT_TEMPLATE`` into a ``PromptTemplate`` and uses
    ``.partial()`` to bind ``{contexto}`` immediately, returning a template
    that only awaits ``{pergunta}`` from the caller (``chat.py``).

    Args:
        context: The retrieved context string from :func:`retrieve_context`.

    Returns:
        ``PromptTemplate`` with ``{contexto}`` already filled, awaiting
        ``{pergunta}``.
    """
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    return prompt.partial(contexto=context)


# ---------------------------------------------------------------------------
# Convenience wrapper (legacy entry-point used by chat.py)
# ---------------------------------------------------------------------------


def search_prompt(question: str | None = None) -> PromptTemplate | None:
    """Retrieve context for *question* and return a partially filled PromptTemplate.

    This is the single entry-point called by ``chat.py``.  It orchestrates
    :func:`retrieve_context` → :func:`build_chain` and returns the partial
    ``PromptTemplate`` ready to be composed into a LCEL chain.

    Args:
        question: The user's question.  If ``None`` or empty, returns ``None``
            so callers can detect the failure path cleanly.

    Returns:
        A ``PromptTemplate`` with ``{contexto}`` filled, or ``None`` on failure.
    """
    if not question:
        logger.error("search_prompt called with empty question.")
        return None

    try:
        context = retrieve_context(question)
        return build_chain(context)
    except Exception:
        logger.exception("Failed to retrieve context or build chain.")
        return None