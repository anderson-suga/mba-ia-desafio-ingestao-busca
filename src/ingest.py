"""PDF Ingestion Pipeline — loads, chunks, and stores a PDF into PGVector.

Reads a single PDF file, splits it into fixed-size chunks, generates
embeddings via the configured provider (see ``llm_handler.py``), and
persists them in a PostgreSQL/pgvector table managed by LangChain.

Chunk parameters
----------------
- ``chunk_size=1000`` characters: keeps each chunk well within the context
  window of any embedding model (OpenAI ada-002 or Gemini embedding-001),
  preventing token-limit stress and ensuring each chunk carries a coherent
  semantic unit of text.
- ``chunk_overlap=150`` characters: preserves sentence boundary context
  across chunk edges, reducing the risk of splitting mid-sentence and
  losing retrieval-relevant context.

Usage (from project root)::

    python src/ingest.py

Environment variables (read from ``.env`` at project root):
    DATABASE_URL              postgresql+psycopg://... (psycopg v3 driver)
    PG_VECTOR_COLLECTION_NAME Name of the pgvector collection (default: pdf_chunks)
    PDF_PATH                  Path to the PDF, relative to project root
    LLM_PROVIDER              "openai" or "google"
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.exc import OperationalError as SAOperationalError
from tqdm import tqdm

from config import settings
from llm_handler import get_llm_handler
from vector_store import get_vector_store

# ---------------------------------------------------------------------------
# Path — project root for resolving relative PDF paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 150


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_pdf_path(raw_path: str) -> Path:
    """Return the absolute path to the PDF, resolving relative paths from the project root.

    The ``PDF_PATH`` env var is expected to be relative to the project root
    (e.g. ``document.pdf``).  Resolving relative to ``src/`` would produce a
    ``FileNotFoundError`` when the script is executed from the root.

    Args:
        raw_path: The raw value from ``PDF_PATH`` in ``.env``.

    Returns:
        An absolute :class:`~pathlib.Path` pointing to the PDF.

    Raises:
        FileNotFoundError: If the resolved path does not exist.
    """
    p = Path(raw_path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(
            f"PDF not found at '{p}'. "
            "Make sure PDF_PATH in .env is relative to the project root "
            "(e.g. PDF_PATH=document.pdf)."
        )
    return p


def _sanitize_text(text: str) -> str:
    """Normalize unicode and strip control characters from extracted PDF text.

    Brazilian PDFs often contain NFC-incompatible surrogates, mixed encodings,
    or control characters that corrupt embedding inputs or cause silent failures
    during vectorization.

    Args:
        text: Raw text as extracted by :class:`PyPDFLoader`.

    Returns:
        Sanitized text safe for embedding model consumption.
    """
    # Filter control characters (category C*) and isolated surrogates (Cs)
    # BEFORE normalize: unicodedata.normalize does not handle isolated
    # surrogates (U+D800–U+DFFF) safely; filtering first guarantees a
    # surrogate-free string for NFC composition.
    cleaned = "".join(
        ch
        for ch in text
        if not unicodedata.category(ch).startswith("C")
        or ch in ("\t", "\n", "\r")
    )
    # Apply NFC normalization after the string is guaranteed surrogate-free.
    return unicodedata.normalize("NFC", cleaned)


# ---------------------------------------------------------------------------
# Main ingestion function
# ---------------------------------------------------------------------------


def ingest_pdf() -> None:
    """Execute the full PDF ingestion pipeline.

    Steps:
        1. Validate PDF path and database URL from environment.
        2. Load PDF pages with :class:`PyPDFLoader`.
        3. Sanitize text encoding for each page.
        4. Split pages into overlapping chunks.
        5. Clear existing embeddings in the collection to avoid duplication.
        6. Insert chunks into PGVector via batch upsert.

    Raises:
        FileNotFoundError: If the PDF path does not exist.
        EnvironmentError: If ``DATABASE_URL`` is not set.
        Exception: Re-raises any connection or embedding errors with context.
    """
    # ------------------------------------------------------------------
    # 1. Validate environment
    # ------------------------------------------------------------------
    pdf_path_raw = settings.pdf_path
    database_url = settings.database_url.strip()
    collection_name = settings.pg_vector_collection_name

    if not database_url:
        raise EnvironmentError(
            "DATABASE_URL is not set. "
            "Please add it to .env in the format: "
            "postgresql+psycopg://user:pass@host:port/dbname"
        )

    pdf_path = _resolve_pdf_path(pdf_path_raw)

    # ------------------------------------------------------------------
    # 2. Load PDF
    # ------------------------------------------------------------------
    print(f"📄 Loading PDF: {pdf_path}...")
    try:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
    except Exception as exc:
        raise RuntimeError(f"Failed to load PDF '{pdf_path}': {exc}") from exc

    if not pages:
        raise ValueError(f"PDF '{pdf_path}' produced no pages. File may be empty or corrupted.")

    # ------------------------------------------------------------------
    # 3. Sanitize text encoding per page
    # ------------------------------------------------------------------
    for doc in pages:
        doc.page_content = _sanitize_text(doc.page_content)

    # ------------------------------------------------------------------
    # 4. Split into chunks
    # ------------------------------------------------------------------
    # chunk_size=1000 keeps each chunk within token limits of all supported
    # embedding models (OpenAI ada-002: 8191 tokens ≈ ~6000 chars;
    # Gemini embedding-001: 2048 tokens ≈ ~1500 chars).
    # chunk_overlap=150 preserves cross-boundary sentence context.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)
    print(f"✂️  Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})... {len(chunks)} chunks created")

    # ------------------------------------------------------------------
    # 5. Build embeddings and connect to PGVector
    # ------------------------------------------------------------------
    handler = get_llm_handler()
    embeddings = handler.get_embeddings()

    print(f"🗄️  Inserting {len(chunks)} chunks into the database (collection: '{collection_name}')...")

    try:
        # Initialize PGVector store via the shared factory (vector_store.py).
        # Configuration (connection, collection, use_jsonb) is centralised there.
        vector_store = get_vector_store(embeddings)

        # ------------------------------------------------------------------
        # 6. Clear existing collection to prevent duplicate embeddings.
        #    If ingest.py runs twice, without a reset the retriever would
        #    return repeated results for the same question, degrading RAG quality.
        # ------------------------------------------------------------------
        vector_store.delete_collection()
        vector_store.create_collection()

        # Insert in batch using tqdm for live progress feedback.
        batch_size = 50
        for i in tqdm(range(0, len(chunks), batch_size), desc="Uploading batches", unit="batch"):
            batch = chunks[i : i + batch_size]
            vector_store.add_documents(batch)

    except Exception as exc:
        # PGVector (langchain_postgres) may wrap sqlalchemy.exc.OperationalError
        # in a plain Exception (see vectorstores.py:509). Walk the __cause__ chain
        # to detect PostgreSQL connection failures robustly, without relying on
        # locale-specific message strings (G-03).
        cause: BaseException | None = exc
        while cause is not None:
            if isinstance(cause, SAOperationalError):
                raise ConnectionError(
                    "Could not connect to PostgreSQL. "
                    "Make sure the database is running (docker-compose up -d) "
                    f"and DATABASE_URL is correct: {database_url}"
                ) from exc
            cause = cause.__cause__
        raise

    print("✅ Ingestion completed successfully!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ingest_pdf()