"""Chat — Interactive RAG loop for the document Q&A pipeline.

Orchestrates the retrieval-augmented generation flow:
  1. Calls ``search_prompt(question)`` to retrieve context and get a partial PromptTemplate.
  2. Builds a LCEL chain: ``prompt_partial | model | StrOutputParser()``.
  3. Invokes the chain with ``{pergunta}`` and prints the answer.

User-facing labels (``PERGUNTA:`` / ``RESPOSTA:``) remain in Portuguese as
required by the challenge specification.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from llm_handler import get_llm_handler
from search import search_prompt

# ---------------------------------------------------------------------------
# Session start timestamp — shared by both log and history filenames so they
# are always paired and easy to correlate.
# ---------------------------------------------------------------------------

_SESSION_START: datetime = datetime.now()
_SESSION_TAG: str = _SESSION_START.strftime("%Y-%m-%d_%H-%M-%S")

# ---------------------------------------------------------------------------
# Logs directory — <project_root>/logs/  (created on first run if absent)
# ---------------------------------------------------------------------------

_LOGS_DIR: Path = Path(__file__).resolve().parent.parent / "logs"
_LOGS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Technical logging → file only (INFO+).  Console stays clean for the user.
# ---------------------------------------------------------------------------

_log_file: Path = _LOGS_DIR / f"app_{_SESSION_TAG}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler(_log_file, encoding="utf-8")],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chat history → JSONL file (one JSON object per Q&A pair).
# ---------------------------------------------------------------------------

_history_file: Path = _LOGS_DIR / f"chat_history_{_SESSION_TAG}.jsonl"

# ---------------------------------------------------------------------------
# ANSI color codes — terminal-only formatting, no external dependency.
# Works natively on Linux/macOS; the project runs in Docker (Linux).
# ---------------------------------------------------------------------------

_CYAN  = "\033[96m"
_GREEN = "\033[92m"
_BOLD  = "\033[1m"
_RESET = "\033[0m"
_SEP   = "-" * 60


def main() -> None:
    """Run the interactive Q&A loop.

    Reads questions from stdin (one per iteration), retrieves context, and
    prints the LLM-generated answer.  Exits cleanly on EOF (Ctrl-D) or
    empty input.
    """
    print("Sistema de busca RAG inicializado. Digite 'sair' para encerrar.\n")

    handler = get_llm_handler()
    model = handler.get_chat_model()
    output_parser = StrOutputParser()

    while True:
        try:
            question = input(f"{_BOLD}{_CYAN}PERGUNTA:{_RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando.")
            sys.exit(0)

        if question.lower() in {"sair", "exit", "quit"}:
            print("Encerrando.")
            sys.exit(0)

        if not question:
            continue

        prompt_partial = search_prompt(question)

        if prompt_partial is None:
            print("Não foi possível iniciar o chat. Verifique os erros de inicialização.\n")
            continue

        # LCEL chain: partial prompt (awaiting {pergunta}) | model | parser
        chain = prompt_partial | model | output_parser

        try:
            answer = chain.invoke({"pergunta": question})
        except Exception:
            logger.exception("Chain invocation failed.")
            print("Ocorreu um erro ao processar sua pergunta. Tente novamente.\n")
            continue

        print(f"\n{_BOLD}{_GREEN}RESPOSTA:{_RESET} {answer}")
        print(f"{_SEP}\n")

        # Append Q&A pair to the JSONL history file.
        record = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
        }
        with _history_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()