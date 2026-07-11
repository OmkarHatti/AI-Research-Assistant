"""
src/utils.py
────────────
General-purpose utility functions used across the project.

Covers:
* Saving Streamlit ``UploadedFile`` objects to disk.
* Formatting retrieved source documents into display-ready labels.
* Clearing a directory without removing the directory itself.
* Safe JSON serialisation for logging.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.logger import get_logger

logger = get_logger(__name__)


# ── File I/O ──────────────────────────────────────────────────────────────────


def save_uploaded_file(uploaded_file: Any, dest_dir: str | Path) -> str:
    """
    Write a Streamlit ``UploadedFile`` to ``dest_dir`` and return the path.

    Parameters
    ----------
    uploaded_file:
        The ``st.UploadedFile`` object from ``st.file_uploader``.
    dest_dir:
        Target directory.  Created if it does not exist.

    Returns
    -------
    str
        Absolute path to the saved file.

    Raises
    ------
    IOError
        If writing fails.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    save_path = dest / uploaded_file.name
    try:
        with open(save_path, "wb") as fh:
            fh.write(uploaded_file.getbuffer())
        logger.info("Saved uploaded file → %s (%d bytes)", save_path, save_path.stat().st_size)
    except Exception as exc:
        logger.error("Failed to save %s: %s", uploaded_file.name, exc, exc_info=True)
        raise IOError(f"Could not save '{uploaded_file.name}': {exc}") from exc

    return str(save_path)


def clear_directory(directory: str | Path, *, keep_directory: bool = True) -> None:
    """
    Remove all contents of ``directory``.

    Parameters
    ----------
    directory:
        Path to the directory to clear.
    keep_directory:
        When ``True`` (default), the directory itself is preserved.
        When ``False``, the directory is also removed.

    Raises
    ------
    FileNotFoundError
        If ``directory`` does not exist.
    """
    path = Path(directory)
    if not path.exists():
        logger.warning("clear_directory: '%s' does not exist.", path)
        return

    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    if not keep_directory:
        path.rmdir()

    logger.info("Cleared directory: %s", path)


# ── Source formatting ─────────────────────────────────────────────────────────


def format_sources(documents: list[Document]) -> list[str]:
    """
    Convert retrieved ``Document`` objects into human-readable source labels.

    Each label has the format::

        filename.pdf (p. 3, chunk 1)

    Duplicate labels are deduplicated while preserving order.

    Parameters
    ----------
    documents:
        List of retrieved ``Document`` objects.

    Returns
    -------
    list[str]
        Ordered, deduplicated source label strings.
    """
    seen: set[str] = set()
    labels: list[str] = []

    for doc in documents:
        raw_source: str = doc.metadata.get("source", "Unknown")
        file_name = Path(raw_source).name if raw_source != "Unknown" else raw_source

        # Optional page / chunk decorations
        parts: list[str] = []
        page = doc.metadata.get("page")
        if page is not None:
            parts.append(f"p. {int(page) + 1}")  # convert 0-index → 1-index

        chunk = doc.metadata.get("chunk_index")
        if chunk is not None:
            parts.append(f"chunk {int(chunk)}")

        label = f"{file_name} ({', '.join(parts)})" if parts else file_name

        if label not in seen:
            seen.add(label)
            labels.append(label)

    return labels


# ── JSON helpers ──────────────────────────────────────────────────────────────


def safe_json(obj: Any, indent: int = 2) -> str:
    """
    Serialise ``obj`` to a JSON string, falling back to ``repr()`` for
    types that are not JSON-serialisable.

    Parameters
    ----------
    obj:
        Any Python object.
    indent:
        JSON pretty-print indentation level.

    Returns
    -------
    str
        JSON string representation, or a best-effort ``repr()`` string.
    """
    try:
        return json.dumps(obj, indent=indent, default=str, ensure_ascii=False)
    except Exception:
        return repr(obj)


# ── Text helpers ──────────────────────────────────────────────────────────────


def truncate(text: str, max_chars: int = 200, suffix: str = "…") -> str:
    """
    Truncate ``text`` to ``max_chars`` characters, appending ``suffix``.

    Parameters
    ----------
    text:
        Input string.
    max_chars:
        Maximum character count (including the suffix).
    suffix:
        Ellipsis-like string appended when the text is truncated.

    Returns
    -------
    str
        Possibly truncated string.
    """
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(suffix)] + suffix
