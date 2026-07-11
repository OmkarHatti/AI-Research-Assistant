"""
src/loaders.py
──────────────
Document loading layer.

Supports:
    .pdf   → PyMuPDFLoader (fast, preserves structure)
    .docx  → Docx2txtLoader
    .txt   → TextLoader
    .md    → UnstructuredMarkdownLoader

All loaders return a list of ``langchain_core.documents.Document`` objects
with ``page_content`` (text) and ``metadata`` (source, page, etc.).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document

from src.logger import get_logger

logger = get_logger(__name__)





_LOADER_MAP: dict[str, Callable[[str], object]] = {
    ".pdf":  PyMuPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt":  TextLoader,
    ".md":   UnstructuredMarkdownLoader,
}

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(_LOADER_MAP.keys())





class DocumentLoader:
    """
    Unified document loader that dispatches to the correct backend based on
    the file extension.

    Usage
    -----
    >>> loader = DocumentLoader()
    >>> docs = loader.load("/path/to/report.pdf")
    """

    def load(self, file_path: str | Path) -> list[Document]:
        """
        Load a single document from disk.

        Parameters
        ----------
        file_path:
            Absolute or relative path to the file.

        Returns
        -------
        list[Document]
            Non-empty list of LangChain Document objects.

        Raises
        ------
        ValueError
            If the file extension is not supported.
        FileNotFoundError
            If the file does not exist on disk.
        RuntimeError
            If the underlying loader raises an unexpected exception.
        """
        path = Path(file_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        extension = path.suffix.lower()
        if extension not in _LOADER_MAP:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(
                f"Unsupported file type '{extension}'. "
                f"Supported types: {supported}"
            )

        logger.info("Loading document: %s", path.name)

        loader_cls = _LOADER_MAP[extension]
        try:
            loader = loader_cls(str(path))  
            docs: list[Document] = loader.load()
            print(f"Loaded {len(docs)} pages")

            print("First page metadata:", docs[0].metadata)
            print("Last page metadata:", docs[-1].metadata)
        except Exception as exc:
            logger.error("Failed to load %s: %s", path.name, exc, exc_info=True)
            raise RuntimeError(f"Could not load '{path.name}': {exc}") from exc

        if not docs:
            logger.warning("Loader returned 0 documents for %s", path.name)
        else:
            logger.info(
                "Loaded %d page(s) / section(s) from %s", len(docs), path.name
            )

        
        for doc in docs:
            doc.metadata.setdefault("source", str(path))

        return docs

    def load_many(
        self,
        file_paths: list[str | Path],
        *,
        skip_errors: bool = True,
    ) -> list[Document]:
        """
        Load multiple documents, optionally skipping files that fail.

        Parameters
        ----------
        file_paths:
            Iterable of file paths to load.
        skip_errors:
            When ``True`` (default), log errors and continue.  When
            ``False``, re-raise the first exception encountered.

        Returns
        -------
        list[Document]
            Concatenated list of documents from all successfully loaded files.
        """
        all_docs: list[Document] = []
        for path in file_paths:
            try:
                all_docs.extend(self.load(path))
            except Exception as exc:
                if skip_errors:
                    logger.warning("Skipping %s — %s", path, exc)
                else:
                    raise
        return all_docs
