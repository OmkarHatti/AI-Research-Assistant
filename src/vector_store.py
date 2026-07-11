"""
src/vector_store.py
───────────────────
ChromaDB-backed vector store.

Wraps ``langchain_chroma.Chroma`` to provide:
* Persistent storage on disk.
* Deduplication: documents are keyed by a SHA-256 hash of their content,
  so re-indexing the same file twice does not create duplicates.
* A ``clear()`` method to wipe the collection.
* Collection-level stats for the UI.
"""

from __future__ import annotations

import hashlib
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.embeddings import EmbeddingModel
from src.logger import get_logger

logger = get_logger(__name__)


def _stable_id(text: str, metadata: dict[str, Any]) -> str:
    """
    Derive a stable, deterministic document ID from its content + metadata.

    Using a content hash means that re-indexing an identical chunk is
    idempotent — ChromaDB will silently overwrite the record with the same
    data rather than creating a duplicate.
    """
    source: str = str(metadata.get("source", ""))
    chunk_index: str = str(metadata.get("chunk_index", ""))
    raw = f"{source}::{chunk_index}::{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class VectorStore:
    """
    Persistent ChromaDB vector store.

    Parameters
    ----------
    embedding_model:
        ``EmbeddingModel`` instance used to compute vectors.
    persist_directory:
        Filesystem path where ChromaDB stores its SQLite + binary data.
    collection_name:
        Name of the ChromaDB collection (analogous to a table).

    Usage
    -----
    >>> store = VectorStore(embedding_model, "./chroma_db")
    >>> store.add_documents(chunks)
    >>> results = store.similarity_search("What is RAG?", k=4)
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        persist_directory: str = "./chroma_db",
        collection_name: str = "research_assistant",
    ) -> None:
        self._embedding_model = embedding_model
        self._persist_directory = persist_directory
        self._collection_name = collection_name

        logger.info(
            "Opening ChromaDB collection '%s' at '%s'",
            collection_name,
            persist_directory,
        )

        try:
            self._chroma = Chroma(
                collection_name=collection_name,
                embedding_function=embedding_model.langchain_embeddings,
                persist_directory=persist_directory,
            )
        except Exception as exc:
            logger.error("Failed to open ChromaDB: %s", exc, exc_info=True)
            raise RuntimeError(f"Could not open ChromaDB: {exc}") from exc

        logger.info(
            "ChromaDB ready. Current document count: %d",
            self.document_count,
        )

    # ── Core operations ───────────────────────────────────────────────────────

    def add_documents(self, documents: list[Document]) -> None:
        """
        Embed and upsert a list of document chunks into the vector store.

        Uses content-based IDs to prevent duplicate entries when the same
        file is indexed more than once.

        Parameters
        ----------
        documents:
            Chunked ``Document`` objects to index.

        Raises
        ------
        ValueError
            If ``documents`` is empty.
        RuntimeError
            If ChromaDB raises an unexpected exception.
        """
        if not documents:
            logger.warning("add_documents() called with empty list — skipping.")
            return

        ids = [_stable_id(doc.page_content, doc.metadata) for doc in documents]

        logger.info("Upserting %d chunks into ChromaDB…", len(documents))
        try:
            self._chroma.add_documents(documents=documents, ids=ids)
        except Exception as exc:
            logger.error("Failed to add documents: %s", exc, exc_info=True)
            raise RuntimeError(f"Failed to upsert documents: {exc}") from exc

        logger.info(
            "Upsert complete. Collection now contains %d document(s).",
            self.document_count,
        )

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Retrieve the top-``k`` most semantically similar documents.

        Parameters
        ----------
        query:
            Natural-language search query.
        k:
            Number of results to return.
        filter_metadata:
            Optional ChromaDB ``where`` filter dict, e.g.
            ``{"source": "report.pdf"}``.

        Returns
        -------
        list[Document]
            Documents sorted by descending similarity to the query.

        Raises
        ------
        RuntimeError
            If the underlying similarity search fails.
        """
        if not query.strip():
            raise ValueError("Query must not be empty.")
        if self.document_count == 0:
            logger.warning("similarity_search called on an empty vector store.")
            return []

        logger.debug("Searching for top-%d results matching: %r", k, query[:80])
        try:
            results: list[Document] = self._chroma.similarity_search(
                query=query,
                k=k,
                filter=filter_metadata,
            )
        except Exception as exc:
            logger.error("Similarity search failed: %s", exc, exc_info=True)
            raise RuntimeError(f"Similarity search failed: {exc}") from exc

        logger.debug("Retrieved %d document(s).", len(results))
        return results

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
    ) -> list[tuple[Document, float]]:
        """
        Retrieve top-``k`` documents along with their similarity scores.

        Returns
        -------
        list[tuple[Document, float]]
            Pairs of (document, cosine-distance score).  Lower score = more
            similar when using L2; higher = more similar for cosine.
        """
        if not query.strip():
            raise ValueError("Query must not be empty.")
        if self.document_count == 0:
            return []

        try:
            return self._chroma.similarity_search_with_score(query=query, k=k)
        except Exception as exc:
            logger.error(
                "similarity_search_with_score failed: %s", exc, exc_info=True
            )
            raise RuntimeError(f"Scored similarity search failed: {exc}") from exc

    def clear(self) -> None:
        """
        Delete all documents from the collection.

        The collection itself is preserved (schema stays intact); only the
        indexed chunks are removed.
        """
        logger.warning("Clearing all documents from collection '%s'…", self._collection_name)
        try:
            self._chroma.delete_collection()
            # Re-create the (now empty) collection
            self._chroma = Chroma(
                collection_name=self._collection_name,
                embedding_function=self._embedding_model.langchain_embeddings,
                persist_directory=self._persist_directory,
            )
            logger.info("Collection cleared and re-created.")
        except Exception as exc:
            logger.error("Failed to clear collection: %s", exc, exc_info=True)
            raise RuntimeError(f"Failed to clear vector store: {exc}") from exc

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def document_count(self) -> int:
        """Number of chunks currently indexed in the collection."""
        try:
            return self._chroma._collection.count()
        except Exception:
            return 0

    @property
    def langchain_chroma(self) -> Chroma:
        """
        Expose the raw LangChain ``Chroma`` instance.

        Use this when constructing a LangChain retriever directly from the
        vector store (see ``Retriever``).
        """
        return self._chroma
