"""
src/retriever.py
────────────────
Retrieval layer — wraps the VectorStore into a LangChain ``BaseRetriever``
and exposes a simple ``retrieve()`` convenience method.

The ``Retriever`` class:
* Delegates semantic search to ``VectorStore.similarity_search``.
* Optionally applies a relevance-score threshold to filter low-quality hits.
* Returns rich metadata alongside each retrieved chunk so the UI can display
  source citations.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from src.vector_store import VectorStore
from src.logger import get_logger

logger = get_logger(__name__)


class Retriever(BaseRetriever):
    """
    Semantic retriever backed by ``VectorStore``.

    Implements LangChain's ``BaseRetriever`` interface so it can be dropped
    into any LangChain chain or expression.

    Parameters
    ----------
    vector_store:
        Initialised ``VectorStore`` instance.
    k:
        Number of documents to retrieve per query.
    score_threshold:
        When set (0.0–1.0 for cosine distance), documents whose similarity
        score exceeds this threshold are filtered out.  Set to ``None`` to
        disable filtering.

    Usage
    -----
    >>> retriever = Retriever(vector_store=store, k=5)
    >>> docs = retriever.retrieve("What is RAG?")
    """

    # ── Pydantic fields (BaseRetriever is a Pydantic model) ───────────────────
    vector_store: VectorStore
    k: int = 5
    score_threshold: float | None = None

    model_config = {"arbitrary_types_allowed": True}

    # ── BaseRetriever protocol ─────────────────────────────────────────────────

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """
        Core retrieval method called by LangChain internals.

        Parameters
        ----------
        query:
            User's natural-language question.
        run_manager:
            LangChain callback hook (unused but required by the interface).

        Returns
        -------
        list[Document]
            Retrieved and optionally filtered document chunks.
        """
        return self.retrieve(query)

    # ── Public convenience API ────────────────────────────────────────────────

    def retrieve(self, query: str) -> list[Document]:
        """
        Retrieve the top-k relevant document chunks for ``query``.

        Parameters
        ----------
        query:
            Search query string.

        Returns
        -------
        list[Document]
            Relevant document chunks, ordered by descending similarity.

        Raises
        ------
        ValueError
            If ``query`` is empty.
        RuntimeError
            If the underlying vector search fails.
        """
        if not query.strip():
            raise ValueError("Query string must not be empty.")

        logger.info("Retrieving top-%d documents for query: %r", self.k, query[:80])

        if self.score_threshold is not None:
            # Use scored search so we can apply the threshold
            scored_results = self.vector_store.similarity_search_with_score(
                query=query,
                k=self.k,
            )
            docs = [
                doc
                for doc, score in scored_results
                if score <= self.score_threshold
            ]
            logger.debug(
                "Score-filtered results: %d / %d passed threshold %.3f",
                len(docs),
                len(scored_results),
                self.score_threshold,
            )
        else:
            docs = self.vector_store.similarity_search(query=query, k=self.k)

        logger.info("Retrieved %d chunk(s).", len(docs))
        return docs

    # ── Metadata helpers ──────────────────────────────────────────────────────

    @staticmethod
    def extract_sources(documents: list[Document]) -> list[str]:
        """
        Extract unique, human-readable source identifiers from retrieved docs.

        Parameters
        ----------
        documents:
            List of retrieved ``Document`` objects.

        Returns
        -------
        list[str]
            Deduplicated list of source file names (base names only).
        """
        from pathlib import Path

        seen: set[str] = set()
        sources: list[str] = []
        for doc in documents:
            raw_source: str = doc.metadata.get("source", "Unknown")
            # Use only the file name to avoid leaking full server paths
            name = Path(raw_source).name if raw_source != "Unknown" else raw_source
            if name not in seen:
                seen.add(name)
                sources.append(name)
        return sources
