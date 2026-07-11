"""
src/chunker.py
──────────────
Text splitting / chunking layer.

Uses LangChain's ``RecursiveCharacterTextSplitter``, which tries to split
along natural boundaries (paragraphs → sentences → words → characters)
before resorting to hard cuts.

This keeps semantic coherence high while respecting the configured chunk
size, which directly affects embedding quality and retrieval precision.
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.logger import get_logger

logger = get_logger(__name__)


class TextChunker:
    """
    Wraps ``RecursiveCharacterTextSplitter`` with project-specific defaults
    and enhanced logging.

    Parameters
    ----------
    chunk_size:
        Target maximum character length for each chunk.
    chunk_overlap:
        Number of characters that consecutive chunks share.  Overlap helps
        the retriever find answers that straddle chunk boundaries.
    separators:
        Ordered list of separator strings the splitter will try.  If
        ``None``, the LangChain defaults are used (``["\\n\\n", "\\n", " ", ""]``).

    Usage
    -----
    >>> chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
    >>> chunks = chunker.split(documents)
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})."
            )

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n\n", "\n", ". ", " ", ""],
            length_function=len,
            add_start_index=True,  
        )

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        logger.debug(
            "TextChunker ready — size=%d overlap=%d",
            chunk_size,
            chunk_overlap,
        )

    

    def split(self, documents: list[Document]) -> list[Document]:
        """
        Split a list of LangChain Documents into smaller chunks.

        Each resulting chunk inherits the metadata of its parent document,
        with an additional ``start_index`` key indicating its character
        offset within the original page content.

        Parameters
        ----------
        documents:
            List of ``Document`` objects as returned by a loader.

        Returns
        -------
        list[Document]
            Flat list of chunk ``Document`` objects.

        Raises
        ------
        ValueError
            If ``documents`` is empty.
        RuntimeError
            If the splitter raises an unexpected error.
        """
        if not documents:
            logger.warning("TextChunker.split() received an empty document list.")
            return []

        logger.info(
            "Splitting %d document(s) into chunks (size=%d, overlap=%d)…",
            len(documents),
            self._chunk_size,
            self._chunk_overlap,
        )

        try:
            chunks: list[Document] = self._splitter.split_documents(documents)
        except Exception as exc:
            logger.error("Chunking failed: %s", exc, exc_info=True)
            raise RuntimeError(f"Text chunking failed: {exc}") from exc

        
        self._add_chunk_indices(chunks)

        logger.info(
            "Produced %d chunks from %d document(s).",
            len(chunks),
            len(documents),
        )
        return chunks

    

    @staticmethod
    def _add_chunk_indices(chunks: list[Document]) -> None:
        """
        Mutate chunks in-place to add ``chunk_index`` to each document's
        metadata, counting separately per source file.
        """
        counters: dict[str, int] = {}
        for chunk in chunks:
            source: str = chunk.metadata.get("source", "unknown")
            idx = counters.get(source, 0)
            chunk.metadata["chunk_index"] = idx
            counters[source] = idx + 1
