"""
src/embeddings.py
─────────────────
Embedding layer wrapping ``sentence-transformers`` via LangChain's
``HuggingFaceEmbeddings`` adapter.

Why sentence-transformers?
* Runs locally — no API key required for embedding.
* Dense bi-encoder models (e.g. ``all-MiniLM-L6-v2``) are fast, small,
  and produce high-quality semantic vectors for RAG retrieval.
* Easily swappable: change ``EMBEDDING_MODEL`` in ``.env`` to any model
  hosted on the HuggingFace Hub.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings

from src.logger import get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    """
    Thin wrapper around ``HuggingFaceEmbeddings`` (sentence-transformers).

    The underlying model is loaded lazily on first use and then cached for
    the lifetime of the instance.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier, e.g. ``"all-MiniLM-L6-v2"``.
    device:
        Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).  When
        ``None`` the library auto-detects the best available device.
    encode_kwargs:
        Additional keyword arguments forwarded to the ``encode()`` call,
        e.g. ``{"normalize_embeddings": True}``.

    Usage
    -----
    >>> model = EmbeddingModel("all-MiniLM-L6-v2")
    >>> vectors = model.embed_documents(["Hello world", "Foo bar"])
    >>> query_vec = model.embed_query("Hello world")
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        encode_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device  
        self._encode_kwargs: dict[str, Any] = encode_kwargs or {
            "normalize_embeddings": True
        }
        logger.info(
            "EmbeddingModel configured — model='%s' device=%s",
            model_name,
            device or "auto",
        )

    

    @cached_property
    def _hf_embeddings(self) -> HuggingFaceEmbeddings:
        """Load the model on first access."""
        logger.info("Loading sentence-transformer model '%s'…", self._model_name)
        kwargs: dict[str, Any] = {
            "model_name": self._model_name,
            "encode_kwargs": self._encode_kwargs,
        }
        if self._device is not None:
            kwargs["model_kwargs"] = {"device": self._device}

        try:
            embeddings = HuggingFaceEmbeddings(**kwargs)
        except Exception as exc:
            logger.error(
                "Failed to load embedding model '%s': %s",
                self._model_name,
                exc,
                exc_info=True,
            )
            raise RuntimeError(
                f"Could not load embedding model '{self._model_name}': {exc}"
            ) from exc

        logger.info("Embedding model loaded successfully.")
        return embeddings

    

    @property
    def langchain_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Return the raw ``HuggingFaceEmbeddings`` instance.

        Use this when integrating with LangChain components (e.g. Chroma)
        that require a ``langchain_core.embeddings.Embeddings`` object.
        """
        return self._hf_embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of document strings.

        Parameters
        ----------
        texts:
            List of text strings to embed.

        Returns
        -------
        list[list[float]]
            One float vector per input text.
        """
        if not texts:
            return []
        logger.debug("Embedding %d document(s)…", len(texts))
        try:
            return self._hf_embeddings.embed_documents(texts)
        except Exception as exc:
            logger.error("embed_documents failed: %s", exc, exc_info=True)
            raise RuntimeError(f"Document embedding failed: {exc}") from exc

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string.

        Parameters
        ----------
        query:
            The search query to embed.

        Returns
        -------
        list[float]
            A single embedding vector.
        """
        if not query.strip():
            raise ValueError("Query string must not be empty.")
        logger.debug("Embedding query (%d chars)…", len(query))
        try:
            return self._hf_embeddings.embed_query(query)
        except Exception as exc:
            logger.error("embed_query failed: %s", exc, exc_info=True)
            raise RuntimeError(f"Query embedding failed: {exc}") from exc

    @property
    def model_name(self) -> str:
        """Return the model identifier string."""
        return self._model_name
