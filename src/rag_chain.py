"""
src/rag_chain.py
────────────────
Orchestration layer — ties together retrieval and generation.

Pipeline per query
──────────────────
1. Retrieve the top-k chunks from the vector store (semantic search).
2. Assemble a context block from the retrieved chunks, labelled with their
   source file name and page / chunk index.
3. Build a system prompt that instructs the LLM to answer *only* from the
   provided context and to admit uncertainty when the context is insufficient.
4. Include the recent conversation history for multi-turn coherence.
5. Call the LLM and return the answer together with the source documents so
   the UI can render citations.

Conversation memory is stored as a simple list of ``(question, answer)``
tuples and trimmed to ``max_history_turns`` to stay within the context window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from src.retriever import Retriever
from src.llm import LLMClient
from src.logger import get_logger

logger = get_logger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT_TEMPLATE: str = """\
You are a knowledgeable research assistant.

Your task is to answer the user's question **based solely on the context \
excerpts provided below**. Each excerpt is labelled with its source.

Rules:
- If the answer is clearly present in the context, provide a concise and \
accurate response.
- If the context is insufficient to answer confidently, say:  
  "The provided documents do not contain enough information to answer this \
question."
- Do **not** fabricate facts, invent citations, or use knowledge outside the \
provided context.
- When quoting or paraphrasing from a specific source, reference it \
naturally (e.g. "According to report.pdf…").
- Format your answer in clear, readable markdown when appropriate.

────────────────────────────────────────────────────────────────────────────
CONTEXT:
{context}
────────────────────────────────────────────────────────────────────────────
"""


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class RAGResult:
    """Structured output from a single RAG query."""

    question: str
    answer: str
    source_documents: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (useful for logging / testing)."""
        return {
            "question": self.question,
            "answer": self.answer,
            "source_documents": self.source_documents,
        }


# ── RAGChain class ────────────────────────────────────────────────────────────


class RAGChain:
    """
    End-to-end RAG pipeline with conversation memory.

    Parameters
    ----------
    retriever:
        Initialised ``Retriever`` instance.
    llm_client:
        Initialised ``LLMClient`` instance.
    max_history_turns:
        Maximum number of (user, assistant) pairs to keep in the context
        window for multi-turn coherence.

    Usage
    -----
    >>> chain = RAGChain(retriever=retriever, llm_client=llm_client)
    >>> result = chain.query("What does the annual report say about revenue?")
    >>> print(result["answer"])
    >>> print(result["source_documents"])
    """

    def __init__(
        self,
        retriever: Retriever,
        llm_client: LLMClient,
        max_history_turns: int = 6,
    ) -> None:
        self._retriever = retriever
        self._llm = llm_client
        self._max_history_turns = max_history_turns
        # History stored as (human_msg, ai_msg) tuples
        self._history: list[tuple[str, str]] = []

        logger.info(
            "RAGChain ready — model=%s k=%d max_history=%d",
            llm_client.model_name,
            retriever.k,
            max_history_turns,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def query(self, question: str) -> dict[str, Any]:
        """
        Answer a question using retrieved context and conversation history.

        Parameters
        ----------
        question:
            The user's natural-language question.

        Returns
        -------
        dict with keys:
            ``"answer"``          – str, the LLM-generated response.
            ``"source_documents"``– list[Document], retrieved chunks used.
            ``"question"``        – str, the original question.

        Raises
        ------
        ValueError
            If ``question`` is blank.
        RuntimeError
            If retrieval or LLM generation fails.
        """
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty.")

        logger.info("Processing query: %r", question[:120])

        # 1 — Retrieve relevant chunks
        docs = self._retriever.retrieve(question)
        logger.debug("Retrieved %d document(s).", len(docs))

        # 2 — Build context block
        context_block = self._build_context(docs)

        # 3 — Build system prompt
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(context=context_block)

        # 4 — Build message history for multi-turn coherence
        history_messages = self._history_to_messages()

        # 5 — Call the LLM
        answer = self._llm.chat(
            user_message=question,
            system_prompt=system_prompt,
            history=history_messages,
        )

        # 6 — Update conversation history
        self._add_to_history(question, answer)

        logger.info(
            "Query answered. Sources: %s",
            [Path(d.metadata.get("source", "")).name for d in docs],
        )

        return {
            "question": question,
            "answer": answer,
            "source_documents": docs,
        }

    def clear_history(self) -> None:
        """Wipe the conversation history (start a fresh chat session)."""
        self._history.clear()
        logger.info("Conversation history cleared.")

    @property
    def history(self) -> list[tuple[str, str]]:
        """Return a copy of the current conversation history."""
        return list(self._history)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_context(docs: list[Any]) -> str:
        """
        Format retrieved documents into a numbered context block.

        Each section is labelled with its source file name and chunk index
        so the LLM can reference them naturally.
        """
        if not docs:
            return "No relevant context found in the uploaded documents."

        sections: list[str] = []
        for idx, doc in enumerate(docs, start=1):
            source = Path(doc.metadata.get("source", "Unknown")).name
            chunk_index = doc.metadata.get("chunk_index", "?")
            page = doc.metadata.get("page", None)

            label_parts = [f"[{idx}] Source: {source}"]
            if page is not None:
                label_parts.append(f"Page: {page + 1}")  # 0-indexed → 1-indexed
            label_parts.append(f"Chunk: {chunk_index}")

            header = " | ".join(label_parts)
            sections.append(f"{header}\n{doc.page_content.strip()}")

        return "\n\n---\n\n".join(sections)

    def _history_to_messages(self) -> list[BaseMessage]:
        """Convert stored (human, ai) tuples to LangChain message objects."""
        messages: list[BaseMessage] = []
        # Take only the most recent N turns
        recent = self._history[-self._max_history_turns:]
        for human_text, ai_text in recent:
            messages.append(HumanMessage(content=human_text))
            messages.append(AIMessage(content=ai_text))
        return messages

    def _add_to_history(self, question: str, answer: str) -> None:
        """Append a turn and trim to max_history_turns."""
        self._history.append((question, answer))
        # Keep only the most recent turns
        if len(self._history) > self._max_history_turns:
            self._history = self._history[-self._max_history_turns:]
