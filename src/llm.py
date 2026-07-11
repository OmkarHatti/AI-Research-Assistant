
"""
src/llm.py
──────────
LLM client layer using Groq.
"""

from __future__ import annotations

from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from src.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    """
    Thin wrapper around ChatGroq.
    """

    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        api_key: str | None = None,
        request_timeout: int = 120,
        **kwargs: Any,
    ) -> None:
        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens

        logger.info(
            "Initializing Groq LLM — model='%s' temperature=%.1f max_tokens=%d",
            model_name,
            temperature,
            max_tokens,
        )

        init_kwargs: dict[str, Any] = {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": request_timeout,
            **kwargs,
        }

        if api_key:
            init_kwargs["api_key"] = api_key

        try:
            self._llm = ChatGroq(**init_kwargs)
        except Exception as exc:
            logger.error("Failed to initialize ChatGroq: %s", exc, exc_info=True)
            raise RuntimeError(f"LLM initialization failed: {exc}") from exc

    def chat(
        self,
        user_message: str,
        system_prompt: str | None = None,
        history: list[BaseMessage] | None = None,
    ) -> str:
        """
        Send a message to the model and return the response.
        """
        if not user_message.strip():
            raise ValueError("user_message must not be empty.")

        messages: list[BaseMessage] = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        if history:
            messages.extend(history)

        messages.append(HumanMessage(content=user_message))

        logger.debug(
            "Sending %d message(s) to %s...",
            len(messages),
            self._model_name,
        )

        try:
            response = self._llm.invoke(messages)
            answer = str(response.content).strip()
        except Exception as exc:
            logger.error("Groq request failed: %s", exc, exc_info=True)
            raise RuntimeError(
                f"LLM call to '{self._model_name}' failed: {exc}"
            ) from exc

        logger.debug("Response received (%d chars).", len(answer))
        return answer

    @property
    def langchain_llm(self) -> ChatGroq:
        return self._llm

    @property
    def model_name(self) -> str:
        return self._model_name