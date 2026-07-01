"""
llm.py — OpenAI Chat Completions wrapper.

HARNESS ENGINEERING:
    The LLM is the most expensive and fragile part of the pipeline.
    This module is a thin but hardened wrapper that:

      - Validates the API key before making any call
      - Applies the @retry decorator for transient failures
      - Logs token usage so you can monitor costs
      - Raises a clear, actionable error if the key is wrong

    Nothing else in the application imports openai directly.  If we
    ever switch to Anthropic, Ollama, or a local model, only this
    file changes.

    Why not use LangChain's LLM wrapper here?
    LangChain adds abstraction that hides what is actually happening —
    which is the opposite of what a learning project wants.  Direct
    API calls are more transparent and easier to debug.
"""

from typing import Any

import openai

from backend.config import CHAT_MODEL, MAX_TOKENS, OPENAI_API_KEY, TEMPERATURE
from backend.utils import retry, setup_logger

logger = setup_logger(__name__)


class LLMClient:
    """
    Wraps OpenAI chat completions with retry and logging.

    Typical usage:
        client = LLMClient(api_key="sk-...")
        answer = client.complete(messages)
    """

    def __init__(self, api_key: str = "") -> None:
        """
        Args:
            api_key: OpenAI API key. Falls back to config / .env if not given.

        Raises:
            ValueError: If no API key is available.
        """
        resolved_key = api_key or OPENAI_API_KEY
        if not resolved_key:
            raise ValueError(
                "OpenAI API key is missing.\n"
                "  Option 1: Set OPENAI_API_KEY in your .env file.\n"
                "  Option 2: Enter the key in the Streamlit sidebar."
            )
        self._client = openai.OpenAI(api_key=resolved_key)
        self._model = CHAT_MODEL
        logger.info("LLMClient initialised with model '%s'.", self._model)

    # ── Public API ─────────────────────────────────────────────────────────────

    @retry()  # HARNESS ENGINEERING: retries on RateLimitError, APIError, etc.
    def complete(self, messages: list[dict[str, str]]) -> str:
        """
        Send a list of messages to the chat model and return the reply.

        Args:
            messages: OpenAI-format message list from PromptBuilder.build().

        Returns:
            The assistant's reply as a plain string.

        Raises:
            openai.AuthenticationError: If the API key is invalid.
            openai.RateLimitError:      If the rate limit is exceeded
                                        (after all retries are exhausted).
        """
        logger.debug(
            "Calling %s with %d messages…", self._model, len(messages)
        )

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,          # type: ignore[arg-type]
            temperature=TEMPERATURE,    # lower = more factual / deterministic
            max_completion_tokens=MAX_TOKENS,
        )

        # Log token usage so the student can monitor API costs
        usage = response.usage
        if usage:
            logger.info(
                "Token usage — prompt: %d, completion: %d, total: %d",
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
            )

        answer = response.choices[0].message.content or ""
        return answer.strip()

    def get_model_name(self) -> str:
        """Return the model name in use (useful for display in the UI)."""
        return self._model
