"""
utils.py — Shared logging and retry utilities.

HARNESS ENGINEERING:
    A harness wraps unreliable external systems (APIs, file I/O) so the rest
    of the application never has to deal with transient failures directly.

    Two harness primitives live here:
      1. setup_logger  — gives every module a consistent, timestamped logger
      2. retry         — a decorator that automatically retries any function
                         that raises an exception, up to a configurable limit.

    By centralising these here we avoid copy-pasting try/except blocks across
    every file that touches the OpenAI API.
"""

import functools
import logging
import time
from typing import Any, Callable, Type

from backend.config import LOG_LEVEL, MAX_RETRIES, RETRY_DELAY_SECONDS


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logger(name: str) -> logging.Logger:
    """
    Create (or retrieve) a named logger with a human-readable format.

    Usage:
        logger = setup_logger(__name__)
        logger.info("PDF loaded successfully")

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Only add a handler the first time — prevents duplicate log lines
    # when the module is imported more than once (e.g. in Streamlit reruns).
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    return logger


# ── Retry decorator ───────────────────────────────────────────────────────────

def retry(
    max_attempts: int = MAX_RETRIES,
    delay: float = RETRY_DELAY_SECONDS,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
) -> Callable:
    """
    Decorator factory that retries a function on failure.

    HARNESS ENGINEERING:
        OpenAI's API occasionally returns rate-limit errors (HTTP 429) or
        transient server errors (HTTP 500/503).  Instead of crashing, we
        wait a moment and try again.  This makes the harness resilient
        without polluting business logic with retry boilerplate.

    Args:
        max_attempts: Total number of attempts (first try + retries).
        delay:        Seconds to wait between attempts.
        exceptions:   Which exception types trigger a retry.

    Returns:
        A decorator that wraps the target function with retry logic.

    Example:
        @retry(max_attempts=3, delay=2.0)
        def call_openai(prompt):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)  # preserves original function name and docstring
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _logger = setup_logger(func.__module__)
            last_exception: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    if attempt < max_attempts:
                        _logger.warning(
                            "Attempt %d/%d failed for '%s': %s. "
                            "Retrying in %.1fs...",
                            attempt, max_attempts, func.__name__, exc, delay,
                        )
                        time.sleep(delay)
                    else:
                        _logger.error(
                            "All %d attempts failed for '%s': %s",
                            max_attempts, func.__name__, exc,
                        )

            # Re-raise the last exception so the caller knows what went wrong
            raise last_exception  # type: ignore[misc]

        return wrapper
    return decorator


# ── Miscellaneous helpers ──────────────────────────────────────────────────────

def truncate_text(text: str, max_chars: int = 200) -> str:
    """Return a shortened version of text for display/logging purposes."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"
