"""
src/logger.py
─────────────
Configures and returns named Python loggers.

Features
--------
* Console handler (colourised by level).
* Rotating file handler that keeps the 5 most recent 5 MB log files.
* Single ``configure_logging()`` call at import time — subsequent calls are
  no-ops so that Streamlit hot-reloads do not duplicate handlers.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_CONFIGURED: bool = False  # guard against double-initialisation


# ── ANSI colour codes (console only) ─────────────────────────────────────────

_COLOURS: dict[int, str] = {
    logging.DEBUG:    "\033[36m",   # cyan
    logging.INFO:     "\033[32m",   # green
    logging.WARNING:  "\033[33m",   # yellow
    logging.ERROR:    "\033[31m",   # red
    logging.CRITICAL: "\033[35m",   # magenta
}
_RESET: str = "\033[0m"


class _ColouredFormatter(logging.Formatter):
    """Formatter that prepends an ANSI colour code to the level name."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        colour = _COLOURS.get(record.levelno, "")
        record.levelname = f"{colour}{record.levelname:<8}{_RESET}"
        return super().format(record)


# ── Public API ────────────────────────────────────────────────────────────────


def configure_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
) -> None:
    """
    Initialise the root logger with a console handler and an optional file handler.

    Parameters
    ----------
    level:
        Minimum logging level string (e.g. ``"DEBUG"``, ``"INFO"``).
    log_file:
        If supplied, log records are also written to this rotating file.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # ── Console ──
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(numeric_level)
    console.setFormatter(
        _ColouredFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(console)

    # ── Rotating file ──
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=str(log_file),
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger, initialising the logging system on first call.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    # Lazy import to avoid a circular dependency with config
    from src.config import settings  # noqa: PLC0415

    configure_logging(
        level=settings.LOG_LEVEL,
        log_file=settings.log_file,
    )
    return logging.getLogger(name)
