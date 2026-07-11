"""
src/__init__.py
───────────────
Marks ``src`` as a Python package and exposes the most commonly used
symbols for convenient top-level imports.

Example
-------
>>> from src import settings, get_logger
"""

from src.config import settings
from src.logger import get_logger

__all__ = ["settings", "get_logger"]
