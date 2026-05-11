"""Testes de logging."""

from __future__ import annotations

from macroeconomia.config import get_settings
from macroeconomia.logging_config import configure_logging


def test_configure_logging_runs() -> None:
    get_settings.cache_clear()
    configure_logging(level="DEBUG")
