"""Configuração de logging estruturado com loguru."""

from __future__ import annotations

import sys
from typing import Literal

from loguru import logger

from macroeconomia.config import get_settings


def configure_logging(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] | None = None,
) -> None:
    """Configura o sink padrão do loguru (stdout, formato legível).

    Args:
        level: Nível de log; se ``None``, usa :attr:`Settings.log_level`.
    """
    settings = get_settings()
    log_level = level or settings.log_level
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",
    )
