"""Structured logging setup with Rich console handler."""

from __future__ import annotations

import logging
import os

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(level: str | None = None) -> None:
    """Configure structured logging with Rich output."""
    log_level = level or os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=Console(stderr=True), rich_tracebacks=True)],
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
