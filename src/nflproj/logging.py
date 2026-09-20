"""Structured logging.

JSON in containers, human-readable at a terminal. Chosen by the ``NFLPROJ_LOG_JSON``
environment variable so the same image behaves sensibly in CI and in a shell.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import structlog

_CONFIGURED = False


def configure_logging(*, level: str | None = None, json_output: bool | None = None) -> None:
    """Install the structlog pipeline. Idempotent."""
    global _CONFIGURED  # noqa: PLW0603
    if _CONFIGURED:
        return

    level = level or os.environ.get("NFLPROJ_LOG_LEVEL", "INFO")
    if json_output is None:
        json_output = os.environ.get("NFLPROJ_LOG_JSON", "").lower() in {"1", "true", "yes"}

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level.upper())

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[level.upper()]
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger, configuring the pipeline on first use."""
    configure_logging()
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger
