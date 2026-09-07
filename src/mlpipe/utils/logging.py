"""Logging utilities for MLPipe."""

import logging
import sys
from typing import Optional

_LOGGER_NAME = "mlpipe"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get the MLPipe logger."""
    logger_name = _LOGGER_NAME if not name else f"{_LOGGER_NAME}.{name}"
    return logging.getLogger(logger_name)


def configure_logging(verbose: bool = False) -> None:
    """Configure MLPipe logging format and verbosity."""
    logger = logging.getLogger(_LOGGER_NAME)
    logger.handlers.clear()

    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)

    if verbose:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    else:
        formatter = logging.Formatter(fmt="%(message)s")

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
