"""Structured logging configuration for T5-SSL.

Provides a pre-configured logger with consistent formatting across all
modules. Usage::

    from t5_ssl.utils.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Training started", extra={"epoch": 1})
"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure_root_logger() -> None:
    """Set up the root ``t5_ssl`` logger once.

    Configures a ``StreamHandler`` writing to *stdout* with a human-readable
    format.  Subsequent calls are no-ops so the handler is never duplicated.
    """
    global _configured
    if _configured:
        return

    root = logging.getLogger("t5_ssl")
    root.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root.addHandler(handler)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``t5_ssl`` namespace.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
        A configured logger instance.

    """
    _configure_root_logger()
    return logging.getLogger(name)
