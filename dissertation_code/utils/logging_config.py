"""Logging configuration (backend guidelines §11).

Configure logging once at an entry point via ``configure_logging()``, then obtain module loggers
with ``logging.getLogger(__name__)``. No ``print()`` for diagnostics anywhere in ``dissertation_code``.
"""

from __future__ import annotations

import logging

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int = logging.INFO) -> None:
    """Initialise root logging once. Idempotent — repeated calls do not add handlers.

    Args:
        level: minimum level to emit (DEBUG internals, INFO milestones, WARNING recoverable).
    """
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    logging.basicConfig(level=level, format=_LOG_FORMAT, datefmt=_DATE_FORMAT)
