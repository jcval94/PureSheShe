"""Shared logging utilities for DelDel modules."""

from __future__ import annotations

import logging


def verbosity_to_level(verbosity: int) -> int:
    """Map a small verbosity integer to a logging level."""
    if verbosity >= 2:
        return logging.DEBUG
    if verbosity == 1:
        return logging.INFO
    return logging.WARNING
