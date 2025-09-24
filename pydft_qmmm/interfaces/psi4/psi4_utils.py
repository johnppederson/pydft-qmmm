"""Functionality for setting Psi4 options and disabling logging.
"""
from __future__ import annotations

__all__ = [
    "Psi4Options",
    "set_options",
    "disable_logging",
]

from typing import TypeAlias

import psi4

Psi4Options: TypeAlias = str | int | float | bool


def set_options(
        **options: Psi4Options,
) -> None:
    """Set additional options for Psi4.

    Args:
        options: Additional options to provide to Psi4.  See
            `Psi4 options`_ for additional Psi4 options.
    """
    psi4.set_options(options)


def disable_logging() -> None:
    """Disable Psi4 logging temporarily by directing output to null.
    """
    psi4.core.set_output_file("/dev/null", True)
