"""Functionality for importing interfaces to external software.

Attributes:
    DISCOVERED_INTERFACES: A tuple of entry points into the interface
        architecture of PyDFT-QMMM from installed package metadata.
    LOADED_INTERFACES: The loaded interface modules.
"""
from __future__ import annotations

__all__ = ["get_interfaces"]

from importlib.metadata import entry_points
from typing import TYPE_CHECKING

import pydft_qmmm.interfaces.psi4 as psi4
import pydft_qmmm.interfaces.openmm as openmm

if TYPE_CHECKING:
    from typing import TypeAlias
    from .interface import QMFactory
    from .interface import MMFactory
    from pydft_qmmm import TheoryLevel

    Factory: TypeAlias = QMFactory | MMFactory


try:
    # This is for Python 3.10-3.11.
    DISCOVERED_INTERFACES = tuple(
        entry_points(
        ).get("pydft_qmmm.interfaces", []),  # type: ignore[attr-defined]
    )
except AttributeError:
    # This is for Python +3.12, importlib.metadata now uses a selectable
    # EntryPoints object.
    DISCOVERED_INTERFACES = tuple(
        entry_points(group="pydft_qmmm.interfaces"),
    )

LOADED_INTERFACES = tuple(
    map(lambda x: x.load(), DISCOVERED_INTERFACES),
) + (psi4, openmm)


def get_interfaces() -> dict[str, tuple[TheoryLevel, Factory]]:
    """Get PyDFT-QMMM interfaces to external packages.

    Returns:
        A dictionary of interface theory levels and factory functions
        indexed by interface name.
    """
    interfaces = dict(
        map(
            lambda y: (y.NAME, (y.THEORY_LEVEL, y.FACTORY)),
            LOADED_INTERFACES,
        ),
    )
    return interfaces
