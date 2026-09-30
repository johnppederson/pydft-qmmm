"""Discover bundled and installed interfaces.

BUNDLED_INTERFACES lists Python module names; LOADED_INTERFACES holds
loaded modules. UNAVAILABLE_INTERFACES records bundled import failures.
"""
from __future__ import annotations

__all__ = ["get_interfaces"]

import importlib
from importlib.metadata import entry_points
from types import ModuleType
from typing import TYPE_CHECKING

BUNDLED_INTERFACES = (
    "psi4", "openmm", "sparc", "vasp", "pyscf", "pyscf_pbc",
)

if TYPE_CHECKING:
    from typing import TypeAlias
    from .interface import QMFactory
    from .interface import MMFactory
    from pydft_qmmm.utils import TheoryLevel

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

UNAVAILABLE_INTERFACES: dict[str, str] = {}


def _load_bundled(name: str) -> ModuleType | None:
    """Import a bundled interface, recording missing dependencies."""
    try:
        return importlib.import_module(f"pydft_qmmm.interfaces.{name}")
    except ImportError as exc:
        UNAVAILABLE_INTERFACES[name] = str(exc)
        return None


LOADED_INTERFACES = tuple(
    map(lambda x: x.load(), DISCOVERED_INTERFACES),
) + tuple(
    module for module in map(_load_bundled, BUNDLED_INTERFACES)
    if module is not None
)


def get_interfaces() -> dict[str, tuple[TheoryLevel, Factory]]:
    """Return interface names mapped to (theory level, factory)."""
    interfaces = dict(
        map(
            lambda y: (y.NAME, (y.THEORY_LEVEL, y.FACTORY)),
            LOADED_INTERFACES,
        ),
    )
    return interfaces
