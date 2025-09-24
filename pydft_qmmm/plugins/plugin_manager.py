"""Functionality for importing third-party plugins.

Attributes:
    DISCOVERED_PLUGINS: A tuple of entry points into the plugin
        architecture of PyDFT-QMMM from installed package metadata.
    LOADED_PLUGINS: The loaded plugin modules.
"""
from __future__ import annotations

__all__ = ["get_plugins", "Plugin"]

from importlib.metadata import entry_points
from typing import TypeAlias

from pydft_qmmm.calculators import CalculatorPlugin
from pydft_qmmm.integrators import IntegratorPlugin

Plugin: TypeAlias = CalculatorPlugin | IntegratorPlugin

try:
    # This is for Python 3.10-3.11.
    DISCOVERED_PLUGINS = tuple(
        entry_points(
        ).get("pydft_qmmm.plugins", []),  # type: ignore[attr-defined]
    )
except AttributeError:
    # This is for Python +3.12, importlib.metadata now uses a selectable
    # EntryPoints object.
    DISCOVERED_PLUGINS = tuple(
        entry_points(group="pydft_qmmm.plugins"),
    )

LOADED_PLUGINS = tuple(
    map(lambda x: x.load(), DISCOVERED_PLUGINS),
)


def get_plugins() -> dict[str, Plugin]:
    """Get third-party PyDFT-QMMM plugins.

    Returns:
        A dictionary of plugin classes indexed by plugin name.
    """
    plugins: dict[str, Plugin] = dict()
    for mod in LOADED_PLUGINS:
        plugins.update(dict(map(lambda y: (y.__name__, y), mod.PLUGINS)))
    return plugins
