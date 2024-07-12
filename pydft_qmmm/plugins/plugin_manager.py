"""Functionality for handling external plugin imports.

Attributes:
    DISCOVERED_PLUGINS: A list of entry points into the plugin
        architecture of PyDFT-QMMM within installed package metadata.
"""
from __future__ import annotations

from importlib import import_module
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .plugin import Plugin
    from importlib.metadata import EntryPoint

DISCOVERED_PLUGINS: list[EntryPoint] = entry_points().get(
    "pydft_qmmm.plugins", [],
)


def get_external_plugins() -> dict[str, Plugin]:
    """Get PyDFT-QMMM plugins from externally installed packages.

    Returns:
        A dictionary of plugin names and loaded classes for the
        PyDFT-QMMM plugin sub-package.
    """
    package_names = [point.name for point in DISCOVERED_PLUGINS]
    plugins = {}
    for name in package_names:
        module = import_module(name, package=name)
        plugins.update({
            plugin: getattr(module, plugin)
            for plugin in dir(module)
            if not plugin.startswith("__")
        })
    return plugins
