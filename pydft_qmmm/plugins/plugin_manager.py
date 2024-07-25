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

try:
    DISCOVERED_PLUGINS: set[str] = {
        point.name for point
        in entry_points().get("pydft_qmmm.plugins", [])
    }
except AttributeError:
    DISCOVERED_PLUGINS = entry_points(
        group="pydft_qmmm.plugins",
    ).names


def get_external_plugins() -> dict[str, Plugin]:
    """Get PyDFT-QMMM plugins from externally installed packages.

    Returns:
        A dictionary of plugin names and loaded classes for the
        PyDFT-QMMM plugin sub-package.
    """
    package_names = [name for name in DISCOVERED_PLUGINS]
    plugins = {}
    for name in package_names:
        module = import_module(name, package=name)
        plugins.update({
            plugin: getattr(module, plugin)
            for plugin in dir(module)
            if not plugin.startswith("__")
        })
    return plugins
