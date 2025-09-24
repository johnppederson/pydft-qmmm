"""An example plugin package.
"""
from __future__ import annotations

from .template_plugin import TemplateCalculatorPlugin
from .template_plugin import TemplateIntegratorPlugin

# The __init__.py of an plugin package must have a PLUGINS list
# specified, since this list of plugins is loaded by the
# plugin_manager.py in the PyDFT-QMMM package.  The plugins in the
# PLUGINS list can be imported at runtime from pydft_qmmm.plugins.
PLUGINS = [TemplateCalculatorPlugin, TemplateIntegratorPlugin]
