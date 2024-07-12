#! /usr/bin/env python3
"""An example plugin sub-package.
"""
# The __init__.py of a plugin only needs to import the Plugin subclass.
from __future__ import annotations

from .template_plugin import TemplatePlugin
