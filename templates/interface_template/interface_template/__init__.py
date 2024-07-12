#! /usr/bin/env python3
"""An example sub-package for interfacing to external software.
"""
from __future__ import annotations

from .template_utils import template_interface_factory
from pydft_qmmm.common import TheoryLevel

# The __init__.py of an interface package must have a FACTORY and
# THEORY_LEVEL specified, as these are loaded by the
# interface_manager.py in the PyDFT-QMMM package.  I have typically
# defined interface factory methods in a *_utils.py module, but this is
# a matter of preference.  The TheoryLevel class is an Enum class with
# enumerations of QM, MM, and NO (a default setting).
FACTORY = template_interface_factory
THEORY_LEVEL = TheoryLevel.NO
