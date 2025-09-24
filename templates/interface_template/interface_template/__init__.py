"""An example package for interfacing to external software.
"""
from __future__ import annotations

from .template_factory import template_interface_factory
from pydft_qmmm.utils import TheoryLevel

# The __init__.py of an interface package must have a NAME, a FACTORY,
# and a THEORY_LEVEL specified, as these are loaded by the
# interface_manager.py in the PyDFT-QMMM package.  The name is a string
# that the user can provide to a Hamiltonian object at runtime to select
# the interface.  I have typically defined interface factory methods in
# a *_factory.py module, but this is a matter of preference.  The
# TheoryLevel class is an Enum class with enumerations of QM, MM, and
# NO (a default setting).
NAME = "template"
FACTORY = template_interface_factory
THEORY_LEVEL = TheoryLevel.NO
