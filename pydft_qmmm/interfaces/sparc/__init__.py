"""A sub-package for interfacing with SPARC (via SPARC-X-API).
"""
from __future__ import annotations

from pydft_qmmm.utils import TheoryLevel
from .sparc_factory import sparc_interface_factory as FACTORY

THEORY_LEVEL = TheoryLevel.QM
NAME = "sparc"

del TheoryLevel
