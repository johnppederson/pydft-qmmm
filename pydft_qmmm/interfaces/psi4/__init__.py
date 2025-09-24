"""A sub-package for interfacing with Psi4.
"""
from __future__ import annotations

__author__ = "John Pederson"

from pydft_qmmm.utils import TheoryLevel
from .psi4_factory import psi4_interface_factory as FACTORY

THEORY_LEVEL = TheoryLevel.QM
NAME = "psi4"

del TheoryLevel
