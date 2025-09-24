"""A sub-package for interfacing with OpenMM.
"""
from __future__ import annotations

__author__ = "John Pederson"

from pydft_qmmm.utils import TheoryLevel
from .openmm_factory import openmm_interface_factory as FACTORY

THEORY_LEVEL = TheoryLevel.MM
NAME = "openmm"

del TheoryLevel
