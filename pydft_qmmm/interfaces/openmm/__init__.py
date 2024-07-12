"""A sub-package for interfacing with OpenMM.
"""
from __future__ import annotations
__author__ = "John Pederson"

from pydft_qmmm.common import TheoryLevel

from .openmm_factory import openmm_interface_factory


FACTORY = openmm_interface_factory
THEORY_LEVEL = TheoryLevel.MM
