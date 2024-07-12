"""A sub-package for interfacing with the QM/MM/PME-hacked OpenMM.
"""
from __future__ import annotations
__author__ = "John Pederson"

from pydft_qmmm.common import TheoryLevel

from .openmm_factory import pme_openmm_interface_factory


FACTORY = pme_openmm_interface_factory
THEORY_LEVEL = TheoryLevel.QM
