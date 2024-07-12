"""A sub-package for interfacing with the QM/MM/PME-hacked Psi4.
"""
from __future__ import annotations
__author__ = "John Pederson"

from pydft_qmmm.common import TheoryLevel

from .psi4_utils import pme_psi4_interface_factory


FACTORY = pme_psi4_interface_factory
THEORY_LEVEL = TheoryLevel.QM
