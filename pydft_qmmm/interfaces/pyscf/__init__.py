"""Interface to PySCF."""
from __future__ import annotations

from .pyscf_factory import pyscf_interface_factory as FACTORY
from pydft_qmmm.utils import TheoryLevel

THEORY_LEVEL = TheoryLevel.QM
NAME = "pyscf-mol"

del TheoryLevel
