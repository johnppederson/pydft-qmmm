"""Interface to periodic PySCF."""
from __future__ import annotations

from pydft_qmmm.utils import TheoryLevel
from .pbc_factory import pyscf_pbc_interface_factory as FACTORY

THEORY_LEVEL = TheoryLevel.QM
NAME = "pyscf-pbc"

del TheoryLevel
