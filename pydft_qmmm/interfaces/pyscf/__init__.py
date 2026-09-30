"""Interface to PySCF."""
from __future__ import annotations

from pydft_qmmm.utils import TheoryLevel
from .pyscf_factory import pyscf_interface_factory as FACTORY

THEORY_LEVEL = TheoryLevel.QM
NAME = "pyscf-mol"

del TheoryLevel
