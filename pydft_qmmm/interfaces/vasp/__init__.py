"""A sub-package for interfacing with VASP.
"""
from __future__ import annotations

__author__ = "Chengyuan Shao"

from pydft_qmmm.utils import TheoryLevel
from .vasp_factory import vasp_interface_factory as FACTORY

THEORY_LEVEL = TheoryLevel.QM
NAME = "vasp"

del TheoryLevel
