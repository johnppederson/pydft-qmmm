#! /usr/bin/env python3
"""A sub-package for interfacing to Psi4.
"""
from __future__ import annotations
__author__ = "John Pederson"

from pydft_qmmm.common import TheoryLevel

from .psi4_utils import psi4_interface_factory


FACTORY = psi4_interface_factory
THEORY_LEVEL = TheoryLevel.QM
