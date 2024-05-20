#! /usr/bin/env python3
"""A sub-package to define records kept by the :class:`System`, which
include data pertaining to input files and the state and topology of
atoms and residues.
"""
from __future__ import annotations

from .atom import Atom
from .system import System
from .variable import array_float
from .variable import array_int
from .variable import array_str
from .variable import ArrayValue
from .variable import ObservedArray
__author__ = "John Pederson"
