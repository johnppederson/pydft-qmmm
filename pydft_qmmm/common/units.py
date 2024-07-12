"""A module containing unit conversions and constants.

Attributes:
    BOHR_PER_ANGSTROM: The number of atomic length units per Angstrom.
    KJMOL_PER_EH: The amount of energy in kilojoules per mole per
        Hartree.
    KB: The Boltzmann constant in kilojoules per mole per Kelvin.
"""
from __future__ import annotations


BOHR_PER_ANGSTROM = 1.889726  # a0 / A
KJMOL_PER_EH = 2625.4996      # (kJ / mol) / Eh
KB = 8.31446261815324         # J / (mol * K)
