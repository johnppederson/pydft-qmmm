r"""A module containing unit conversions and constants.

Attributes:
    BOHR_PER_ANGSTROM: The number of atomic length units per Angstrom.
    KJMOL_PER_EH: The amount of energy in kilojoules per mole per
        Hartree.
    KB: The Boltzmann constant in kilojoules per mole per Kelvin.
    ELEMENT_TO_MASS: A dictionary of atomic weights
        (:math:`\mathrm{AMU}`) keyed by the element symbol.
"""
from __future__ import annotations

from enum import Enum


BOHR_PER_ANGSTROM = 1.889726  # a0 / A
KJMOL_PER_EH = 2625.4996      # (kJ / mol) / Eh
KB = 8.31446261815324         # J / (mol * K)
ELEMENT_TO_MASS = {
    "H": 1.0079,
    "He": 4.0026,
    "Li": 6.941,
    "Be": 9.0122,
    "B": 10.811,
    "C": 12.0107,
    "N": 14.0067,
    "O": 15.9994,
    "F": 18.9984,
    "Ne": 20.1797,
    "Na": 22.9898,
    "Mg": 24.3050,
    "Al": 26.9815,
    "Si": 28.0855,
    "P": 30.9738,
    "S": 32.065,
    "Cl": 35.453,
    "Ar": 39.948,
    "K": 39.0983,
    "Ca": 40.078,
    "Sc": 44.9559,
    "Ti": 47.867,
    "V": 50.9415,
    "Cr": 51.9961,
    "Mn": 54.9380,
    "Fe": 55.845,
    "Co": 58.9331,
    "Ni": 58.6934,
    "Cu": 63.546,
    "Zn": 65.409,
    "Ga": 69.723,
    "Ge": 72.64,
    "As": 74.9216,
    "Se": 78.96,
    "Br": 79.904,
    "Kr": 83.798,
    "Rb": 85.4678,
    "Sr": 87.62,
    "Y": 88.9059,
    "Zr": 91.224,
    "Nb": 92.9064,
    "Mo": 95.94,
    "Tc": 98.,
    "Ru": 101.07,
    "Rh": 102.9055,
    "Pd": 106.42,
    "Ag": 107.8682,
    "Cd": 112.411,
    "In": 114.818,
    "Sn": 118.710,
    "Sb": 121.760,
    "Te": 127.60,
    "I": 126.9045,
    "Xe": 131.293,
    "Cs": 132.9055,
    "Ba": 137.327,
    "La": 138.9055,
    "Ce": 140.116,
    "Pr": 140.9077,
    "Nd": 144.242,
    "Pm": 145.,
    "Sm": 150.36,
    "Eu": 151.964,
    "Gd": 157.25,
    "Tb": 158.9254,
    "Dy": 162.500,
    "Ho": 164.9303,
    "Er": 167.259,
    "Tm": 168.9342,
    "Yb": 173.04,
    "Lu": 174.967,
    "Hf": 178.49,
    "Ta": 180.9479,
    "W": 183.84,
    "Re": 186.207,
    "Os": 190.23,
    "Ir": 192.217,
    "Pt": 195.084,
    "Au": 196.9666,
    "Hg": 200.59,
    "Tl": 204.3833,
    "Pb": 207.2,
    "Bi": 208.9804,
    "Po": 209.,
    "At": 210.,
    "Rn": 222.,
    "Fr": 223.,
    "Ra": 226.,
    "Ac": 227.,
    "Th": 232.0381,
    "Pa": 231.0359,
    "U": 238.0289,
    "Np": 237.,
    "Pu": 244.,
    "Am": 243.,
    "Cm": 247.,
    "Bk": 247.,
    "Cf": 251.,
    "Es": 252.,
    "Fm": 257.,
    "Md": 258.,
    "No": 259.,
    "Lr": 262.,
    "Rf": 261.,
    "Db": 262.,
    "Sg": 266.,
    "Bh": 264.,
    "Hs": 277.,
    "Mt": 268.,
    "Ds": 281.,
    "Rg": 272.,
    "Cn": 285.,
    "Nh": 286.,
    "Fl": 289.,
    "Mc": 289.,
    "Lv": 293.,
    "Ts": 294.,
    "Og": 294.,
}


class TheoryLevel(Enum):
    """Enumeration of the different levels of theory.
    """
    NO = "No level of theory (a default)."
    QM = "The quantum mechanical (DFT) level of theory."
    MM = "The molecular mechanical (forcefield) level of theory."


class Subsystem(Enum):
    """Enumeration of the regions of the system.
    """
    NULL = "NULL"
    I = "I"  # noqa: E741
    II = "II"
    III = "III"
