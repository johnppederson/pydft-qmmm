"""The atom data container.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import numpy as np

from .constants import Subsystem
from .utils import zero_vector

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class Atom:
    r"""The atom data container.

    Args:
        position: The position (:math:`\mathrm{\mathring{A}}`) of the atom
            within the system.
        velocity: The velocity (:math:`\mathrm{\mathring{A}\;fs^{-1}}`) of
            the atom.
        force: The force (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`)
            acting on the atom.
        mass: The mass (:math:`\mathrm{AMU}`) of the atom.
        charge: The partial charge (:math:`e`) of the atom.
        residue: The index of the residue to which the atom belongs.
        element: The element symbol of the atom.
        name: The name (type) of the atom, as in a PDB file.
        residue_name: The name of the residue to which the atom belongs.
        subsystem: The subsystem of which the atom is a part.
    """
    # 3D vector quantities
    position: NDArray[np.float64] = field(
        default_factory=zero_vector,
    )
    velocity: NDArray[np.float64] = field(
        default_factory=zero_vector,
    )
    force: NDArray[np.float64] = field(
        default_factory=zero_vector,
    )
    # Scalar quantities
    mass: float = 0.
    charge: float = 0.
    residue: int = 0
    # String values
    element: str = ""
    name: str = ""
    residue_name: str = ""
    # Enumerated values
    subsystem: Subsystem = Subsystem.NULL
