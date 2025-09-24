"""The atom and system-atom data containers.
"""
from __future__ import annotations

__all__ = ["Atom"]

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import numpy as np

from pydft_qmmm.utils import Subsystem

from .variable import observed_class

if TYPE_CHECKING:
    from typing import Any
    from numpy.typing import NDArray
    from .variable import ObservedArray
    from .variable import array_float
    from .variable import ArrayValue


def _zero_vector() -> NDArray[np.float64]:
    """Create a zero vector with three dimensions.

    Returns:
        An array with three dimensions of zero magnitude.
    """
    return np.array([0., 0., 0.])


@dataclass
class Atom:
    r"""The atom data container.

    Args:
        position: The position (:math:`\mathrm{\mathring{A}}`) of the
            atom within the system.
        velocity: The velocity (:math:`\mathrm{\mathring{A}\;fs^{-1}}`)
            of the atom.
        force: The force
            (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`) acting
            on the atom.
        mass: The mass (:math:`\mathrm{AMU}`) of the atom.
        charge: The partial charge (:math:`e`) of the atom.
        residue: The index of the residue to which the atom belongs.
        element: The element symbol of the atom.
        name: The name (type) of the atom, as in a PDB file.
        residue_name: The name of the residue to which the atom belongs.
        chain: The indentifier of the chain to which the atom belongs.
        subsystem: The subsystem to which the atom belongs.
    """
    # 3D vector quantities
    position: NDArray[np.float64] = field(
        default_factory=_zero_vector,
    )
    velocity: NDArray[np.float64] = field(
        default_factory=_zero_vector,
    )
    force: NDArray[np.float64] = field(
        default_factory=_zero_vector,
    )
    # Scalar quantities
    mass: float = 0.
    charge: float = 0.
    residue: int = 0
    # String values
    element: str = ""
    name: str = ""
    residue_name: str = ""
    chain: str = ""
    # Enumerated values
    subsystem: Subsystem = Subsystem.NULL


@observed_class
class _SystemAtom:
    r"""The atom data container for atoms within a system.

    Attributes:
        position: The position (:math:`\mathrm{\mathring{A}}`) of the
           atom within the system.
        velocity: The velocity (:math:`\mathrm{\mathring{A}\;fs^{-1}}`)
           of the atom.
        force: The force
           (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`) acting
           on the atom.
        mass: The mass (:math:`\mathrm{AMU}`) of the atom.
        charge: The partial charge (:math:`e`) of the atom.
        residue: The index of the residue to which the atom belongs.
        element: The element symbol of the atom.
        name: The name (type) of the atom, as in a PDB file.
        residue_name: The name of the residue to which the atom belongs.
        chain: The indentifier of the chain to which the atom belongs.
        subsystem: The subsystem to which the atom belongs.
    """
    # Arrays
    position: ObservedArray[Any, array_float]
    velocity: ObservedArray[Any, array_float]
    force: ObservedArray[Any, array_float]
    # Array values
    mass: ArrayValue[float]
    charge: ArrayValue[float]
    residue: ArrayValue[int]
    element: ArrayValue[str]
    name: ArrayValue[str]
    residue_name: ArrayValue[str]
    chain: ArrayValue[str]
    subsystem: ArrayValue[Subsystem]

    def __init__(self, **kwargs: Any) -> None:
        for name in getattr(self, "__dataclass_fields__"):
            if name not in kwargs:
                TypeError(f"Missing input {name} to _SystemAtom")
            setattr(self, "_" + name, kwargs[name])
