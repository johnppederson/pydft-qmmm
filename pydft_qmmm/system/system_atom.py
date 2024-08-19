"""The atom data container.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pydft_qmmm.common import Subsystem

if TYPE_CHECKING:
    from typing import Any
    from numpy.typing import NDArray
    from .variable import ObservedArray
    from .variable import array_float
    from .variable import ArrayValue


@dataclass
class _SystemAtom:
    r"""The atom data container for atoms within a system.

    Args:
        _position: The position (:math:`\mathrm{\mathring{A}}`) of the atom
            within the system.
        _velocity: The velocity (:math:`\mathrm{\mathring{A}\;fs^{-1}}`) of
            the atom.
        _force: The force (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`)
            acting on the atom.
        _mass: The mass (:math:`\mathrm{AMU}`) of the atom.
        _charge: The partial charge (:math:`e`) of the atom.
        _residue: The index of the residue to which the atom belongs.
        _element: The element symbol of the atom.
        _name: The name (type) of the atom, as in a PDB file.
        _residue_name: The name of the residue to which the atom
            belongs.
        _subsystem: The subsystem of which the atom is a part.
    """
    # Arrays
    _position: ObservedArray[Any, array_float]
    _velocity: ObservedArray[Any, array_float]
    _force: ObservedArray[Any, array_float]
    # Array values
    _mass: ArrayValue[float]
    _charge: ArrayValue[float]
    _residue: ArrayValue[int]
    _element: ArrayValue[str]
    _name: ArrayValue[str]
    _residue_name: ArrayValue[str]
    _subsystem: ArrayValue[Subsystem]

    def __repr__(self) -> str:
        """Write a string representation of the system atom.

        Returns:
            The string representation of the system atom.
        """
        string = "SystemAtom("
        string += f"position={self.position}, "
        string += f"velocity={self.velocity}, "
        string += f"force={self.force}, "
        string += f"mass={self.mass}, "
        string += f"charge={self.charge}, "
        string += f"residue={int(self.residue)}, "
        string += f"element={self.element}, "
        string += f"name={self.name}, "
        string += f"residue_name={self.residue_name}, "
        string += f"subsystem={self.subsystem})"
        return string

    @property
    def position(self) -> ObservedArray[Any, array_float]:
        r"""The position (:math:`\mathrm{\mathring{A}}`) of the atom."""
        return self._position

    @position.setter
    def position(self, position: NDArray[np.float64]) -> None:
        self._position[:] = position

    @property
    def velocity(self) -> ObservedArray[Any, array_float]:
        r"""The velocity (:math:`\mathrm{\mathring{A}\;fs^{-1}}`) of the
        atom.
        """
        return self._velocity

    @velocity.setter
    def velocity(self, velocity: NDArray[np.float64]) -> None:
        self._velocity[:] = velocity

    @property
    def force(self) -> ObservedArray[Any, array_float]:
        r"""The force (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`) on the
        atom.
        """
        return self._force

    @force.setter
    def force(self, force: NDArray[np.float64]) -> None:
        self._force[:] = force

    @property
    def mass(self) -> float:
        r"""The mass (:math:`\mathrm{AMU}`) of the atom."""
        return self._mass.value()

    @mass.setter
    def mass(self, mass: float) -> None:
        self._mass.update(mass)

    @property
    def charge(self) -> float:
        """The partial charge (:math:`e`) of the atom."""
        return self._charge.value()

    @charge.setter
    def charge(self, charge: float) -> None:
        self._charge.update(charge)

    @property
    def residue(self) -> float:
        """The index of the residue to which the atom belongs."""
        return self._residue.value()

    @residue.setter
    def residue(self, residue: int) -> None:
        self._residue.update(residue)

    @property
    def element(self) -> str:
        """The element symbol of the atom."""
        return self._element.value()

    @element.setter
    def element(self, element: str) -> None:
        self._element.update(element)

    @property
    def name(self) -> str:
        """The name (type) of the atom, as in a PDB file."""
        return self._name.value()

    @name.setter
    def name(self, name: str) -> None:
        self._name.update(name)

    @property
    def residue_name(self) -> str:
        """The name of the residue to which the atom belongs."""
        return self._residue_name.value()

    @residue_name.setter
    def residue_name(self, residue_name: str) -> None:
        self._residue_name.update(residue_name)

    @property
    def subsystem(self) -> Subsystem:
        """The subsystem of which the atom is a part."""
        return self._subsystem.value()

    @subsystem.setter
    def subsystem(self, subsystem: Subsystem) -> None:
        self._subsystem.update(subsystem)
