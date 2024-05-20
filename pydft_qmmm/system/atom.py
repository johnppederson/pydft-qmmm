#! /usr/bin/env python3
"""A module defining the :class:`State` data container.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import numpy as np

from pydft_qmmm.common import Subsystem

if TYPE_CHECKING:
    from typing import Any
    from numpy.typing import NDArray
    from .variable import ObservedArray
    from .variable import array_float
    from .variable import ArrayValue


def _zero_vector() -> NDArray[np.float64]:
    return np.array([0., 0., 0.])


@dataclass
class Atom:
    """
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
    molecule: int = 0
    # String values
    element: str = ""
    name: str = ""
    molecule_name: str = ""
    # Enumerated values
    subsystem: Subsystem = Subsystem.NULL


@dataclass
class _SystemAtom:
    """
    """
    # Arrays
    _position: ObservedArray[Any, array_float]
    _velocity: ObservedArray[Any, array_float]
    _force: ObservedArray[Any, array_float]
    # Array values
    _mass: ArrayValue[float]
    _charge: ArrayValue[float]
    _molecule: ArrayValue[int]
    _element: ArrayValue[str]
    _name: ArrayValue[str]
    _molecule_name: ArrayValue[str]
    _subsystem: ArrayValue[Subsystem]

    def __repr__(self) -> str:
        string = "SystemAtom("
        string += f"position={self.position}, "
        string += f"velocity={self.velocity}, "
        string += f"force={self.force}, "
        string += f"mass={self.mass}, "
        string += f"charge={self.charge}, "
        string += f"molecule={int(self.molecule)}, "
        string += f"element={self.element}, "
        string += f"name={self.name}, "
        string += f"molecule_name={self.molecule_name}, "
        string += f"subsystem={self.subsystem})"
        return string

    @property
    def position(self) -> ObservedArray[Any, array_float]: return self._position

    @position.setter
    def position(self, position: NDArray[np.float64]) -> None:
        self._position[:] = position

    @property
    def velocity(self) -> ObservedArray[Any, array_float]: return self._velocity

    @velocity.setter
    def velocity(self, velocity: NDArray[np.float64]) -> None:
        self._velocity[:] = velocity

    @property
    def force(self) -> ObservedArray[Any, array_float]: return self._force

    @force.setter
    def force(self, force: NDArray[np.float64]) -> None:
        self._force[:] = force

    @property
    def mass(self) -> float: return self._mass.value()

    @mass.setter
    def mass(self, mass: float) -> None:
        self._mass.update(mass)

    @property
    def charge(self) -> float: return self._charge.value()

    @charge.setter
    def charge(self, charge: float) -> None:
        self._charge.update(charge)

    @property
    def molecule(self) -> float: return self._molecule.value()

    @molecule.setter
    def molecule(self, molecule: int) -> None:
        self._molecule.update(molecule)

    @property
    def element(self) -> str: return self._element.value()

    @element.setter
    def element(self, element: str) -> None:
        self._element.update(element)

    @property
    def name(self) -> str: return self._name.value()

    @name.setter
    def name(self, name: str) -> None:
        self._name.update(name)

    @property
    def molecule_name(self) -> str: return self._molecule_name.value()

    @molecule_name.setter
    def molecule_name(self, molecule_name: str) -> None:
        self._molecule_name.update(molecule_name)

    @property
    def subsystem(self) -> Subsystem: return self._subsystem.value()

    @subsystem.setter
    def subsystem(self, subsystem: Subsystem) -> None:
        self._subsystem.update(subsystem)
