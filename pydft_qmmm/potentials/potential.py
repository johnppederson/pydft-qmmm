"""Base classes for the potential API.

This module contains the atomic potential abstract base class, which
represents any method that can be used to calculate energies and forces
on atomic centers, and the electronic potential abstract base class,
which can be used to evaluate an electronic potential at arbitrary
points for the purpose of integration by quadrature.
"""
from __future__ import annotations

__all__ = [
    "AtomicPotential",
    "ElectronicPotential",
]

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class AtomicPotential(ABC):
    """Representation of an arbitrary potential on atoms."""

    @abstractmethod
    def compute_energy(self) -> float:
        r"""Compute the system energy associated with the potential.

        Returns:
            The energy (:math:`\mathrm{kJ\;mol^{-1}}`) of the potential.
        """

    @abstractmethod
    def compute_forces(self) -> NDArray[np.float64]:
        r"""Compute the forces on the system induced by the potential.

        Returns:
            The forces
            (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`) acting
            on atoms in the system.
        """

    @abstractmethod
    def compute_components(self) -> dict[str, float]:
        r"""Compute the components of energy associated with the potential.

        Returns:
            The components of the energy (:math:`\mathrm{kJ\;mol^{-1}}`)
            of the system.
        """


class ElectronicPotential(ABC):
    """Representation of an arbitrary potential on electrons.

    This class is aimed at providing the necessary tooling for
    generating a potential matrix via numerical quadrature.
    """

    @abstractmethod
    def compute_potential(
            self,
            coordinates: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""Calculate the potential at arbitrary coordinates.

        Args:
            coordinates: An array of coordinates
                (:math:`\mathrm{\mathring{A}}`) at which to calculate
                the potential.

        Returns:
            An array of the potential
            (:math:`\mathrm{kJ\;mol^{-1}\;e^{-1}}`),
            corresponding to the provided coordinates.
        """
