"""Base classes for software interfaces and related classes.

This module contains the software interface base class and the base
classes for QM and MM interfaces, as well as the MM and QM potential
and factory types.
"""
from __future__ import annotations

__all__ = [
    "SoftwareInterface",
    "QMInterface",
    "QMPotential",
    "QMFactory",
    "MMInterface",
    "MMPotential",
    "MMFactory",
]

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from collections.abc import Callable
from typing import TypeAlias
from typing import TYPE_CHECKING

# These are required outside of the TYPE_CHECKING guard in order to
# produce correct docs.
import numpy as np
from numpy.typing import NDArray

from pydft_qmmm.utils import TheoryLevel
from pydft_qmmm.potentials import AtomicPotential

if TYPE_CHECKING:
    from pydft_qmmm import System
    from pydft_qmmm.potentials import ElectronicPotential


@dataclass(frozen=True)
class SoftwareInterface(ABC):
    """The abstract software interface base class.

    Args:
        system: The system that will inform the interface to the
            external software.
    """
    system: System


@dataclass(frozen=True)
class MMInterface(SoftwareInterface):
    """The abstract MM interface base class.

    Args:
        system: The system that will inform the interface to the
            external software.

    Attributes:
        theory_level: The level of theory that the software applies in
            energy and force calculations.
    """
    theory_level: TheoryLevel = field(default=TheoryLevel.MM, init=False)

    @abstractmethod
    def zero_intramolecular(self, atoms: frozenset[int]) -> None:
        """Remove intra-molecular interactions for the specified atoms.

        Args:
            atoms: The indices of atoms to from which to remove
                intra-molecular interactions.
        """

    @abstractmethod
    def zero_charges(self, atoms: frozenset[int]) -> None:
        """Remove charges from the specified atoms.

        Args:
            atoms: The indices of atoms from which to remove charges.
        """

    @abstractmethod
    def zero_intermolecular(self, atoms: frozenset[int]) -> None:
        """Remove inter-molecular interactions for the specified atoms.

        Args:
            atoms: The indices of atoms from which to remove
                inter-molecular interactions.
        """

    @abstractmethod
    def zero_forces(self, atoms: frozenset[int]) -> None:
        """Zero forces on the specified atoms.

        Args:
            atoms: The indices of atoms for which to zero forces.
        """

    @abstractmethod
    def add_real_elst(
            self,
            atoms: frozenset[int],
            const: float | int = 1,
            inclusion: NDArray[np.float64] | None = None,
    ) -> None:
        """Add Coulomb interaction for the specified atoms.

        Args:
            atoms: The indices of atoms for which to add a Coulomb
                interaction.
            const: A constant to multiply at the beginning of the
                coulomb expression.
            inclusion: An Nx3 array with values that will be applied to
                the forces of the Coulomb interaction through
                element-wise multiplication.
        """

    @abstractmethod
    def add_non_elst(
            self,
            atoms: frozenset[int],
            inclusion: NDArray[np.float64] | None = None,
    ) -> None:
        """Add a non-electrostatic interaction for the specified atoms.

        Args:
            atoms: The indices of atoms for which to add
                non-electrostatic, non-bonded interactions.
            inclusion: An Nx3 array with values that will be applied to
                the forces of the non-electrostatic interaction through
                element-wise multiplication.
        """

    @abstractmethod
    def get_pme_parameters(self) -> tuple[float, tuple[int, int, int], int]:
        r"""Get the parameters used for PME summation.

        Returns:
            The Gaussian width parameter in Ewald summation
            (:math:`\mathrm{\mathring(A)^{-1}}`), the number of grid
            points to include along each lattice edge, and the order of
            splines used on the FFT grid.
        """


@dataclass(frozen=True)
class QMInterface(SoftwareInterface):
    """The abstract QM interface base class.

    Args:
        system: The system that will inform the interface to the
            external software.

    Attributes:
        theory_level: The level of theory that the software applies in
            energy and force calculations.
    """
    theory_level: TheoryLevel = field(default=TheoryLevel.QM, init=False)

    @abstractmethod
    def add_electronic_potential(self, potential: ElectronicPotential) -> None:
        """Add an electronic potential to apply before calculations.

        Args:
            potential: The electronic potential to incorporate into
                QM calculations.
        """


class MMPotential(AtomicPotential, MMInterface):
    """A subclass of MM interface and potential for typing purposes.
    """


class QMPotential(AtomicPotential, QMInterface):
    """A subclass of QM interface and potential for typing purposes.
    """


MMFactory: TypeAlias = Callable[..., MMPotential]
QMFactory: TypeAlias = Callable[..., QMPotential]
