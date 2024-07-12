"""Base classes for software interfaces and settings.
"""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import TypeVar

from pydft_qmmm.common import TheoryLevel

if TYPE_CHECKING:
    from pydft_qmmm import System
    from numpy.typing import NDArray
    import numpy as np

T = TypeVar("T")


class SoftwareSettings(ABC):
    """The abstract software settings base class.

    .. note:: This currently doesn't do anything.
    """


@dataclass(frozen=True)
class MMSettings(SoftwareSettings):
    r"""Immutable container which holds settings for an MM interface.

    Args:
        system: The system which will be tied to the MM interface.
        forcefield_file: The FF XML file containing forcefield data
            for the system.
        topology_file: The FF XML file containing topology data for
            the system.
        nonbonded_method: The method for treating non-bonded
            interactions, as in OpenMM.
        nonbonded_cutoff: The distance at which to truncate close-range
            non-bonded interactions.
        pme_gridnumber: The number of grid points to include along each
            lattice edge in PME summation.
        pme_alpha: The Gaussian width parameter in Ewald summation
            (:math:`\mathrm{nm^{-1}}`).
    """
    system: System
    forcefield_file: str | list[str]
    topology_file: list[str] | None = None
    nonbonded_method: str = "PME"
    nonbonded_cutoff: float | int = 14.
    pme_gridnumber: int | None = None
    pme_alpha: float | int | None = None


@dataclass(frozen=True)
class QMSettings(SoftwareSettings):
    """Immutable container which holds settings for an QM interface.

    Args:
        system: The system which will be tied to the QM interface.
        basis_set: The name of the basis set to be used in QM
            calculations.
        functional: The name of the functional set to be used in QM
            calculations.
        charge: The net charge (:math:`e`) of the system represented by
            the QM Hamiltonian.
        spin: The net spin of the system represented by the QM
            Hamiltonian
        quadrature_spherical: The number of spherical Lebedev points
            to use in the DFT quadrature.
        quadrature_radial: The number of radial points to use in the
            DFT quadrature.
        scf_type: The name of the type of SCF to perform, relating to
            the JK build algorithms as in Psi4.
        read_guess: Whether or not to reuse previous wavefunctions as
            initial guesses in subsequent QM calculations.
    """
    system: System
    basis_set: str
    functional: str
    charge: int
    spin: int
    quadrature_spherical: int = 302
    quadrature_radial: int = 75
    scf_type: str = "df"
    read_guess: bool = True


class SoftwareInterface(ABC):
    """The abstract software interface base class.

    Attributes:
        theory_level: The level of theory that the software applies in
            energy and force calculations.
    """
    theory_level: TheoryLevel

    @abstractmethod
    def compute_energy(self) -> float:
        r"""Compute the energy of the system using the external software.

        Returns:
            The energy (:math:`\mathrm{kJ\;mol^{-1}}`) of the system.
        """

    @abstractmethod
    def compute_forces(self) -> NDArray[np.float64]:
        r"""Compute the forces on the system using the external software.

        Returns:
            The forces (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`) acting
            on atoms in the system.
        """

    @abstractmethod
    def compute_components(self) -> dict[str, float]:
        r"""Compute the components of energy using the external software.

        Returns:
            The components of the energy (:math:`\mathrm{kJ\;mol^{-1}}`)
            of the system.
        """

    @abstractmethod
    def update_threads(self, threads: int) -> None:
        """Set the number of threads used by the external software.

        Args:
            threads: The number of threads to utilize.
        """

    @abstractmethod
    def update_memory(self, memory: str) -> None:
        """Set the amount of memory used by the external software.

        Args:
            memory: The amount of memory to utilize.
        """


class MMInterface(SoftwareInterface):
    """The abstract MM interface base class.

    Attributes:
        theory_level: The level of theory that the software applies in
            energy and force calculations.
    """
    theory_level = TheoryLevel.MM

    @abstractmethod
    def zero_intramolecular(self, atoms: frozenset[int]) -> None:
        """Remove intra-molecular interactions for the specified atoms.

        Args:
            atoms: The indices of atoms to remove intra-molecular
                interactions from.
        """

    @abstractmethod
    def zero_charges(self, atoms: frozenset[int]) -> None:
        """Remove charges from the specified atoms.

        Args:
            atoms: The indices of atoms to remove charges from.
        """

    @abstractmethod
    def zero_intermolecular(self, atoms: frozenset[int]) -> None:
        """Remove inter-molecular interactions for the specified atoms.

        Args:
            atoms: The indices of atoms to remove inter-molecular
                interactions from.
        """

    @abstractmethod
    def zero_forces(self, atoms: frozenset[int]) -> None:
        """Zero forces on the specified atoms.

        Args:
            atoms: The indices of atoms to zero forces for.
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
            atoms: The indices of atoms to add a Coulomb interaction
                for.
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
            atoms: The indices of atoms to add a non-electrostatic,
                non-bonded interaction for.
            inclusion: An Nx3 array with values that will be applied to
                the forces of the non-electrostatic interaction through
                element-wise multiplication.
        """


class QMInterface(SoftwareInterface):
    """The abstract QM interface base class.

    Attributes:
        theory_level: The level of theory that the software applies in
            energy and force calculations.
    """
    theory_level = TheoryLevel.QM

    @abstractmethod
    def disable_embedding(self) -> None:
        """Disable electrostatic embedding.
        """
