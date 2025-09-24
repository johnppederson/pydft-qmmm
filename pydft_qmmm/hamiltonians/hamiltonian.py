"""Base classes for the Hamiltonian API.

This module contains the Hamiltonian abstract base class as well as
several derived classes for standalone Hamiltonians, Hamiltonians that
couple other standalone Hamiltonians, and composite Hamiltonians.
"""
from __future__ import annotations

__all__ = [
    "Hamiltonian",
    "StandaloneHamiltonian",
    "PotentialHamiltonian",
    "CouplingHamiltonian",
    "CompositeHamiltonian",
]

from abc import ABC
from abc import abstractmethod
from collections.abc import Iterable
from copy import deepcopy
from typing import Any
from typing import TYPE_CHECKING

from pydft_qmmm.calculators import CompositeCalculator
from pydft_qmmm.utils import Subsystem
from pydft_qmmm.utils import TheoryLevel

if TYPE_CHECKING:
    from pydft_qmmm import System
    from pydft_qmmm.calculators import Calculator


class Hamiltonian(ABC):
    """The abstract Hamiltonian base class.
    """

    @abstractmethod
    def __add__(self, other: Any) -> Hamiltonian:
        """Add Hamiltonians together.

        Args:
            other: The object being added to the Hamiltonian.

        Returns:
            A new Hamiltonian.
        """

    def __radd__(self, other: Any) -> Any:
        """Add Hamiltonians together.

        Args:
            other: The object being added to the Hamiltonian.

        Returns:
            A new Hamiltonian.
        """
        return self.__add__(other)

    @abstractmethod
    def __str__(self) -> str:
        """Create a LaTeX string representation of the Hamiltonian.

        Returns:
            The string representation of the Hamiltonian.
        """


class StandaloneHamiltonian(Hamiltonian):
    """A base class for Hamiltonians that build calculators.
    """

    @abstractmethod
    def build_calculator(self, system: System) -> Calculator:
        """Build the calculator corresponding to the Hamiltonian.

        Args:
            system: The system that will be used in calculations.

        Returns:
            The calculator which is defined by the system and the
            Hamiltonian.
        """


class CouplingHamiltonian(Hamiltonian):
    """A base class for Hamiltonians that couple other Hamiltonians.

    Attributes:
        force_matrix: A matrix representing the gradient of potential
            expressions representing interactions between subsystems
            treated by different Hamiltonians.
    """
    force_matrix: dict[Subsystem, dict[Subsystem, TheoryLevel]]

    def __add__(self, other: Hamiltonian) -> CompositeHamiltonian:
        """Add Hamiltonians together.

        Args:
            other: The object being added to the Hamiltonian.

        Returns:
            A new Hamiltonian.
        """
        if not isinstance(other, Hamiltonian):
            raise TypeError("...")
        return CompositeHamiltonian(self, other)

    @abstractmethod
    def modify_calculator(
            self,
            calculator: CompositeCalculator,
            system: System,
    ) -> None:
        """Modify a composite calculator to include the coupling.

        Args:
            calculator: A composite calculator which is defined in part
                by the system.
            system: The system corresponding to the calculator.
        """


class PotentialHamiltonian(StandaloneHamiltonian):
    """A base class for Hamiltonians that build potential calculators.

    Attributes:
        atoms: Indices corresponding to the atoms for which the
            Hamiltonian is applicable.
        theory_level: The level of theory of the Hamiltonian.
        interface: The name of the software that implements the desired
            level of theory described by the Hamiltonian.
    """
    atoms: tuple[int | slice, ...] = ()
    theory_level: TheoryLevel = TheoryLevel.NO
    interface: str = ""

    def __getitem__(
            self,
            indices: int | slice | Iterable[int | slice],
    ) -> Hamiltonian:
        """Sets the indices for atoms treated by the Hamiltonian.

        Args:
            indices: Indices corresponding to the atoms for which the
                Hamiltonian is applicable.

        Returns:
            A copy of the Hamiltonian with the selected atoms.
        """
        indices = indices if isinstance(indices, Iterable) else (indices,)
        atoms: tuple[int | slice, ...] = ()
        for i in indices:
            if isinstance(i, (int, slice)):
                atoms += (i,)
            else:
                raise TypeError("...")
        ret = deepcopy(self)
        ret.atoms = atoms
        return ret

    def _parse_atoms(self, system: System) -> list[int]:
        """Parse the Hamiltonian's indices in the context of the system.

        Args:
            system: The system whose atoms will be selected by the by
                the Hamiltonian.

        Returns:
            The atoms selected for representation by the Hamiltonian.
        """
        indices = []
        for i in self.atoms:
            if isinstance(i, int):
                indices.append(i)
            else:
                indices.extend(
                    list(
                        range(
                            i.start if i.start else 0,
                            i.stop if i.stop else len(system),
                            i.step if i.step else 1,
                        ),
                    ),
                )
        if not self.atoms:
            indices = [i for i in range(len(system))]
        return indices

    def __add__(self, other: Any) -> CompositeHamiltonian:
        """Add Hamiltonians together.

        Args:
            other: The object being added to the Hamiltonian.

        Returns:
            A new Hamiltonian.
        """
        if not isinstance(other, Hamiltonian):
            raise TypeError("...")
        return CompositeHamiltonian(self, other)

    def __str__(self) -> str:
        """Create a LaTeX string representation of the Hamiltonian.

        Returns:
            The string representation of the Hamiltonian.
        """
        string = "_{"
        for atom in self.atoms:
            string += f"{atom}, "
        string += "}"
        return string


class CompositeHamiltonian(StandaloneHamiltonian):
    """A Hamiltonian that can combine other Hamiltonians.

    Args:
        hamiltonians: A set of Hamiltonians belonging to the composite
            Hamiltonian.
    """

    def __init__(self, *hamiltonians: Hamiltonian) -> None:
        self.hamiltonians = hamiltonians

    def __add__(self, other: Hamiltonian) -> CompositeHamiltonian:
        """Add Hamiltonians together.

        Args:
            other: The object being added to the Hamiltonian.

        Returns:
            A new Hamiltonian.
        """
        if not isinstance(other, Hamiltonian):
            raise TypeError("...")
        if isinstance(other, CompositeHamiltonian):
            ret = CompositeHamiltonian(
                *self.hamiltonians, *other.hamiltonians,
            )
        else:
            ret = CompositeHamiltonian(
                *self.hamiltonians,
                other,
            )
        return ret

    def __str__(self) -> str:
        """Create a LATEX string representation of the Hamiltonian.

        Returns:
            The string representation of the Hamiltonian.
        """
        string = "H^{Total} ="
        for hamiltonian in self.hamiltonians:
            string += " " + str(hamiltonian)
        return string

    def build_calculator(self, system: System) -> Calculator:
        """Build the calculator corresponding to the Hamiltonian.

        Args:
            system: The system that will be used in calculations.

        Returns:
            The calculator which is defined by the system and the
            Hamiltonian.
        """
        standalone = self._calculator_hamiltonians()
        coupling = self._coupling_hamiltonians()
        calculators = []
        # Build first.
        for hamiltonian in standalone:
            calculator = hamiltonian.build_calculator(system)
            calculators.append(calculator)
        calculator = CompositeCalculator(
            system=system,
            calculators=calculators,
        )
        # Modify the built calculator.
        for coupler in coupling:
            coupler.modify_calculator(calculator, system)
        return calculator

    def _calculator_hamiltonians(self) -> list[StandaloneHamiltonian]:
        """Sort out calculator-building Hamiltonians.

        Returns:
            A list of calculator-building Hamiltonians.
        """
        standalone = []
        for hamiltonian in self.hamiltonians:
            if isinstance(hamiltonian, StandaloneHamiltonian):
                standalone.append(hamiltonian)
        return standalone

    def _coupling_hamiltonians(self) -> list[CouplingHamiltonian]:
        """Sort out coupling Hamiltonians.

        Returns:
            A list of coupling Hamiltonians.
        """
        coupling = []
        for hamiltonian in self.hamiltonians:
            if isinstance(hamiltonian, CouplingHamiltonian):
                coupling.append(hamiltonian)
        return coupling
