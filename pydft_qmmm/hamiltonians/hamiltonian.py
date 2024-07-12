"""Hamiltonian base and derived classes.
"""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from typing import Any
from typing import TYPE_CHECKING

from pydft_qmmm.calculators import CompositeCalculator
from pydft_qmmm.common import Subsystem
from pydft_qmmm.common import TheoryLevel

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
        """Create a LATEX string representation of the Hamiltonian.

        Returns:
            The string representation of the Hamiltonian.
        """


class CalculatorHamiltonian(Hamiltonian):
    """An abstract Hamiltonian base class for creating calculators.

    Attributes:
        atoms: Indices corresponding to the atoms for which the
            Hamiltonian is applicable.
        theory_level: The level of theory of the Hamiltonian.
    """
    atoms: list[int | slice] = []
    theory_level: TheoryLevel = TheoryLevel.NO

    def __getitem__(
            self,
            indices: int | slice | tuple[int | slice, ...],
    ) -> Hamiltonian:
        """Sets the indices for atoms treated by the Hamiltonian.

        Args:
            indices: Indices corresponding to the atoms for which the
                Hamiltonian is applicable.

        Returns:
            A copy of the Hamiltonian with the selected atoms.
        """
        indices = indices if isinstance(indices, tuple) else (indices,)
        atoms = []
        for i in indices:
            if isinstance(i, (int, slice)):
                atoms.append(i)
            else:
                raise TypeError("...")
        ret = deepcopy(self)
        ret.atoms = atoms
        return ret

    def _parse_atoms(self, system: System) -> list[int]:
        """Parse the indices provided to the :class:`Hamiltonian` object
        to create the list of residue-grouped atom indices.

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
        """Create a LATEX string representation of the Hamiltonian.

        Returns:
            The string representation of the Hamiltonian.
        """
        string = "_{"
        for atom in self.atoms:
            string += f"{atom}, "
        string += "}"
        return string

    @abstractmethod
    def build_calculator(self, system: System) -> Calculator:
        """Build the calculator corresponding to the Hamiltonian.

        Args:
            system: The system that will be used to calculate the
                calculator.

        Returns:
            The calculator which is defined by the system and the
            Hamiltonian.
        """


class CouplingHamiltonian(Hamiltonian):
    """An abstract Hamiltonian base class for coupling Hamiltonians.

    Attributes:
        force_matrix: A matrix representing the gradient of potential
            expressions representing interactions between differing
            subsystems.
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
    def modify_calculator(self, calculator: Calculator, system: System) -> None:
        """Modify a calculator to represent the coupling.

        Args:
            calculator: A calculator which is defined in part by the
                system.
            system: The system that will be used to modify the
                calculator.
        """

    @abstractmethod
    def modify_composite(
            self,
            calculator: CompositeCalculator,
            system: System,
    ) -> None:
        """Modify a composite calculator to represent the coupling.

        Args:
            calculator: A composite calculator which is defined in part
                by the system.
            system: The system that will be used to modify the
                calculator.
        """


class CompositeHamiltonian(Hamiltonian):
    """An abstract Hamiltonian base class for combining Hamiltonians.

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
            system: The system that will be used to calculate the
                calculator.

        Returns:
            The calculator which is defined by the system and the
            Hamiltonian.
        """
        standalone = self._calculator_hamiltonians()
        coupling = self._coupling_hamiltonians()
        calculators = []
        for hamiltonian in standalone:
            calculator = hamiltonian.build_calculator(system)
            for coupler in coupling:
                coupler.modify_calculator(calculator, system)
            calculators.append(calculator)
        calculator = CompositeCalculator(
            system=system,
            calculators=calculators,
        )
        for coupler in coupling:
            coupler.modify_composite(calculator, system)
        return calculator

    def _calculator_hamiltonians(self) -> list[CalculatorHamiltonian]:
        """Sort out calculator-building Hamiltonians.

        Returns:
            A list of calculator-building Hamiltonians.
        """
        standalone = []
        for hamiltonian in self.hamiltonians:
            if isinstance(hamiltonian, CalculatorHamiltonian):
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
