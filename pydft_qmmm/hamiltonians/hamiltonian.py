#! /usr/bin/env python3
"""A module defining the base :class:`Hamiltonian` class and derived
interface classes.
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
    """An abstract :class:`Hamiltonian` base class for creating the
    Hamiltonian API.
    """

    @abstractmethod
    def __add__(self, other: Any) -> Hamiltonian:
        """Add :class:`Hamiltonian` objects together.

        :param other: The object being added to the
            :class:`Hamiltonian`.
        :return: A new :class:`Hamiltonian` object.
        """

    def __radd__(self, other: Any) -> Any:
        """Add :class:`Hamiltonian` objects together.

        :param other: The object being added to the
            :class:`Hamiltonian`.
        :return: A new :class:`Hamiltonian` object.
        """
        return self.__add__(other)

    @abstractmethod
    def __str__(self) -> str:
        """Create a LATEX string representation of the
        :class:`Hamiltonian` object.

        :return: The string representation of the :class:`Hamiltonian`
            object.
        """


class CalculatorHamiltonian(Hamiltonian):
    """An abstract :class:`Hamiltonian` base class for creating
    Hamiltonians which can create standalone calculators.
    """
    atoms: list[int | slice] = []
    theory_level: TheoryLevel = TheoryLevel.NO

    def __getitem__(
            self,
            indices: int | slice | tuple[int | slice, ...],
    ) -> Hamiltonian:
        """Sets the indices for atoms that are treated with this
        :class:`Hamiltonian`.

        :return: |Hamiltonian|.
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

        :param system: |system| to calculate energy and forces for.
        :return: |atoms|
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

    def __add__(self, other: Hamiltonian) -> CompositeHamiltonian:
        if not isinstance(other, Hamiltonian):
            raise TypeError("...")
        return CompositeHamiltonian(self, other)

    def __str__(self) -> str:
        string = "_{"
        for atom in self.atoms:
            string += f"{atom}, "
        string += "}"
        return string

    @abstractmethod
    def build_calculator(self, system: System) -> Calculator:
        """Build the :class:`Calculator` corresponding to the
        :class:`Hamiltonian` object.

        :param system: |system| to calculate energy and forces for.
        :return: |calculator|.
        """


class CouplingHamiltonian(Hamiltonian):
    """An abstract :class:`Hamiltonian` base class for creating
    Hamiltonians which couple the interactions between subsystems
    modeled by :class:`StandaloneHamiltonian` objects.
    """
    force_matrix: dict[Subsystem, dict[Subsystem, TheoryLevel]]

    def __add__(self, other: Hamiltonian) -> CompositeHamiltonian:
        if not isinstance(other, Hamiltonian):
            raise TypeError("...")
        return CompositeHamiltonian(self, other)

    @abstractmethod
    def modify_calculator(self, calculator: Calculator, system: System) -> None:
        """
        """


class CompositeHamiltonian(Hamiltonian):
    """An abstract :class:`Hamiltonian` base class for creating
    Hamiltonians which contain subsystems modeled by
    :class:`StandaloneHamiltonian` objects and their respective
    :class:`CouplingHamiltonian` objects.
    """

    def __init__(self, *hamiltonians: Hamiltonian) -> None:
        self.hamiltonians = hamiltonians

    def __add__(self, other: Hamiltonian) -> CompositeHamiltonian:
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
        string = "H^{Total} ="
        for hamiltonian in self.hamiltonians:
            string += " " + str(hamiltonian)
        return string

    def build_calculator(self, system: System) -> Calculator:
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
        return calculator

    def _calculator_hamiltonians(self) -> list[CalculatorHamiltonian]:
        """
        """
        standalone = []
        for hamiltonian in self.hamiltonians:
            if isinstance(hamiltonian, CalculatorHamiltonian):
                standalone.append(hamiltonian)
        return standalone

    def _coupling_hamiltonians(self) -> list[CouplingHamiltonian]:
        """
        """
        coupling = []
        for hamiltonian in self.hamiltonians:
            if isinstance(hamiltonian, CouplingHamiltonian):
                coupling.append(hamiltonian)
        return coupling
