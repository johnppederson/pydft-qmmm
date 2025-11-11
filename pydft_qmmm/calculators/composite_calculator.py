"""A calculator class that performs and collates sub-calculations.

This module contains the composite calculator class as well as a
couple calculator plugin subclasses for the composite calculator.
"""
from __future__ import annotations

__all__ = [
    "CompositeCalculator",
    "CompositeCalculatorPlugin",
    "PartitionPlugin",
]

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .calculator import Calculator
from .calculator import CalculatorPlugin
from .calculator import Results
from pydft_qmmm.utils import pluggable_method

if TYPE_CHECKING:
    from .calculator import Components
    from pydft_qmmm import System  # noqa: F401


@dataclass(frozen=True)
class CompositeCalculator(Calculator):
    """A calculator that performs and collates sub-calculations.

    Args:
        system: The system whose atom positions, atom identities, and
            geometry will be used to calculate energies and forces.
        calculators: The calculators that will perform sub-calculations.
    """
    calculators: list[Calculator]

    @pluggable_method
    def calculate(
            self,
            return_forces: bool = True,
            return_components: bool = True,
    ) -> Results:
        r"""Calculate energies and forces.

        Args:
            return_forces: Whether or not to return forces.
            return_components: Whether or not to return the components
                of the energy.

        Returns:
            A wrapper containing the energy
            (:math:`\mathrm{kJ\;mol^{-1}}`), forces
            (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`), and
            energy components (:math:`\mathrm{kJ\;mol^{-1}}`) of the
            calculation.
        """
        energy = 0.
        forces = np.zeros(self.system.forces.shape)
        components: Components = dict()
        for i, calculator in enumerate(self.calculators):
            # Calculate the energy, forces, and components.
            results = calculator.calculate(
                return_forces,
                return_components,
            )
            energy += results.energy
            if return_forces:
                forces += results.forces
            # Determine a unique name for the calculator.
            name = calculator.name
            suffix = "0"
            while name in components:
                suffix = str(int(suffix) + 1)
                name = calculator.name + suffix
            # Assign the components appropriately.
            components[name] = results.energy
            components["."*(i + 1)] = results.components
        results = Results(energy, forces, components)
        return results

    @property
    def name(self) -> str:
        """The name of the calculator, for logging purposes.
        """
        name = "Composite[ "
        for calculator in self.calculators:
            name += calculator.name + " "
        return name + "]"


class CompositeCalculatorPlugin(CalculatorPlugin):
    """The plugin base class for modifying composite calculator routines.
    """

    def modify(self, calculator: Calculator) -> None:
        """Modify the functionality of a calculator.

        Args:
            calculator: The calculator whose functionality will be
                modified by the plugin.
        """
        if isinstance(calculator, CompositeCalculator):
            self.modify_composite(calculator)
        else:
            raise TypeError  # Todo: Make this informative.

    def modify_composite(self, calculator: CompositeCalculator) -> None:
        """Modify the functionality of a composite calculator.

        Args:
            calculator: The composite calculator whose functionality
                will be modified by the plugin.
        """
        self.calculator = calculator


class PartitionPlugin(CompositeCalculatorPlugin):
    r"""The plugin base class for performing partitioning routines.

    Attributes:
        query: The VMD-like query representing atoms which will be
            evaluated with the partitioning scheme.
        cutoff: The cutoff distance (:math:`\mathrm{\mathring{A}}`) to
            apply in the partition.
    """
    query: str = ""
    cutoff: int | float = 0

    @abstractmethod
    def generate_partition(self) -> None:
        """Perform the system partitioning.
        """
