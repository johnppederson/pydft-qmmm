"""A calculator that performs and collates sub-calculations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .calculator import Calculator
from pydft_qmmm.common import Results

if TYPE_CHECKING:
    from pydft_qmmm import System
    from pydft_qmmm.common import Components
    from pydft_qmmm.plugins.plugin import CalculatorPlugin
    from pydft_qmmm.plugins.plugin import CompositeCalculatorPlugin


@dataclass
class CompositeCalculator(Calculator):
    """A calculator that performs and collates sub-calculations.

    Args:
        system: The system whose atom positions, atom identities, and
            geometry will be used to calculate energies and forces.
        calculators: The calculators that will perform sub-calculations.
        cutoff: The cutoff between regions treated with different
            levels of theory, comprising the embedding region.
    """
    system: System
    calculators: list[Calculator]
    cutoff: float | int = 0.

    def __post_init__(self) -> None:
        """Determine the sequence in which to perform sub-calculations.
        """
        self.calculation_sequence = dict()
        for i, calculator in enumerate(self.calculators):
            self.calculation_sequence[f"{calculator.name}_{i}"] = calculator

    def calculate(
            self,
            return_forces: bool | None = True,
            return_components: bool | None = True,
    ) -> Results:
        r"""Calculate energies and forces.

        Args:
            return_forces: Whether or not to return forces.
            return_components: Whether or not to return the components of
                the energy.

        Returns:
            The energy (:math:`\mathrm{kJ\;mol^{-1}}`), forces
            (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`), and energy
            components (:math:`\mathrm{kJ\;mol^{-1}}`) of the
            calculation.
        """
        energy = 0.
        forces = np.zeros(self.system.forces.shape)
        components: Components = dict()
        for i, (name, calculator) in enumerate(
                self.calculation_sequence.items(),
        ):
            results = calculator.calculate()
            energy += results.energy
            forces += results.forces
            components[name] = results.energy
            components["."*(i + 1)] = results.components
        results = Results(energy, forces, components)
        return results

    def register_plugin(
            self,
            plugin: CalculatorPlugin | CompositeCalculatorPlugin,
    ) -> None:
        """Record plugin name and apply the plugin to the calculator.

        Args:
            plugin: A plugin that will modify the behavior of one or
                more calculator routines.
        """
        self._plugins.append(type(plugin).__name__)
        plugin.modify(self)
