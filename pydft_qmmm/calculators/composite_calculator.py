#! /usr/bin/env python3
"""A module to define the :class:`QMMMCalculator` class.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from .calculator import Calculator
from pydft_qmmm.common import Results

if TYPE_CHECKING:
    from pydft_qmmm import System
    from pydft_qmmm.common import Components


@dataclass
class CompositeCalculator(Calculator):
    """A :class:`Calculator` class for performing QM/MM calculations for
    an entire system.

    :param system: |system| to perform calculations on.
    :param calculators: The subsystem :class:`Calculators` to perform
        calculations with.
    :param embedding_cutoff: |embedding_cutoff|
    :param options: Options to provide to either of the
        :class:`SoftwareInterface` objects.
    """
    system: System
    calculators: list[Calculator]
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.calculation_sequence = dict()
        for i, calculator in enumerate(self.calculators):
            self.calculation_sequence[f"{calculator.name}_{i}"] = calculator

    def calculate(
            self,
            return_forces: bool | None = True,
            return_components: bool | None = True,
    ) -> Results:
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
