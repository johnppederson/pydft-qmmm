"""A plugin interface to the Plumed enhanced sampling suite.
"""
from __future__ import annotations

from typing import Callable
from typing import TYPE_CHECKING

import numpy as np

from pydft_qmmm.common import lazy_load
from pydft_qmmm.plugins.plugin import CalculatorPlugin

if TYPE_CHECKING:
    from qmmm_pme.calculators import Calculator
    from pydft_qmmm.common import Results
    import mypy_extensions
    CalculateMethod = Callable[
        [
            mypy_extensions.DefaultArg(
                bool | None,
                "return_forces",  # noqa: F821
            ),
            mypy_extensions.DefaultArg(
                bool | None,
                "return_components",  # noqa: F821
            ),
        ],
        Results,
    ]


class Plumed(CalculatorPlugin):
    """Apply enhanced sampling biases to energy and force calculations.

    Args:
        input_commands: A multi-line string containing all pertinent
            instructions for Plumed.
        log_file: A directory for recording output from Plumed.
    """

    def __init__(
            self,
            input_commands: str,
            log_file: str,
    ) -> None:
        plumed = lazy_load("plumed")
        self.input_commands = input_commands
        self.log_file = log_file
        self.plumed = plumed.Plumed()
        self.plumed.cmd("setMDEngine", "python")
        self.frame = 0

    def modify(
            self,
            calculator: Calculator,
    ) -> None:
        """Modify the functionality of a calculator and set up Plumed.

        Args:
            calculator: The calculator whose functionality will be
                modified by the plugin.
        """
        self._modifieds.append(type(calculator).__name__)
        self.system = calculator.system
        self.plumed.cmd("setNatoms", len(self.system))
        self.plumed.cmd("setMDLengthUnits", 1/10)
        self.plumed.cmd("setMDTimeUnits", 1/1000)
        self.plumed.cmd("setMDMassUnits", 1.)
        self.plumed.cmd("setTimestep", 1.)
        self.plumed.cmd("setKbT", 1.)
        self.plumed.cmd("setLogFile", self.log_file)
        self.plumed.cmd("init")
        for line in self.input_commands.split("\n"):
            self.plumed.cmd("readInputLine", line)
        calculator.calculate = self._modify_calculate(calculator.calculate)

    def _modify_calculate(
            self,
            calculate: CalculateMethod,
    ) -> CalculateMethod:
        """Modify the calculate routine to perform biasing afterward.

        Args:
            calculate: The calculation routine to modify.

        Returns:
            The modified calculation routine which implements Plumed
            enhanced sampling after performing the unbiased calculation
            routine.
        """
        def inner(
                return_forces: bool | None = True,
                return_components: bool | None = True,
        ) -> Results:
            results = calculate(return_forces, return_components)
            self.plumed.cmd("setStep", self.frame)
            self.frame += 1
            self.plumed.cmd("setBox", self.system.box.T)
            self.plumed.cmd("setPositions", self.system.positions)
            self.plumed.cmd("setEnergy", results.energy)
            self.plumed.cmd("setMasses", self.system.masses)
            biased_forces = np.zeros(self.system.positions.shape)
            self.plumed.cmd("setForces", biased_forces)
            virial = np.zeros((3, 3))
            self.plumed.cmd("setVirial", virial)
            self.plumed.cmd("prepareCalc")
            self.plumed.cmd("performCalc")
            biased_energy = np.zeros((1,))
            self.plumed.cmd("getBias", biased_energy)
            results.energy += biased_energy[0]
            results.forces += biased_forces
            results.components.update(
                {"Plumed Bias Energy": biased_energy[0]},
            )
            return results
        return inner
