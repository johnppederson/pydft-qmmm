"""The core object which organizes simulations.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .logger import NullLogger
from pydft_qmmm.calculators import Calculator
from pydft_qmmm.calculators import CompositeCalculator
from pydft_qmmm.calculators import InterfaceCalculator
from pydft_qmmm.common import ELEMENT_TO_MASS
from pydft_qmmm.common import ResourceManager
from pydft_qmmm.hamiltonians import CalculatorHamiltonian
from pydft_qmmm.hamiltonians import CompositeHamiltonian
from pydft_qmmm.hamiltonians import QMMMHamiltonian
from pydft_qmmm.plugins import CalculatorCenter
from pydft_qmmm.plugins import CalculatorWrap
from pydft_qmmm.plugins import CentroidPartition
from pydft_qmmm.plugins import Stationary
from pydft_qmmm.plugins.plugin import PartitionPlugin

if TYPE_CHECKING:
    from typing import Any
    from pydft_qmmm.hamiltonians import Hamiltonian
    from pydft_qmmm.integrators import Integrator
    from pydft_qmmm.plugins.plugin import Plugin
    from .system import System


class Simulation:
    """Manages and performs simulations.

    Args:
        system: The system to simulate.
        integrator: The integrator defining how the system evolves.
        hamiltonian: The Hamiltonian defining how calculations will be
            performed.
        calculator: A user-defined calculator, which may be provided in
            place of a Hamiltonian object.
        plugins: A list of plugins to apply to the calculator or
            integrator objects before simulation.
        logger: A logger to record data generated during the
            simulation.

    Attributes:
        _frame: The current frame of the simulation, defaults to zero
            upon instantiation.
    """
    _frame: int = 0

    def __init__(
            self,
            system: System,
            integrator: Integrator,
            hamiltonian: Hamiltonian | None = None,
            calculator: Calculator | None = None,
            plugins: list[Plugin] | None = None,
            logger: Any = NullLogger(),
    ) -> None:
        # Make initial assignments.
        self.system = system
        self.integrator = integrator
        self.hamiltonian = hamiltonian
        if isinstance(calculator, Calculator):
            self.calculator = calculator
        elif (
            isinstance(hamiltonian, CompositeHamiltonian)
            or isinstance(hamiltonian, CalculatorHamiltonian)
        ):
            self.calculator = hamiltonian.build_calculator(self.system)
        else:
            raise TypeError
        if plugins is None:
            plugins = []
        self.plugins = plugins
        self.logger = logger
        # Perform additional simulation setup
        self._register_plugins()
        self._offset = np.zeros(self.system.positions.shape)
        if isinstance(self.hamiltonian, CompositeHamiltonian):
            for hamiltonian in self.hamiltonian.hamiltonians:
                if (
                    isinstance(hamiltonian, QMMMHamiltonian)
                    and self.system.box.any()
                    and isinstance(self.calculator, CompositeCalculator)
                ):
                    query = "not ("
                    for plugin in self.plugins:
                        if isinstance(plugin, PartitionPlugin):
                            query += plugin._query + " or "
                    query = query.strip(" or ") + ")"
                    if query == "not ()":
                        query == "all"
                    self.calculator.register_plugin(CentroidPartition(query))
                    self.calculator.register_plugin(CalculatorWrap())
                    self.calculator.register_plugin(CalculatorCenter())
        calculators = []
        if isinstance(self.calculator, InterfaceCalculator):
            calculators.append(self.calculator)
        elif isinstance(self.calculator, CompositeCalculator):
            for calculator in self.calculator.calculators:
                if isinstance(calculator, InterfaceCalculator):
                    calculators.append(calculator)
        if system.masses[system.masses == 0].size > 0:
            query = "atom"
            for atom in np.where(system.masses.base == 0)[0]:
                query += f" {atom}"
                system.masses[atom] = ELEMENT_TO_MASS.get(
                    system.elements[atom],
                    0.1,
                )
            self.integrator.register_plugin(Stationary(query))
        self._resources = ResourceManager(calculators)
        self.calculate_energy_forces()

    def run_dynamics(self, steps: int) -> None:
        """Perform a number of simulation steps.

        Args:
            steps: The number of steps to take.
        """
        with self.logger as logger:
            logger.record(self)
            for i in range(steps):
                new_positions, new_velocities = self.integrator.integrate(
                    self.system,
                )
                self.system.positions = new_positions
                self.system.velocities = new_velocities
                self.calculate_energy_forces()
                self._frame += 1
                logger.record(self)

    def calculate_energy_forces(self) -> None:
        """Update total system energy and forces on atoms in the system.
        """
        temp = self.system.positions.base.copy()
        results = self.calculator.calculate()
        self.system.forces = results.forces
        kinetic_energy = self.integrator.compute_kinetic_energy(
            self.system,
        )
        energy = {
            "Total Energy": kinetic_energy + results.energy,
            ".": {
                "Kinetic Energy": kinetic_energy,
                "Potential Energy": results.energy,
                ".": results.components,
            },
        }
        self.energy = energy
        self._offset += temp - self.system.positions.base

    def set_threads(self, threads: int) -> None:
        """Set the number of threads that calculators can use.

        Args:
            threads: The number of threads to utilize.
        """
        self._resources.update_threads(threads)

    def set_memory(self, memory: str) -> None:
        """Set the amount of memory that calculators can use.

        Args:
            memory: The amount of memory to utilize.
        """
        self._resources.update_memory(memory)

    def _register_plugins(self) -> None:
        """Dynamically load plugins for calculators and integrators.
        """
        for plugin in self.plugins:
            getattr(self, plugin._key).register_plugin(plugin)
