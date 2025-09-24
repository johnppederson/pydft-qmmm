"""The class for organizing and running optimizations.
"""
from __future__ import annotations

__all__ = ["Optimization"]

import logging
import os
import tempfile
from typing import TYPE_CHECKING

import numpy as np

from pydft_qmmm.calculators import CalculatorPlugin
from pydft_qmmm.utils import Loggable
from pydft_qmmm.utils import DependencyImportError
from pydft_qmmm.plugins import CalculatorCenter
from pydft_qmmm.plugins import CalculatorWrap

if TYPE_CHECKING:
    from typing import Any
    from pydft_qmmm.calculators import Calculator
    from pydft_qmmm.hamiltonians import StandaloneHamiltonian
    from pydft_qmmm.system import System

logger = logging.getLogger(__name__)
logger.setLevel(20)


class Optimization(Loggable):
    """Organizes and performs optimizations.

    Args:
        query: A VMD-like selection query describing the atoms to
            optimize.
        system: The system containing the atoms to optimize.
        hamiltonian: The Hamiltonian describing how calculations are
            performed.
        calculator: A user-defined energy and force calculator, which
            may be provided in place of a Hamiltonian object.
        plugins: A list of plugins to apply to the calculator object
            before optimization.
        kwargs: Additional options to provide to the logging
            functionality, see :class:`pydft_qmmm.utils.logging.Loggable`
            for a list of options.

    Attributes:
        _frame: The current frame of the optimization, defaults to zero
            upon instantiation.
    """
    _frame: int = 0

    def __init__(
            self,
            query: str,
            system: System,
            hamiltonian: StandaloneHamiltonian | None = None,
            calculator: Calculator | None = None,
            plugins: list[CalculatorPlugin] | None = None,
            **kwargs: Any,
    ) -> None:
        # Perform logging setup.
        super().__init__(**kwargs)
        for handler in self.handlers:
            logger.addHandler(handler)
        # Set calculator and build if necessary.
        if calculator is not None:
            self.calculator = calculator
        elif hamiltonian is not None:
            self.calculator = hamiltonian.build_calculator(system)
        else:
            raise TypeError
        # Perform additional simulation setup.
        self.query = query
        self._offset = np.zeros(system.positions.shape)
        # Todo: See if this is desirable or necessary.
        if system.box.any():
            self.calculator.register_plugin(CalculatorWrap(), 0)
        if system.select("subsystem I"):
            self.calculator.register_plugin(CalculatorCenter(), 0)
        self.system = system
        # Apply plugins.
        if plugins is not None:
            self._register_plugins(plugins)

    def optimize(self) -> None:
        """Perform the optimization using geomeTRIC."""
        try:
            import geometric
            import geometric.molecule
        except ImportError:
            raise DependencyImportError(
                "geomeTRIC",
                "performing geometry optimizations",
                "https://github.com/leeping/geomeTRIC",
            )
        opt_indices = sorted(self.system.select(self.query))

        # Define objective function.
        def model(x, *args):
            # geomeTRIC provides coordinates in a.u.
            y = x/1.88973
            self.system.positions[opt_indices, :] = (
                y - self._offset[opt_indices, :]
            )
            self.calculate_energy_forces()
            # geomeTRIC wants energies and gradients in a.u.
            e = self.energy["Total Energy"] / 2625.5
            G = -self.system.forces / 2625.5 / 1.88973
            self._frame += 1
            return e, G[opt_indices, :]

        # Define Geometric engine.
        class CustomEngine(geometric.engine.Engine):
            def __init__(self, molecule):
                super(CustomEngine, self).__init__(molecule)

            def calc_new(self, coords, dirname):
                energy, gradient = model(coords.reshape(-1, 3))
                return {'energy': energy, 'gradient': gradient.ravel()}

        # Instantiate optimizer.
        molecule = geometric.molecule.Molecule()
        molecule.elem = self.system.elements[np.ix_(opt_indices)].tolist()
        molecule.xyzs = [self.system.positions[opt_indices, :]]
        customengine = CustomEngine(molecule)

        # Run the optimizer.
        fd, tmp_path = tempfile.mkstemp(suffix=".tmp", text=True)
        os.close(fd)

        try:
            m = geometric.optimize.run_optimizer(
                customengine=customengine,
                check=1,
                input=tmp_path,
            )
        finally:
            os.remove(tmp_path)

        # Set positions ot the optimized positions.
        self.system.positions[opt_indices, :] = m.xyzs[-1]

    def calculate_energy_forces(self) -> None:
        """Update total system energy and forces on atoms in the system.
        """
        unwrapped_positions = self.system.positions.base + self._offset
        logger.info(
            "",
            extra={"frame": self._frame,
                   "positions": unwrapped_positions,
                   "box": self.system.box.base},
        )
        temp = self.system.positions.base.copy()
        results = self.calculator.calculate()
        self.system.forces = results.forces
        energy = {
            "Total Energy": results.energy,
            ".": results.components,
        }
        self.energy = energy
        self._offset += temp - self.system.positions.base
        logger.info("", extra={"frame": self._frame, "energy": self.energy})

    def _register_plugins(self, plugins: list[CalculatorPlugin]) -> None:
        """Dynamically load plugins for calculators.

        Args:
            plugins: A list of plugins to apply to the calculator
                object before optimization.
        """
        for plugin in plugins:
            if isinstance(plugin, CalculatorPlugin):
                self.calculator.register_plugin(plugin)
            else:
                raise TypeError
