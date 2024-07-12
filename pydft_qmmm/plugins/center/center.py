"""Plugins for centering coordinates.
"""
from __future__ import annotations

from typing import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from pydft_qmmm.plugins.plugin import CalculatorPlugin
from pydft_qmmm.plugins.plugin import IntegratorPlugin

if TYPE_CHECKING:
    from pydft_qmmm.integrator import Integrator
    from pydft_qmmm.integrator import Returns
    from pydft_qmmm.calculator import Calculator
    from pydft_qmmm.common import Results
    from pydft_qmmm import System


def _center_positions(
        positions: NDArray[np.ndarray],
        system: System,
        query: str,
) -> NDArray[np.float64]:
    r"""Center positions about the centroid of a query selection.

    Args:
        positions: The positions (:math:`\mathrm{\mathring{A}}`) which will be
            centered.
        system: The system whose positions will be centered.
        query: The VMD-like query representing the group of atoms whose
            centroid will be taken to be the center of the system.

    Returns:
        The new centered positions of the system.
    """
    atoms = system.select(query)
    box = system.box
    center = 0.5*box.sum(axis=0)
    centroid = np.average(positions[list(atoms), :], axis=0)
    differential = center - centroid
    new_positions = positions + differential
    return new_positions


class CalculatorCenter(CalculatorPlugin):
    """Center positions before performing a calculation.

    Args:
        query: The VMD-like query representing the group of atoms whose
            centroid will be taken to be the center of the system.
    """

    def __init__(
            self,
            query: str = "subsystem I",
    ) -> None:
        self.query = query

    def modify(
            self,
            calculator: Calculator,
    ) -> None:
        """Modify the functionality of a calculator.

        Args:
            calculator: The calculator whose functionality will be
                modified by the plugin.
        """
        self._modifieds.append(type(calculator).__name__)
        self.system = calculator.system
        calculator.calculate = self._modify_calculate(
            calculator.calculate,
        )

    def _modify_calculate(
            self,
            calculate: Callable[[bool, bool], Results],
    ) -> Callable[[bool, bool], Results]:
        """Modify the calculate routine to perform centering beforehand.

        Args:
            calculate: The calculation routine to modify.

        Returns:
            The modified calculation routine which implements the
            coordinate-centering before calculation.
        """
        def inner(
                return_forces: bool = True,
                return_components: bool = True,
        ) -> Results:
            self.system.positions = _center_positions(
                self.system.positions,
                self.system,
                self.query,
            )
            results = calculate(return_forces, return_components)
            return results
        return inner


class IntegratorCenter(IntegratorPlugin):
    """Center positions after performing an integration.

    Args:
        query: The VMD-like query representing the group of atoms whose
            centroid will be taken to be the center of the system.
    """

    def __init__(
            self,
            query: str = "subsystem I",
    ) -> None:
        self.query = query

    def modify(
            self,
            integrator: Integrator,
    ) -> None:
        """Modify the functionality of an integrator.

        Args:
            integrator: The integrator whose functionality will be
                modified by the plugin.
        """
        self._modifieds.append(type(integrator).__name__)
        self.integrator = integrator
        integrator.integrate = self._modify_integrate(
            integrator.integrate,
        )

    def _modify_integrate(
            self,
            integrate: Callable[[System], Returns],
    ) -> Callable[[System], Returns]:
        """Modify the integrate routine to perform centering afterward.

        Args:
            integrate: The integration routine to modify.

        Returns:
            The modified integration routine which implements the
            coordinate-centering after integration.
        """
        def inner(system: System) -> Returns:
            positions, velocities = integrate(system)
            positions = _center_positions(positions, system, self.query)
            return positions, velocities
        return inner
