"""Plugins for centering coordinates.
"""
from __future__ import annotations

__all__ = [
    "CalculatorCenter",
    "IntegratorCenter",
]

from typing import TYPE_CHECKING

import numpy as np

from pydft_qmmm.calculators.calculator import CalculatorPlugin
from pydft_qmmm.integrators.integrator import IntegratorPlugin

if TYPE_CHECKING:
    from collections.abc import Callable
    from numpy.typing import NDArray
    from pydft_qmmm.integrators import Returns
    from pydft_qmmm.calculators import Results
    from pydft_qmmm import System


def _center_positions(
        positions: NDArray[np.float64],
        system: System,
        query: str,
) -> NDArray[np.float64]:
    r"""Center positions about the centroid of a query selection.

    Args:
        positions: The positions (:math:`\mathrm{\mathring{A}}`) to
            center.
        system: The system whose positions will be centered.
        query: The VMD-like query representing the group of atoms whose
            centroid will be taken to be the center of the system.

    Returns:
        The new centered positions of the system.
    """
    atoms = sorted(system.select(query))
    box = system.box
    center = 0.5*box.sum(axis=1)
    centroid = np.average(positions[atoms, :], axis=0)
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
            self.calculator.system.positions[:] = _center_positions(
                self.calculator.system.positions,
                self.calculator.system,
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
