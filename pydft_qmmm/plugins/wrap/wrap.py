"""Plugins for wrapping system coordinates when a PBC is in use.
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


def _wrap_positions(
        positions: NDArray[np.float64],
        system: System,
) -> NDArray[np.float64]:
    r"""Wrap atom positions in accord with PBC.

    Atoms are wrapped to stay inside of the periodic box.  This
    function ensures molecules are not broken up by a periodic
    boundary, since OpenMM electrostatics will be incorrect if atoms
    in a molecule are not on the same side of the periodic box.
    This method currently assumes an isotropic box.

    Args:
        positions: The positions (:math:`\mathrm{\mathring{A}}`) which will be
            wrapped.
        system: The system whose positions will be wrapped.

    Returns:
        The new wrapped positions of the system.
    """
    box = system.box
    inverse_box = np.linalg.inv(box)
    new_positions = np.zeros(positions.shape)
    for residue in system.residue_map.values():
        atoms = sorted(residue)
        residue_positions = positions[atoms, :]
        residue_centroid = np.average(residue_positions, axis=0)
        inverse_centroid = residue_centroid @ inverse_box
        mask = np.floor(inverse_centroid)
        diff = (-mask @ box).reshape((-1, 3))
        temp = residue_positions + diff[:, np.newaxis, :]
        new_positions[atoms] = temp.reshape((len(atoms), 3))
    return new_positions


class CalculatorWrap(CalculatorPlugin):
    """Wrap positions before performing a calculation.
    """

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
        """Modify the calculate routine to perform wrapping beforehand.

        Args:
            calculate: The calculation routine to modify.

        Returns:
            The modified calculation routine which implements the PBC
            coordinate-wrapping before calculation.
        """
        def inner(
                return_forces: bool = True,
                return_components: bool = True,
        ) -> Results:
            self.system.positions = _wrap_positions(
                self.system.positions,
                self.system,
            )
            results = calculate(return_forces, return_components)
            return results
        return inner


class IntegratorWrap(IntegratorPlugin):
    """Wrap positions after performing an integration.
    """

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
        """Modify the integrate routine to perform wrapping afterward.

        Args:
            integrate: The integration routine to modify.

        Returns:
            The modified integration routine which implements the PBC
            coordinate-wrapping after integration.
        """
        def inner(system: System) -> Returns:
            positions, velocities = integrate(system)
            positions = _wrap_positions(positions, system)
            return positions, velocities
        return inner
