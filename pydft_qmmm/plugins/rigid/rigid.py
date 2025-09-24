"""Plugins for implementing stationary or rigid-body residues.
"""
from __future__ import annotations

__all__ = ["Stationary"]

from typing import TYPE_CHECKING

import numpy as np

from pydft_qmmm.integrators import IntegratorPlugin

if TYPE_CHECKING:
    from collections.abc import Callable
    from numpy.typing import NDArray
    from pydft_qmmm.integrators import Returns
    from pydft_qmmm import System


class Stationary(IntegratorPlugin):
    """Keep select residues stationary during integration.

    Args:
        query: The VMD-like selection query which corresponds to
            residues that should be kept stationary during integration.
    """

    def __init__(
            self,
            query: str,
    ) -> None:
        self.query = query

    def constrain_velocities(self, system: System) -> NDArray[np.float64]:
        """Zero velocities for stationary residues.

        Args:
            system: The system with stationary residues.

        Returns:
            New velocities which result from zeroing the
            velocities of stationary residues.
        """
        velocities = system.velocities
        atoms = sorted(system.select(self.query))
        velocities[atoms, :] = 0
        return velocities

    def _modify_integrate(
            self,
            integrate: Callable[[System], Returns],
    ) -> Callable[[System], Returns]:
        """Modify the integrate routine to stop stationary residues.

        Args:
            integrate: The integration routine to modify.

        Returns:
            The modified integration routine which keeps stationary
            residues in place.
        """
        def inner(system: System) -> Returns:
            positions, velocities = integrate(system)
            atoms = sorted(system.select(self.query))
            positions[atoms, :] = system.positions[atoms, :]
            velocities[atoms, :] = 0.
            return positions, velocities
        return inner

    def _modify_compute_kinetic_energy(
            self,
            compute_kinetic_energy: Callable[[System], float],
    ) -> Callable[[System], float]:
        """Modify the kinetic energy computation for stationary residues.

        Args:
            compute_kinetic_energy: The kinetic energy routine to
                modify.

        Returns:
            The modified kinetic energy routine which zeros stationary
            residue velocities before the energy calculation.
        """
        def inner(system: System) -> float:
            masses = system.masses.reshape(-1, 1)
            velocities = (
                system.velocities
                + (
                    0.5*self.integrator.timestep
                    * system.forces*(10**-4)/masses
                )
            )
            atoms = sorted(system.select(self.query))
            velocities[atoms, :] = 0.
            kinetic_energy = (
                np.sum(0.5*masses*(velocities)**2)
                * (10**4)
            )
            return kinetic_energy
        return inner
