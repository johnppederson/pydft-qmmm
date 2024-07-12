"""An integrator implementing the Leapfrog Verlet algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .integrator import Integrator

if TYPE_CHECKING:
    from pydft_qmmm import System
    from .integrator import Returns


@dataclass
class VerletIntegrator(Integrator):
    r"""An integrator implementing the Leapfrog Verlet algorithm.

    Args:
        timestep: The timestep (:math:`\mathrm{fs}`) used to perform
            integrations.
    """
    timestep: float | int

    def integrate(self, system: System) -> Returns:
        r"""Integrate forces into new positions and velocities.

        Args:
            system: The system whose forces
                (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`) and existing
                positions (:math:`\mathrm{\mathring{A}}`) and velocities
                (:math:`\mathrm{\mathring{A}\;fs^{-1}}`) will be used to
                determine new positions and velocities.

        Returns:
            New positions (:math:`\mathrm{\mathring{A}}`) and velocities
            (:math:`\mathrm{\mathring{A}\;fs^{-1}}`) integrated from the forces
            (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`) and existing
            positions and velocities of the system.

        .. note:: Based on the implementation of the integrator
            kernels from OpenMM.
        """
        masses = system.masses.reshape((-1, 1))
        momenta = system.velocities * masses
        momenta = momenta + self.timestep * system.forces*(10**-4)
        final_positions = (
            system.positions
            + self.timestep*momenta/masses
        )
        final_velocities = momenta/masses
        return final_positions, final_velocities
