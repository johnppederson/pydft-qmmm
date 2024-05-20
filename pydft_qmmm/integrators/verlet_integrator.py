#! /usr/bin/env python3
"""A module defining the :class:`VerletIntegrator` class.
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
    """An :class:`Integrator` based on the Verlet algorithm.
    """
    timestep: float | int

    def integrate(self, system: System) -> Returns:
        masses = system.masses.reshape((-1, 1))
        momenta = system.velocities * masses
        momenta = momenta + self.timestep * system.forces*(10**-4)
        final_positions = (
            system.positions
            + self.timestep*momenta/masses
        )
        final_velocities = momenta/masses
        return final_positions, final_velocities
