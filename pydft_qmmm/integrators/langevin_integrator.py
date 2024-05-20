#! /usr/bin/env python3
"""A module defining the :class:`LangevinIntegrator` class.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .integrator import Integrator
from pydft_qmmm.common import KB

if TYPE_CHECKING:
    from pydft_qmmm import System
    from .integrator import Returns


@dataclass
class LangevinIntegrator(Integrator):
    """An :class:`Integrator` based on Langevin dynamics.

    :param temperature: |temperature|
    :param friction: |friction|
    """
    timestep: float | int
    temperature: int | float
    friction: int | float

    def integrate(self, system: System) -> Returns:
        masses = system.masses.reshape((-1, 1))
        vel_scale = np.exp(-self.timestep*self.friction)
        frc_scale = (
            self.timestep
            if self.friction == 0
            else (1 - vel_scale)/self.friction
        )
        noi_scale = (KB*self.temperature*(1 - vel_scale**2)*1000)**0.5
        z = np.random.standard_normal((len(masses), 3))
        momenta = system.velocities*masses
        momenta = (
            vel_scale*momenta
            + frc_scale*system.forces*(10**-4)
            + noi_scale*(10**-5)*z*masses**0.5
        )
        final_positions = (
            system.positions
            + self.timestep*momenta/masses
        )
        final_velocities = momenta/masses
        return final_positions, final_velocities
