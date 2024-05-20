#! /usr/bin/env python3
"""A module for defining the :class:`Integrator` base class.
"""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pydft_qmmm import System
    from pydft_qmmm.plugins.plugin import IntegratorPlugin

Returns = tuple[NDArray[np.float64], NDArray[np.float64]]


class Integrator(ABC):
    """An abstract :class:`Integrator` base class for interfacing with
    plugins.
    """
    _plugins: list[str] = []
    timestep: float | int

    @abstractmethod
    def integrate(self, system: System) -> Returns:
        """Integrate forces from the :class:`System` into new positions
        and velocities.

        :return: The new positions and velocities of the
            :class:`System`, in Angstroms and Angstroms per
            femtosecond, respectively.

        .. note:: Based on the implementation of the integrator
            kernels from OpenMM.
        """

    def compute_kinetic_energy(self, system: System) -> float:
        masses = system.masses.reshape(-1, 1)
        velocities = (
            system.velocities
            + (
                0.5*self.timestep
                * system.forces*(10**-4)/masses
            )
        )
        kinetic_energy = np.sum(0.5*masses*(velocities)**2) * (10**4)
        return kinetic_energy

    def register_plugin(self, plugin: IntegratorPlugin) -> None:
        """Register a :class:`Plugin` modifying an :class:`Integrator`
        routine.

        :param plugin: An :class:`IntegratorPlugin` object.
        """
        self._plugins.append(type(plugin).__name__)
        plugin.modify(self)

    def active_plugins(self) -> list[str]:
        """Get the current list of active plugins.

        :return: A list of the active plugins being employed by the
            :class:`Integrator`.
        """
        return self._plugins
