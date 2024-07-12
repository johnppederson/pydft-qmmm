"""The integrator base class.
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
    r"""The abstract integrator base class.

    Attributes:
        _plugins: (class attribute) The list of plugin names that have
            been registered to the integrator.
        timestep: (class attribute) The timestep (:math:`\mathrm{fs}`)
            used to perform integrations.
    """
    _plugins: list[str] = []
    timestep: float | int

    @abstractmethod
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
        """

    def compute_kinetic_energy(self, system: System) -> float:
        r"""Calculate kinetic energy via leapfrog algorithm.

        Args:
            system: The system whose forces
                (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`) and existing
                velocities (:math:`\mathrm{\mathring{A}\;fs^{-1}}`) will be used
                to calculate the kinetic energy of the system.

        Returns:
            The kinetic energy (:math:`\mathrm{kJ\;mol^{-1}}`) of the
            system.

        .. note:: Based on the implementation of the kinetic energy
            kernels from OpenMM.
        """
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
        """Record plugin name and apply the plugin to the integrator.

        Args:
            plugin: A plugin that will modify the behavior of one or
                more integrator routines.
        """
        self._plugins.append(type(plugin).__name__)
        plugin.modify(self)

    def active_plugins(self) -> list[str]:
        """Get the current list of active plugins.

        Returns:
            A list of the active plugins registered by the integrator.
        """
        return self._plugins
