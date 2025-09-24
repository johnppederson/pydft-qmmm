"""Base classes for integrators and related classes.

This module contains the base classes for integrators and integrator
plugins, as well as the returns type.
"""
from __future__ import annotations

__all__ = [
    "Integrator",
    "IntegratorPlugin",
    "Returns",
]

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from pydft_qmmm.utils import pluggable_method

if TYPE_CHECKING:
    from pydft_qmmm import System

Returns: TypeAlias = tuple[NDArray[np.float64], NDArray[np.float64]]


@dataclass(frozen=True)
class Integrator(ABC):
    r"""The abstract integrator base class.

    Args:
        timestep: The timestep (:math:`\mathrm{fs}`) used to perform
            integrations.

    Attributes:
        _plugins: The list of plugins that have been registered by the
            integrator.
    """
    timestep: float | int
    _plugins: list[IntegratorPlugin] = field(default_factory=list, init=False)

    @pluggable_method
    @abstractmethod
    def integrate(self, system: System) -> Returns:
        r"""Integrate forces into new positions and velocities.

        Args:
            system: The system whose forces
                (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`) and
                existing positions (:math:`\mathrm{\mathring{A}}`) and
                velocities (:math:`\mathrm{\mathring{A}\;fs^{-1}}`) will
                be used to determine new positions and velocities.

        Returns:
            New positions (:math:`\mathrm{\mathring{A}}`) and velocities
            (:math:`\mathrm{\mathring{A}\;fs^{-1}}`) integrated from the
            forces (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`)
            and existing positions and velocities of the system.
        """

    @pluggable_method
    def compute_kinetic_energy(self, system: System) -> float:
        r"""Calculate kinetic energy via leapfrog algorithm.

        This method is based off of the implementation of OpenMM in
        :openmm:`ReferenceKernels.cpp`.

        Args:
            system: The system whose forces
                (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`) and
                velocities (:math:`\mathrm{\mathring{A}\;fs^{-1}}`) will
                be used to calculate the kinetic energy of the system.

        Returns:
            The kinetic energy (:math:`\mathrm{kJ\;mol^{-1}}`) of the
            system.
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

    def register_plugin(
            self,
            plugin: IntegratorPlugin,
            index: int | None = None,
    ) -> None:
        """Record plugin and apply it to the calculator.

        Args:
            plugin: A plugin that will modify the behavior of the
                integrator's pluggable methods.
            index: The index at which to insert the plugin in the
                plugin load order, i.e., 0 will be called first,
                1 will be called second, etc.  The default behavior
                is to append the plugin at the end of the plugin
                load order.
        """
        if index is not None:
            self._plugins.insert(index, plugin)
        else:
            self._plugins.append(plugin)
        plugin.modify(self)

    def active_plugins(self) -> list[IntegratorPlugin]:
        """Get the current list of active plugins.

        Returns:
            A list of the active plugins registered by the integrator.
        """
        return self._plugins


class IntegratorPlugin:
    """The abstract base class for modifying integrator routines.
    """

    def modify(self, integrator: Integrator) -> None:
        """Modify the functionality of an integrator.

        Args:
            integrator: The integrator whose functionality will be
                modified by the plugin.
        """
        self.integrator = integrator
