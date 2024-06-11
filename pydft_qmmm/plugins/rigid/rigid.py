#! /usr/bin/env python3
"""A module defining the pluggable implementation of the rigid bodies
algorithm for the |package| repository.
"""
from __future__ import annotations

from typing import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from pydft_qmmm.common import Subsystem
from pydft_qmmm.plugins.plugin import IntegratorPlugin

if TYPE_CHECKING:
    from pydft_qmmm.integrators import Returns
    from pydft_qmmm.integrators import Integrator
    from pydft_qmmm import System


def _make_residue_method(
        self: Stationary,
        type_: type,
) -> Callable[[System], list[int]]:
    if type_ is int:
        def inner(system: System) -> list[int]:
            residue_indices = list({
                int(atom.molecule) for atom in system
                if atom.molecule in self.stationary_residues
                and atom.subsystem != Subsystem.I
            })
            residue_indices.sort()
            return residue_indices
    elif type_ is str:
        def inner(system: System) -> list[int]:
            residue_indices = list({
                int(atom.molecule) for atom in system
                if atom.molecule_name in self.stationary_residues
                and atom.subsystem != Subsystem.I
            })
            residue_indices.sort()
            return residue_indices
    return inner


class Stationary(IntegratorPlugin):
    """A :class:`Plugin` which implements stationary residues during
    simulation.

    :param stationary_residues: The names of residues to hold stationary
        in the :class:`System`.
    """

    def __init__(
            self,
            stationary_residues: list[str] | list[int],
    ) -> None:
        self.stationary_residues = stationary_residues
        if all([isinstance(x, int) for x in stationary_residues]):
            self._get_residue_indices = _make_residue_method(self, int)
        elif all([isinstance(x, str) for x in stationary_residues]):
            self._get_residue_indices = _make_residue_method(self, str)
        else:
            raise TypeError

    def modify(
            self,
            integrator: Integrator,
    ) -> None:
        self._modifieds.append(type(integrator).__name__)
        self.integrator = integrator
        integrator.integrate = self._modify_integrate(integrator.integrate)
        integrator.compute_kinetic_energy = self._modify_compute_kinetic_energy(
            integrator.compute_kinetic_energy,
        )

    def constrain_velocities(self, system: System) -> NDArray[np.float64]:
        """Modify the compute_velocities call in the :class:`Integrator`
        to make the velocities of a subset of residues zero.

        :param compute_velocities: The default compute_velocities method
            of the :class:`Integrator`.
        :return: The modified compute_velocities method.
        """
        velocities = system.velocities
        residues = self._get_stat_residues(system)
        velocities[residues, :] = 0
        return velocities

    def _get_stat_residues(self, system: System) -> list[list[int]]:
        residue_indices = self._get_residue_indices(system)
        residues: list[list[int]] = [[] for _ in residue_indices]
        for i, atom in enumerate(system):
            if atom.molecule in residue_indices:
                residues[residue_indices.index(int(atom.molecule))].append(i)
        return residues

    def _modify_integrate(
            self,
            integrate: Callable[[System], Returns],
    ) -> Callable[[System], Returns]:
        """Modify the integrate call in the :class:`Integrator` to
        make the positions of a subset of residues constant and their
        velocities zero.

        :param integrate: The default integrate method of the
            :class:`Integrator`.
        :return: The modified integrate method.
        """
        def inner(system: System) -> Returns:
            positions, velocities = integrate(system)
            residues = self._get_stat_residues(system)
            positions[residues, :] = system.positions[residues, :]
            velocities[residues, :] = 0.
            return positions, velocities
        return inner

    def _modify_compute_kinetic_energy(
            self,
            compute_kinetic_energy: Callable[[System], float],
    ) -> Callable[[System], float]:
        """Modify the compute_kinetic_energy call in the
        :class:`Integrator` to keep water residues rigid when evaluating
        velocities.

        :param compute_kinetic_energy: The default
            compute_kinetic_energy method of the :class:`Integrator`.
        :return: The modified compute_kinetic_energy method.
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
            residues = self._get_stat_residues(system)
            velocities[residues, :] = 0.
            kinetic_energy = (
                np.sum(0.5*masses*(velocities)**2)
                * (10**4)
            )
            return kinetic_energy
        return inner


class RigidBody(IntegratorPlugin):
    """A :class:`Plugin` which implements rigid body dynamics during
    simulation.
    """

    def __init__(self) -> None:
        raise NotImplementedError

    def modify(
            self,
            integrator: Integrator,
    ) -> None:
        pass
