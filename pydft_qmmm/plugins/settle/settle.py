#! /usr/bin/env python3
"""A module defining the pluggable implementation of the SETTLE
algorithm for the |package| repository.
"""
from __future__ import annotations

from typing import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .settle_utils import settle_positions
from .settle_utils import settle_velocities
from pydft_qmmm.plugins.plugin import IntegratorPlugin
from pydft_qmmm.common import Subsystem

if TYPE_CHECKING:
    from pydft_qmmm.integrator import Integrator
    from pydft_qmmm.integrator import Returns
    from pydft_qmmm import System


class SETTLE(IntegratorPlugin):
    """A :class:`Plugin` which implements the SETTLE algorithm for
    positions and velocities.

    :param oh_distance: The distance between the oxygen and hydrogen, in
        Angstroms.
    :param hh_distance: The distance between the hydrogens, in
        Angstroms.
    :param hoh_residue: The name of the water residues in the
        :class:`System`.
    """

    def __init__(
            self,
            oh_distance: float | int = 1.,
            hh_distance: float | int = 1.632981,
            hoh_residue: str = "HOH",
    ) -> None:
        self.oh_distance = oh_distance
        self.hh_distance = hh_distance
        self.hoh_residue = hoh_residue

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

    def constrain_velocities(system: System) -> NDArray[np.float64]:
        residues = self._get_hoh_residues(system)
        velocities = settle_velocities(
            residues,
            system.positions,
            system.velocities,
            system.masses,
        )
        return velocities

    def _get_hoh_residues(self, system: System) -> list[list[int]]:
        residue_indices = list({atom.molecule for atom in system
                                if atom.molecule_name == self.hoh_residue
                                and atom.subsystem != Subsystem.I})
        residue_indices.sort()
        residues = [[] for _ in residue_indices]
        for i, atom in enumerate(system):
            if atom.molecule in residue_indices:
                residues[residue_indices.index(atom.molecule)].append(i)
        return residues

    def _modify_integrate(
            self,
            integrate: Callable[[System], Returns],
    ) -> Callable[[System], Returns]:
        """Modify the integrate call in the :class:`Integrator` to
        hold H-O and H-H distances constant for water residues.

        :param integrate: The default integrate method of the
            :class:`Integrator`.
        :return: The modified integrate method.
        """
        def inner(system: System) -> Returns:
            positions, velocities = integrate(system)
            residues = self._get_hoh_residues(system)
            positions = settle_positions(
                residues,
                system.positions,
                positions,
                system.masses,
                self.oh_distance,
                self.hh_distance,
            )
            velocities[residues, :] = (
                (
                    positions[residues, :]
                    - system.positions[residues, :]
                ) / self.integrator.timestep
            )
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
            residues = self._get_hoh_residues(system)
            velocities = settle_velocities(
                residues,
                system.positions,
                velocities,
                system.masses,
            )
            kinetic_energy = (
                np.sum(0.5*masses*(velocities)**2)
                * (10**4)
            )
            return kinetic_energy
        return inner
