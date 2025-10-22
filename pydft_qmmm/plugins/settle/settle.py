"""Plugin for applying SETTLE to select residues after integration.
"""
from __future__ import annotations

__all__ = ["SETTLE"]

from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .settle_utils import settle_positions
from .settle_utils import settle_velocities
from pydft_qmmm.integrators import IntegratorPlugin

if TYPE_CHECKING:
    from pydft_qmmm.integrators import Returns
    from pydft_qmmm import System


class SETTLE(IntegratorPlugin):
    r"""Apply the SETTLE algorithm to water residues after integration.

    This plugin is based off of the implementation of OpenMM in
    :openmm:`SimTKReference/ReferenceSETTLEAlgorithm.cpp`.

    Args:
        query: The VMD-like selection query which should correspond to
            water residues.
        oh_distance: The distance between the oxygen and hydrogens
            (:math:`\mathrm{\mathring{A}}`).
        hh_distance: The distance between the hydrogens
            (:math:`\mathrm{\mathring{A}}`).
    """

    def __init__(
            self,
            query: str = "resname HOH",
            oh_distance: float | int = 1.,
            hh_distance: float | int = 1.632981,
    ) -> None:
        self.query = "(" + query + ") and not subsystem I"
        self.oh_distance = oh_distance
        self.hh_distance = hh_distance

    def constrain_velocities(self, system: System) -> NDArray[np.float64]:
        """Apply the SETTLE algorithm to system velocities.

        Args:
            system: The system whose velocities will be SETTLEd.

        Returns:
            New velocities which result from the application of the
            SETTLE algorithm to system velocities.
        """
        residues = self._get_hoh_residues(system)
        velocities = settle_velocities(
            residues,
            system.positions,
            system.velocities,
            system.masses,
        )
        return velocities

    @lru_cache
    def _get_hoh_residues(
            self,
            residues: tuple[int, ...],
            residue_set: frozenset[tuple[int, frozenset[int]]],
            select: Callable[[str], frozenset[int]],
    ) -> list[list[int]]:
        """Get the water residues from the system.

        Args:
            residues: The indices of the residue to which the atoms
                of the system belong.
            residue_set: The residue index and the corresponding sets of
                atoms.
            select: The select method of the system.

        Returns:
            A list of list of atom indices, representing the all water
            residues in the system.
        """
        residue_indices = np.unique(
            np.array(residues)[sorted(select(self.query))],
        )
        residue_map = dict(residue_set)
        hoh_residues = [sorted(residue_map[i]) for i in residue_indices]
        if any([len(residue) != 3 for residue in hoh_residues]):
            raise ValueError("Some SETTLE residues do not have 3 atoms")
        return hoh_residues

    def _modify_integrate(
            self,
            integrate: Callable[[System], Returns],
    ) -> Callable[[System], Returns]:
        """Modify the integrate routine to perform SETTLE afterward.

        Args:
            integrate: The integration routine to modify.

        Returns:
            The modified integration routine which implements the SETTLE
            algorithm after integration.
        """
        def inner(system: System) -> Returns:
            positions, velocities = integrate(system)
            residues = self._get_hoh_residues(
                tuple(system.residues),
                frozenset(system.residue_map.items()),
                system.select,
            )
            if residues:
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
        """Modify the kinetic energy computation to use SETTLE.

        Args:
            compute_kinetic_energy: The kinetic energy routine to
                modify.

        Returns:
            The modified kinetic energy routine which applies the SETTLE
            algorithm to velocities.
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
            residues = self._get_hoh_residues(
                tuple(system.residues),
                frozenset(system.residue_map.items()),
                system.select,
            )
            if residues:
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
