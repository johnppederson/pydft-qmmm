#! /usr/bin/env python3
"""A module to define the :class:`OpenMMInterface` class.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import openmm
from simtk.unit import angstrom
from simtk.unit import kilojoule_per_mole

from .openmm_utils import _exclude_electrostatic
from .openmm_utils import _exclude_intermolecular
from .openmm_utils import _exclude_intramolecular
from .openmm_utils import _generate_state
from .openmm_utils import _non_electrostatic
from .openmm_utils import _real_electrostatic
from pydft_qmmm.common import Subsystem
from pydft_qmmm.interfaces import MMInterface
from pydft_qmmm.interfaces import MMSettings

if TYPE_CHECKING:
    from typing import Any
    from numpy.typing import NDArray


class OpenMMInterface(MMInterface):
    """A :class:`SoftwareInterface` class which wraps the functional
    components of OpenMM.

    :param context: The OpenMM Context object for the interface.
    """

    def __init__(
            self,
            settings: MMSettings,
            base_context: openmm.Context,
            ixn_context: openmm.Context,
    ) -> None:
        self._settings = settings
        self._base_context = base_context
        self._base_force_mask = np.ones(
            (self._base_context.getSystem().getNumParticles(), 3),
        )
        self._ixn_context = ixn_context
        self._ixn_energy_group: set[int] = set()
        self._ixn_forces_group: set[int] = set()
        self._ixn_force_mask = np.ones(
            (self._base_context.getSystem().getNumParticles(), 3),
        )

    def compute_energy(self) -> float:
        base_state = _generate_state(self._base_context)
        energy = base_state.getPotentialEnergy() / kilojoule_per_mole
        ixn_state = _generate_state(
            self._ixn_context, groups=self._ixn_energy_group,
        )
        energy += ixn_state.getPotentialEnergy() / kilojoule_per_mole
        return energy

    def compute_forces(self) -> NDArray[np.float64]:
        base_state = _generate_state(self._base_context)
        forces = (
            self._base_force_mask * base_state.getForces(asNumpy=True)
            / kilojoule_per_mole * angstrom
        )
        ixn_state = _generate_state(self._ixn_context, self._ixn_energy_group)
        forces += (
            ixn_state.getForces(asNumpy=True)
            / kilojoule_per_mole * angstrom
        )
        ixn_state = _generate_state(self._ixn_context, self._ixn_forces_group)
        forces += (
            self._ixn_force_mask * ixn_state.getForces(asNumpy=True)
            / kilojoule_per_mole * angstrom
        )
        return forces

    def compute_components(self) -> dict[str, float]:
        components = {}
        for force in range(self._base_context.getSystem().getNumForces()):
            key = type(
                self._base_context.getSystem().getForce(force),
            ).__name__.replace("Force", "Energy")
            key = "Base " + " ".join(re.findall("[A-Z][a-z]*", key))
            value = _generate_state(
                self._base_context,
                {force},
            ).getPotentialEnergy() / kilojoule_per_mole
            components[key] = value
        for force in self._ixn_energy_group:
            key = type(
                self._ixn_context.getSystem().getForce(force),
            ).__name__.replace("Force", "Energy")
            key = "Correction " + " ".join(re.findall("[A-Z][a-z]*", key))
            value = _generate_state(
                self._ixn_context,
                {force},
            ).getPotentialEnergy() / kilojoule_per_mole
            components[key] = value
        return components

    def zero_intramolecular(self, atoms: frozenset[int]) -> None:
        """
        """
        system = self._base_context.getSystem()
        _exclude_intramolecular(system, atoms)
        state = self._base_context.getState(getPositions=True)
        positions = state.getPositions()
        self._base_context.reinitialize()
        self._base_context.setPositions(positions)

    def zero_forces(self, atoms: frozenset[int]) -> None:
        """
        """
        for i in atoms:
            for j in range(3):
                self._base_force_mask[i, j] = 0

    def add_real_elst(
            self,
            atoms: frozenset[int],
            const: float | int = 1,
            inclusion: NDArray[np.float64] | None = None,
    ) -> None:
        """
        """
        system = self._base_context.getSystem()
        forces = _real_electrostatic(system, atoms, const)
        system = self._ixn_context.getSystem()
        indices = set()
        for force in forces:
            i = system.addForce(force)
            force.setForceGroup(i)
            indices.add(i)
        if inclusion is None:
            self._ixn_energy_group.update(indices)
        else:
            self._ixn_forces_group.update(indices)
            self._ixn_force_mask = inclusion
        state = self._base_context.getState(getPositions=True)
        positions = state.getPositions()
        self._ixn_context.reinitialize()
        self._ixn_context.setPositions(positions)

    def add_non_elst(
            self,
            atoms: frozenset[int],
            inclusion: NDArray[np.float64] | None = None,
    ) -> None:
        """
        """
        system = self._base_context.getSystem()
        forces = _non_electrostatic(system, atoms)
        system = self._ixn_context.getSystem()
        indices = set()
        for force in forces:
            i = system.addForce(force)
            force.setForceGroup(i)
            indices.add(i)
        if inclusion is None:
            self._ixn_energy_group.update(indices)
        else:
            self._ixn_forces_group.update(indices)
            self._ixn_force_mask = inclusion
        state = self._base_context.getState(getPositions=True)
        positions = state.getPositions()
        self._ixn_context.reinitialize()
        self._ixn_context.setPositions(positions)

    def zero_charges(self, atoms: frozenset[int]) -> None:
        """
        """
        system = self._base_context.getSystem()
        _exclude_electrostatic(system, atoms)
        state = self._base_context.getState(getPositions=True)
        positions = state.getPositions()
        self._base_context.reinitialize()
        self._base_context.setPositions(positions)

    def zero_intermolecular(self, atoms: frozenset[int]) -> None:
        """
        """
        system = self._base_context.getSystem()
        _exclude_intermolecular(system, atoms)
        state = self._base_context.getState(getPositions=True)
        positions = state.getPositions()
        self._base_context.reinitialize()
        self._base_context.setPositions(positions)

    def update_charges(self, charges: NDArray[np.float64]) -> None:
        """Update the atom charges for OpenMM.

        :param charges: |charges|
        """
        nonbonded_forces = [
            force for force in self._base_context.getSystem().getForces()
            if isinstance(force, openmm.NonbondedForce)
        ]
        for force in nonbonded_forces:
            for i, charge in enumerate(charges):
                _, sigma, epsilon = force.getParticleParameters(i)
                force.setParticleParameters(i, charge, sigma, epsilon)
            force.updateParametersInContext(self._base_context)
        custom_nonbonded_forces = [
            force for force in self._ixn_context.getSystem().getForces()
            if isinstance(force, openmm.CustomNonbondedForce)
        ]
        for force in custom_nonbonded_forces:
            if "nn" in force.getEnergyFunction():
                for i, charge in enumerate(charges):
                    _, n = force.getParticleParameters(i)
                    force.setParticleParameters(i, [charge, n])
                force.updateParametersInContext(self._ixn_context)

    def update_positions(self, positions: NDArray[np.float64]) -> None:
        """Update the atom positions for OpenMM.

        :param positions: |positions|
        """
        positions_temp = []
        for i in range(len(positions)):
            positions_temp.append(
                openmm.Vec3(
                    positions[i][0],
                    positions[i][1],
                    positions[i][2],
                ) * angstrom,
            )
        self._base_context.setPositions(positions_temp)
        self._ixn_context.setPositions(positions_temp)

    def update_box(self, box: NDArray[np.float64]) -> None:
        """Update the box vectors for OpenMM.

        :param box: |box|

        .. warning:: This method is not currently implemented.
        """

    def update_subsystems(
            self,
            subsystems: np.ndarray[Any, np.dtype[np.object_]],
    ) -> None:
        """Update the box vectors for OpenMM.
        """
        custom_nonbonded_forces = [
            force for force in self._ixn_context.getSystem().getForces()
            if isinstance(force, openmm.CustomNonbondedForce)
        ]
        for force in custom_nonbonded_forces:
            if "nn" in force.getEnergyFunction():
                for i, s in enumerate(subsystems):
                    q, _ = force.getParticleParameters(i)
                    if s == Subsystem.III:
                        force.setParticleParameters(i, [q, 0])
                    else:
                        force.setParticleParameters(i, [q, 1])
                force.updateParametersInContext(self._ixn_context)
