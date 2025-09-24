"""The OpenMM software interface and potential.

This module contains the software interface for storing and manipulating
OpenMM data types and the potential using OpenMM to calculate energies
and forces.
"""
from __future__ import annotations

__all__ = [
    "OpenMMInterface",
    "OpenMMPotential",
]

from dataclasses import dataclass
from dataclasses import field
import re
from typing import TYPE_CHECKING

import numpy as np
import openmm
import openmm.unit

from pydft_qmmm.utils import Subsystem
from pydft_qmmm.interfaces import MMInterface
from pydft_qmmm.potentials import AtomicPotential

from . import openmm_utils

if TYPE_CHECKING:
    from typing import Any
    from numpy.typing import NDArray
    from pydft_qmmm import System  # noqa: F401


@dataclass(frozen=True)
class OpenMMInterface(MMInterface):
    r"""A mix-in for storing and manipulating OpenMM data types.

    Args:
        system: The system that will inform the interface to the
            external software.
        base_context: The base OpenMM Context required to perform
            energy and force calculations.
        aux_context: An auxiliary OpenMM Context to allow for different
            periodic QM/MM algorithms.
        base_force_mask: A mask to element-wise multiply against the
            forces computed by the base OpenMM Context.
        aux_energy_group_force_mask: A mask to element-wise multiply
            against the forces computed by the auxiliary OpenMM Context
            with the energy force group.
        aux_forces_group_force_mask: A mask to element-wise multiply
            against the forces computed by the auxiliary OpenMM Context
            with the forces force group.
        aux_energy_group: A force group for the auxiliary OpenMM Context
            whose masked forces and energy will be included.
        aux_forces_group: A force group for the auxiliary OpenMM Context
            whose masked forces will be included.
    """
    base_context: openmm.Context
    aux_context: openmm.Context
    base_force_mask: NDArray[np.float64]
    aux_energy_group_force_mask: NDArray[np.float64]
    aux_forces_group_force_mask: NDArray[np.float64]
    aux_energy_group: set[int] = field(default_factory=set)
    aux_forces_group: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Register update methods with relevant system attributes"""
        self.system.charges.register_notifier(self.update_charges)
        self.system.positions.register_notifier(self.update_positions)
        self.system.box.register_notifier(self.update_box)
        self.system.subsystems.register_notifier(self.update_subsystems)

    def zero_intramolecular(self, atoms: frozenset[int]) -> None:
        """Remove intra-molecular interactions for the specified atoms.

        Args:
            atoms: The indices of atoms from which to remove
                intra-molecular interactions.
        """
        base_system = self.base_context.getSystem()
        openmm_utils._exclude_intramolecular(base_system, atoms)
        base_state = self.base_context.getState(getPositions=True)
        omm_pos = base_state.getPositions()
        self.base_context.reinitialize()
        self.base_context.setPositions(omm_pos)

    def zero_forces(self, atoms: frozenset[int]) -> None:
        """Zero forces on the specified atoms.

        Args:
            atoms: The indices of atoms for which to zero forces.
        """
        for i in atoms:
            for j in range(3):
                self.base_force_mask[i, j] = 0
                self.aux_energy_group_force_mask[i, j] = 0

    def add_real_elst(
            self,
            atoms: frozenset[int],
            const: float | int = 1,
            inclusion: NDArray[np.float64] | None = None,
    ) -> None:
        """Add Coulomb interaction for the specified atoms.

        Args:
            atoms: The indices of atoms for which to add a Coulomb
                interaction.
            const: A constant to multiply at the beginning of the
                coulomb expression.
            inclusion: An Nx3 array with values that will be applied to
                the forces of the Coulomb interaction through
                element-wise multiplication.
        """
        base_system = self.base_context.getSystem()
        forces = openmm_utils._real_electrostatic(base_system, atoms, const)
        aux_system = self.aux_context.getSystem()
        indices = set()
        for force in forces:
            i = aux_system.addForce(force)
            force.setForceGroup(i)
            indices.add(i)
        if inclusion is None:
            self.aux_energy_group.update(indices)
        else:
            self.aux_forces_group.update(indices)
            self.aux_forces_group_force_mask[:, :] = inclusion
        base_state = self.base_context.getState(getPositions=True)
        omm_pos = base_state.getPositions()
        self.aux_context.reinitialize()
        self.aux_context.setPositions(omm_pos)

    def add_non_elst(
            self,
            atoms: frozenset[int],
            inclusion: NDArray[np.float64] | None = None,
    ) -> None:
        """Add non-electrostatic interactions for the specified atoms.

        Args:
            atoms: The indices of atoms for which to add
                non-electrostatic, non-bonded interactions.
            inclusion: An Nx3 array with values that will be applied to
                the forces of the non-electrostatic interaction through
                element-wise multiplication.
        """
        base_system = self.base_context.getSystem()
        forces = openmm_utils._non_electrostatic(base_system, atoms)
        aux_system = self.aux_context.getSystem()
        indices = set()
        for force in forces:
            i = aux_system.addForce(force)
            force.setForceGroup(i)
            indices.add(i)
        if inclusion is None:
            self.aux_energy_group.update(indices)
        else:
            self.aux_forces_group.update(indices)
            self.aux_forces_group_force_mask[:, :] = inclusion
        base_state = self.base_context.getState(getPositions=True)
        omm_pos = base_state.getPositions()
        self.aux_context.reinitialize()
        self.aux_context.setPositions(omm_pos)

    def zero_charges(self, atoms: frozenset[int]) -> None:
        """Remove charges from the specified atoms.

        Args:
            atoms: The indices of atoms from which to remove charges.
        """
        base_system = self.base_context.getSystem()
        openmm_utils._exclude_electrostatic(base_system, atoms)
        base_state = self.base_context.getState(getPositions=True)
        omm_pos = base_state.getPositions()
        self.base_context.reinitialize()
        self.base_context.setPositions(omm_pos)

    def zero_intermolecular(self, atoms: frozenset[int]) -> None:
        """Remove inter-molecular interactions for the specified atoms.

        Args:
            atoms: The indices of atoms from which to remove
                inter-molecular interactions.
        """
        base_system = self.base_context.getSystem()
        openmm_utils._exclude_intermolecular(base_system, atoms)
        base_state = self.base_context.getState(getPositions=True)
        omm_pos = base_state.getPositions()
        self.base_context.reinitialize()
        self.base_context.setPositions(omm_pos)

    def get_pme_parameters(self) -> tuple[float, tuple[int, int, int], int]:
        r"""Get the parameters used for PME summation.

        Returns:
            The Gaussian width parameter in Ewald summation
            (:math:`\mathrm{\mathring(A)^{-1}}`), the number of grid
            points to include along each lattice edge, and the order of
            splines used on the FFT grid.
        """
        nonbonded_forces = [
            force for force in self.base_context.getSystem().getForces()
            if isinstance(force, openmm.NonbondedForce)
        ]
        pme_forces = [
            force for force in nonbonded_forces
            if force.getNonbondedMethod() == openmm.NonbondedForce.PME
        ]
        if len(pme_forces) != 1:
            raise TypeError(f"{len(pme_forces)} OpenMM Forces have PME params")
        pme_alpha, *pme_gridnumber = pme_forces[0].getPMEParametersInContext(
            self.base_context,
        )
        return pme_alpha/10, pme_gridnumber, 5

    def update_charges(self, charges: NDArray[np.float64]) -> None:
        """Set the atomic partial charges used by OpenMM.

        Args:
            charges: The partial charges (:math:`e`) of the atoms.
        """
        nonbonded_forces = [
            force for force in self.base_context.getSystem().getForces()
            if isinstance(force, openmm.NonbondedForce)
        ]
        for force in nonbonded_forces:
            openmm_utils._update_exceptions(force, charges)
            for i, charge in enumerate(charges):
                _, sigma, epsilon = force.getParticleParameters(i)
                force.setParticleParameters(i, charge, sigma, epsilon)
            force.updateParametersInContext(self.base_context)
        custom_nonbonded_forces = [
            force for force in self.aux_context.getSystem().getForces()
            if isinstance(force, openmm.CustomNonbondedForce)
        ]
        for force in custom_nonbonded_forces:
            if "nn" in force.getEnergyFunction():
                for i, charge in enumerate(charges):
                    _, n = force.getParticleParameters(i)
                    force.setParticleParameters(i, [charge, n])
                force.updateParametersInContext(self.aux_context)

    def update_positions(self, positions: NDArray[np.float64]) -> None:
        r"""Set the atomic positions used by OpenMM.

        Args:
            positions: The positions (:math:`\mathrm{\mathring{A}}`) of
                the atoms within the system.
        """
        omm_pos = [openmm.Vec3(*x)*openmm.unit.angstrom for x in positions]
        self.base_context.setPositions(omm_pos)
        self.aux_context.setPositions(omm_pos)

    def update_box(self, box: NDArray[np.float64]) -> None:
        r"""Set the lattice vectors used by OpenMM.

        Args:
            box: The lattice vectors (:math:`\mathrm{\mathring{A}}`) of
                the box containing the system.
        """
        pass

    def update_subsystems(
            self,
            subsystems: np.ndarray[Any, np.dtype[np.object_]],
    ) -> None:
        """Adjust embedding-related values by subsystem membership.

        Args:
            subsystems: The subsystems of which the atoms are a part.
        """
        custom_nonbonded_forces = [
            force for force in self.aux_context.getSystem().getForces()
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
                force.updateParametersInContext(self.aux_context)


class OpenMMPotential(OpenMMInterface, AtomicPotential):
    """A potential wrapping OpenMM functionality.

    Args:
        system: The system that will inform the interface to the
            external software.
        base_context: The base OpenMM Context required to perform
            energy and force calculations.
        aux_context: An auxiliary OpenMM Context to allow for different
            periodic QM/MM algorithms.
        base_force_mask: A mask to element-wise multiply against the
            forces computed by the base OpenMM Context.
        aux_energy_group_force_mask: A mask to element-wise multiply
            against the forces computed by the auxiliary OpenMM Context
            with the energy force group.
        aux_forces_group_force_mask: A mask to element-wise multiply
            against the forces computed by the auxiliary OpenMM Context
            with the forces force group.
        aux_energy_group: A force group for the auxiliary OpenMM Context
            whose masked forces and energy will be included.
        aux_forces_group: A force group for the auxiliary OpenMM Context
            whose masked forces will be included.
    """

    def compute_energy(self) -> float:
        r"""Compute the energy of the system using OpenMM.

        Returns:
            The energy (:math:`\mathrm{kJ\;mol^{-1}}`) of the system.
        """
        base_state = openmm_utils._generate_state(self.base_context)
        energy = (
            base_state.getPotentialEnergy()
            / openmm.unit.kilojoule_per_mole
        )
        if self.aux_energy_group:
            aux_state = openmm_utils._generate_state(
                self.aux_context, groups=self.aux_energy_group,
            )
            energy += (
                aux_state.getPotentialEnergy()
                / openmm.unit.kilojoule_per_mole
            )
        return energy

    def compute_forces(self) -> NDArray[np.float64]:
        r"""Compute the forces on the system using OpenMM.

        Returns:
            The forces
            (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`) acting
            on atoms in the system.
        """
        base_state = openmm_utils._generate_state(self.base_context)
        forces = (
            self.base_force_mask * base_state.getForces(asNumpy=True)
            / openmm.unit.kilojoule_per_mole * openmm.unit.angstrom
        )
        if self.aux_energy_group:
            aux_state = openmm_utils._generate_state(
                self.aux_context, self.aux_energy_group,
            )
            forces += (
                self.aux_energy_group_force_mask
                * aux_state.getForces(asNumpy=True)
                / openmm.unit.kilojoule_per_mole
                * openmm.unit.angstrom
            )
        if self.aux_forces_group:
            aux_state = openmm_utils._generate_state(
                self.aux_context, self.aux_forces_group,
            )
            forces += (
                self.aux_forces_group_force_mask
                * aux_state.getForces(asNumpy=True)
                / openmm.unit.kilojoule_per_mole
                * openmm.unit.angstrom
            )
        return forces

    def compute_components(self) -> dict[str, float]:
        r"""Compute the components of energy using OpenMM.

        Returns:
            The components of the energy (:math:`\mathrm{kJ\;mol^{-1}}`)
            of the system.
        """
        components = {}
        for force in range(self.base_context.getSystem().getNumForces()):
            key = type(
                self.base_context.getSystem().getForce(force),
            ).__name__.replace("Force", "Energy")
            key = "Base " + " ".join(re.findall("[A-Z][a-z]*", key))
            value = openmm_utils._generate_state(
                self.base_context,
                {force},
            ).getPotentialEnergy() / openmm.unit.kilojoule_per_mole
            components[key] = value
        for force in self.aux_energy_group:
            key = type(
                self.aux_context.getSystem().getForce(force),
            ).__name__.replace("Force", "Energy")
            key = "Correction " + " ".join(re.findall("[A-Z][a-z]*", key))
            value = openmm_utils._generate_state(
                self.aux_context,
                {force},
            ).getPotentialEnergy() / openmm.unit.kilojoule_per_mole
            components[key] = value
        return components
