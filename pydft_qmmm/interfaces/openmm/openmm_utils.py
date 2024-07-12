"""Functionality for performing exclusions and generating State objects.
"""
from __future__ import annotations

import openmm
from simtk.unit import kilojoule_per_mole
from simtk.unit import nanometer


def _generate_state(
        context: openmm.Context,
        groups: set[int] | int | None = -1,
) -> openmm.State:
    """Generate an OpenMM State in order to collect energies and forces.

    Args:
        context: An OpenMM Context object containing a representation
            of the system and appropriate forces.
        groups: The force groups of the context to include in the
            State evaluation.

    Return:
        An OpenMM State object containing the energies and forces of
        the current state of the system represented within the Context
        object for the specified groups of forces.
    """
    if groups is None:
        groups = -1
    return context.getState(getEnergy=True, getForces=True, groups=groups)


def _exclude_intramolecular(
        system: openmm.System,
        atoms: frozenset[int],
) -> None:
    """Remove intramolecular interactions for a set of atoms.

    Args:
        system: The OpenMM representation of forces, constraints, and
            particles.
        atoms: The indices of atoms to remove intra-molecular
            interactions from.
    """
    # Remove double-counted intramolecular interactions for QM atoms.
    # This doesn't currently generalize to residues which are part MM
    # and part QM.
    _exclude_harmonic_bond(system, atoms)
    _exclude_harmonic_angle(system, atoms)
    _exclude_periodic_torsion(system, atoms)
    _exclude_rb_torsion(system, atoms)
    nonbonded_forces = [
        force for force in system.getForces()
        if isinstance(force, openmm.NonbondedForce)
    ]
    custom_nonbonded_forces = [
        force for force in system.getForces()
        if isinstance(force, openmm.CustomNonbondedForce)
    ]
    atom_list = list(atoms)
    atom_list.sort()
    for force in nonbonded_forces:
        for i, j in enumerate(atom_list):
            for k in atom_list[i+1:]:
                force.addException(j, k, 0, 1, 0, True)
    for force in custom_nonbonded_forces:
        exclusions = [
            force.getExclusionParticles(
                i,
            ) for i in range(force.getNumExclusions())
        ]
        for i, j in enumerate(atom_list):
            for k in atom_list[i+1:]:
                if not ([j, k] in exclusions or [k, j] in exclusions):
                    force.addExclusion(j, k)


def _exclude_harmonic_bond(
        system: openmm.System,
        atoms: frozenset[int],
) -> None:
    """Remove harmonic bond interactions for a set of atoms.

    Args:
        system: The OpenMM representation of forces, constraints, and
            particles.
        atoms: The indices of atoms to remove harmonic bond
            interactions from.
    """
    harmonic_bond_forces = [
        force for force in system.getForces()
        if isinstance(force, openmm.HarmonicBondForce)
    ]
    for force in harmonic_bond_forces:
        for i in range(force.getNumBonds()):
            *p, r0, k = force.getBondParameters(i)
            if not set(p).isdisjoint(atoms):
                k *= 0
                force.setBondParameters(i, *p, r0, k)


def _exclude_harmonic_angle(
        system: openmm.System,
        atoms: frozenset[int],
) -> None:
    """Remove harmonic angle interactions for a set of atoms.

    Args:
        system: The OpenMM representation of forces, constraints, and
            particles.
        atoms: The indices of atoms to remove harmonic angle
            interactions from.
    """
    harmonic_angle_forces = [
        force for force in system.getForces()
        if isinstance(force, openmm.HarmonicAngleForce)
    ]
    for force in harmonic_angle_forces:
        for i in range(force.getNumAngles()):
            *p, r0, k = force.getAngleParameters(i)
            if not set(p).isdisjoint(atoms):
                k *= 0
                force.setAngleParameters(i, *p, r0, k)


def _exclude_periodic_torsion(
        system: openmm.System,
        atoms: frozenset[int],
) -> None:
    """Remove periodic torsion interactions for a set of atoms.

    Args:
        system: The OpenMM representation of forces, constraints, and
            particles.
        atoms: The indices of atoms to remove periodic torsion
            interactions from.
    """
    periodic_torsion_forces = [
        force for force in system.getForces()
        if isinstance(force, openmm.PeriodicTorsionForce)
    ]
    for force in periodic_torsion_forces:
        for i in range(force.getNumTorsions()):
            *p, n, t, k = force.getTorsionParameters(i)
            if not set(p).isdisjoint(atoms):
                k *= 0
                force.setTorsionParameters(i, *p, n, t, k)


def _exclude_rb_torsion(
        system: openmm.System,
        atoms: frozenset[int],
) -> None:
    """Remove Ryckaert-Bellemans torsion interactions for a set of atoms.

    Args:
        system: The OpenMM representation of forces, constraints, and
            particles.
        atoms: The indices of atoms to remove Ryckaert-Bellemans torsion
            interactions from.
    """
    rb_torsion_forces = [
        force for force in system.getForces()
        if isinstance(force, openmm.RBTorsionForce)
    ]
    for force in rb_torsion_forces:
        for i in range(force.getNumTorsions()):
            *p, c0, c1, c2, c3, c4, c5 = force.getTorsionParameters(i)
            if not set(p).isdisjoint(atoms):
                c0, c1, c2, c3, c4, c5 = (
                    x*0 for x in (c0, c1, c2, c3, c4, c5)
                )
                force.setTorsionParameters(i, *p, c0, c1, c2, c3, c4, c5)


def _real_electrostatic(
        system: openmm.System,
        atoms: frozenset[int],
        const: int | float,
) -> list[openmm.customNonbondedForce]:
    """Add Coulomb interactions for a set of atoms.

    Args:
        system: The OpenMM representation of forces, constraints, and
            particles.
        atoms: The indices of atoms to add a Coulomb interaction
            for.
        const: A constant to multiply at the beginning of the
            coulomb expression.
    """
    other_atoms = {i for i in range(system.getNumParticles())} - atoms
    nonbonded_forces = [
        force for force in system.getForces()
        if isinstance(force, openmm.NonbondedForce)
    ]
    forces = []
    for force in nonbonded_forces:
        new_force = openmm.CustomNonbondedForce(
            f"""{const}*138.935459977*nn*qq/r;
            qq=q1*q2;
            nn=n1*n2""",
        )
        new_force.addPerParticleParameter("q")
        new_force.addPerParticleParameter("n")
        for atom in range(system.getNumParticles()):
            q, _, _ = force.getParticleParameters(
                atom,
            )
            new_force.addParticle(
                [q, 0],
            )
        for atom in atoms:
            q, _ = new_force.getParticleParameters(atom)
            new_force.setParticleParameters(atom, [q, 1])
        force.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)
        new_force.addInteractionGroup(
            atoms,
            other_atoms,
        )
        exclusions = [
            force.getExceptionParameters(
                i,
            ) for i in range(force.getNumExceptions())
        ]
        exclusions = [
            [x[0], x[1]]
            for x in exclusions if x[-1] / kilojoule_per_mole == 0
        ]
        for x in exclusions:
            new_force.addExclusion(*x)
        forces.append(new_force)
    return forces


def _non_electrostatic(
        system: openmm.System,
        atoms: frozenset[int],
) -> list[openmm.customNonbondedForce]:
    """Add a non-electrostatic interactions for a set of atoms.

    Args:
        system: The OpenMM representation of forces, constraints, and
            particles.
        atoms: The indices of atoms to add a non-electrostatic,
            non-bonded interaction for.
    """
    other_atoms = {i for i in range(system.getNumParticles())} - atoms
    nonbonded_forces = [
        force for force in system.getForces()
        if isinstance(force, openmm.NonbondedForce)
    ]
    custom_nonbonded_forces = [
        force for force in system.getForces()
        if isinstance(force, openmm.CustomNonbondedForce)
    ]
    forces = []
    if custom_nonbonded_forces:
        for force in custom_nonbonded_forces:
            forces.append(force.__copy__())
    for force in nonbonded_forces:
        new_force = openmm.CustomNonbondedForce(
            """4*epsilon*((sigma/r)^12-(sigma/r)^6);
            sigma=0.5*(sigma1+sigma2);
            epsilon=sqrt(epsilon1*epsilon2)""",
        )
        new_force.addPerParticleParameter("epsilon")
        new_force.addPerParticleParameter("sigma")
        for atom in range(system.getNumParticles()):
            _, sigma, epsilon = force.getParticleParameters(
                atom,
            )
            new_force.addParticle(
                [epsilon / kilojoule_per_mole, sigma / nanometer],
            )
        exclusions = [
            force.getExceptionParameters(
                i,
            ) for i in range(force.getNumExceptions())
        ]
        exclusions = [
            [x[0], x[1]]
            for x in exclusions if x[-1] / kilojoule_per_mole == 0
        ]
        for x in exclusions:
            new_force.addExclusion(*x)
        forces.append(new_force)
    for force in forces:
        force.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)
        force.addInteractionGroup(
            atoms,
            other_atoms,
        )
    return forces


def _exclude_intermolecular(
        system: openmm.System,
        atoms: frozenset[int],
) -> None:
    """Remove inter-molecular interactions for a set of atoms.

    Args:
        system: The OpenMM representation of forces, constraints, and
            particles.
        atoms: The indices of atoms to remove inter-molecular
            interactions from.
    """
    _exclude_electrostatic(system, atoms)
    _exclude_lennard_jones(system, atoms)
    _exclude_custom_nonbonded(system, atoms)


def _exclude_electrostatic(
        system: openmm.System,
        atoms: frozenset[int],
) -> None:
    """Remove electrostatic interactions for a set of atoms.

    Args:
        system: The OpenMM representation of forces, constraints, and
            particles.
        atoms: The indices of atoms to remove electrostatic
            interactions from.
    """
    nonbonded_forces = [
        force for force in system.getForces()
        if isinstance(force, openmm.NonbondedForce)
    ]
    for force in nonbonded_forces:
        for i in atoms:
            q, s, e = force.getParticleParameters(i)
            q *= 0
            force.setParticleParameters(i, q, s, e)


def _exclude_lennard_jones(
        system: openmm.System,
        atoms: frozenset[int],
) -> None:
    """Remove Lennard-Jones interactions for a set of atoms.

    Args:
        system: The OpenMM representation of forces, constraints, and
            particles.
        atoms: The indices of atoms to remove Lennard-Jones
            interactions from.
    """
    nonbonded_forces = [
        force for force in system.getForces()
        if isinstance(force, openmm.NonbondedForce)
    ]
    for force in nonbonded_forces:
        for i in atoms:
            q, s, e = force.getParticleParameters(i)
            s /= s._value
            e *= 0
            force.setParticleParameters(i, q, s, e)


def _exclude_custom_nonbonded(
        system: openmm.System,
        atoms: frozenset[int],
) -> None:
    """Remove user-defined non-bonded interactions for a set of atoms.

    Args:
        system: The OpenMM representation of forces, constraints, and
            particles.
        atoms: The indices of atoms to remove user-defined non-bonded
            interactions from.
    """
    custom_nonbonded_forces = [
        force for force in system.getForces()
        if isinstance(force, openmm.CustomNonbondedForce)
    ]
    all_atoms = {i for i in range(system.getNumParticles())}
    other_atoms = all_atoms - atoms
    for force in custom_nonbonded_forces:
        force.addInteractionGroup(
            other_atoms,
            other_atoms,
        )
