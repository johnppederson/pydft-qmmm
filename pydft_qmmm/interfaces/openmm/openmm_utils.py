"""Functionality for performing exclusions and generating State objects.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

import openmm
from simtk.unit import elementary_charge
from simtk.unit import kilojoule_per_mole
from simtk.unit import nanometer

if TYPE_CHECKING:
    from typing import Any

    import numpy as np
    from numpy.typing import NDArray


_EXCEPTIONS_CACHE: dict[int, list[tuple[int, list[Any]]]] = {}


def _generate_state(
        omm_context: openmm.Context,
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
    return omm_context.getState(getEnergy=True, getForces=True, groups=groups)


def _exclude_intramolecular(
        omm_system: openmm.System,
        atoms: frozenset[int],
) -> None:
    """Remove intramolecular interactions for a set of atoms.

    Args:
        omm_system: The OpenMM representation of forces, constraints,
            and particles.
        atoms: The indices of atoms from which to remove
            intra-molecular interactions.
    """
    # Remove double-counted intramolecular interactions for QM atoms.
    _exclude_harmonic_bond(omm_system, atoms)
    _exclude_harmonic_angle(omm_system, atoms)
    _exclude_periodic_torsion(omm_system, atoms)
    _exclude_rb_torsion(omm_system, atoms)
    _exclude_custom_bond(omm_system, atoms)
    nonbonded_forces = [
        force for force in omm_system.getForces()
        if isinstance(force, openmm.NonbondedForce)
    ]
    custom_nonbonded_forces = [
        force for force in omm_system.getForces()
        if isinstance(force, openmm.CustomNonbondedForce)
    ]
    atom_list = sorted(atoms)
    for force in nonbonded_forces:
        for i, j in enumerate(atom_list):
            for k in atom_list[i+1:]:
                force.addException(j, k, 0, 1, 0, True)
    for force in custom_nonbonded_forces:
        exclusions = [
            set(
                force.getExclusionParticles(i),
            ) for i in range(force.getNumExclusions())
        ]
        for i, j in enumerate(atom_list):
            for k in atom_list[i+1:]:
                if not {j, k} in exclusions:
                    force.addExclusion(j, k)


def _exclude_harmonic_bond(
        omm_system: openmm.System,
        atoms: frozenset[int],
) -> None:
    """Remove harmonic bond interactions for a set of atoms.

    Args:
        omm_system: The OpenMM representation of forces, constraints,
            and particles.
        atoms: The indices of atoms from which to remove harmonic bond
            interactions.
    """
    harmonic_bond_forces = [
        force for force in omm_system.getForces()
        if isinstance(force, openmm.HarmonicBondForce)
    ]
    for force in harmonic_bond_forces:
        for i in range(force.getNumBonds()):
            *p, r0, k = force.getBondParameters(i)
            if not set(p) - atoms:
                k *= 0
                force.setBondParameters(i, *p, r0, k)


def _exclude_harmonic_angle(
        omm_system: openmm.System,
        atoms: frozenset[int],
) -> None:
    """Remove harmonic angle interactions for a set of atoms.

    Args:
        omm_system: The OpenMM representation of forces, constraints,
            and particles.
        atoms: The indices of atoms from which to remove harmonic angle
            interactions.
    """
    harmonic_angle_forces = [
        force for force in omm_system.getForces()
        if isinstance(force, openmm.HarmonicAngleForce)
    ]
    for force in harmonic_angle_forces:
        for i in range(force.getNumAngles()):
            *p, r0, k = force.getAngleParameters(i)
            if not set(p) - atoms:
                k *= 0
                force.setAngleParameters(i, *p, r0, k)


def _exclude_periodic_torsion(
        omm_system: openmm.System,
        atoms: frozenset[int],
) -> None:
    """Remove periodic torsion interactions for a set of atoms.

    Args:
        omm_system: The OpenMM representation of forces, constraints,
            and particles.
        atoms: The indices of atoms from which to remove periodic
            torsion interactions.
    """
    periodic_torsion_forces = [
        force for force in omm_system.getForces()
        if isinstance(force, openmm.PeriodicTorsionForce)
    ]
    for force in periodic_torsion_forces:
        for i in range(force.getNumTorsions()):
            *p, n, t, k = force.getTorsionParameters(i)
            if not set(p) - atoms:
                k *= 0
                force.setTorsionParameters(i, *p, n, t, k)


def _exclude_rb_torsion(
        omm_system: openmm.System,
        atoms: frozenset[int],
) -> None:
    """Remove Ryckaert-Bellemans interactions for a set of atoms.

    Args:
        omm_system: The OpenMM representation of forces, constraints,
            and particles.
        atoms: The indices of atoms from which to remove
            Ryckaert-Bellemans torsion interactions.
    """
    rb_torsion_forces = [
        force for force in omm_system.getForces()
        if isinstance(force, openmm.RBTorsionForce)
    ]
    for force in rb_torsion_forces:
        for i in range(force.getNumTorsions()):
            *p, c0, c1, c2, c3, c4, c5 = force.getTorsionParameters(i)
            if not set(p) - atoms:
                c0, c1, c2, c3, c4, c5 = (
                    x*0 for x in (c0, c1, c2, c3, c4, c5)
                )
                force.setTorsionParameters(i, *p, c0, c1, c2, c3, c4, c5)


def _exclude_custom_bond(
        omm_system: openmm.System,
        atoms: frozenset[int],
) -> None:
    """Remove Custom Bond forces for a set of atoms.

    Args:
        omm_system: The OpenMM representation of forces, constraints,
            and particles.
        atoms: The indices of atoms from which to remove Custom Bond
            interactions.
    """
    custom_bond_forces = [
        force for force in omm_system.getForces()
        if isinstance(force, openmm.CustomBondForce)
    ]
    for force in custom_bond_forces:
        for i in range(force.getNumBonds()):
            *p, params = force.getBondParameters(i)
            if not set(p) - atoms:
                params = tuple(x*0. for x in params)
                force.setBondParameters(i, *p, params)


def _real_electrostatic(
        omm_system: openmm.System,
        atoms: frozenset[int],
        const: int | float,
) -> list[openmm.customNonbondedForce]:
    """Add Coulomb interactions for a set of atoms.

    Args:
        omm_system: The OpenMM representation of forces, constraints,
            and particles.
        atoms: The indices of atoms for which to add a Coulomb
            interaction.
        const: A constant to multiply at the beginning of the
            coulomb expression.

    Returns:
        A list of OpenMM custom nonbonded forces implementing Coulomb
        interactions for the given atoms.
    """
    other_atoms = (
        {i for i in range(omm_system.getNumParticles())}
        - atoms
    )
    nonbonded_forces = [
        force for force in omm_system.getForces()
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
        cbf_force = openmm.CustomBondForce(
            f"{const}*138.935459977*n*q/r",
        )
        cbf_force.addPerBondParameter("q")
        cbf_force.addPerBondParameter("n")
        for atom in range(omm_system.getNumParticles()):
            q, _, _ = force.getParticleParameters(
                atom,
            )
            if atom in atoms:
                new_force.addParticle([q/elementary_charge, 1])
            else:
                new_force.addParticle([q/elementary_charge, 0])
        exclusions = []
        for i in range(force.getNumExceptions()):
            *p, q, _, _ = force.getExceptionParameters(i)
            exclusions.append(p)
            if (
                set(p) & atoms
                and q/elementary_charge/elementary_charge
            ):
                cbf_force.addBond(
                    *p, [q/elementary_charge/elementary_charge, 0],
                )
        for x in exclusions:
            new_force.addExclusion(*x)
        new_force.addInteractionGroup(
            atoms,
            other_atoms,
        )
        forces.append(new_force)
        if cbf_force.getNumBonds():
            forces.append(cbf_force)
    return forces


def _non_electrostatic(
        omm_system: openmm.System,
        atoms: frozenset[int],
) -> list[openmm.customNonbondedForce]:
    """Add a non-electrostatic interactions for a set of atoms.

    Args:
        omm_system: The OpenMM representation of forces, constraints,
            and particles.
        atoms: The indices of atoms for which to add non-electrostatic,
            non-bonded interactions.

    Returns:
        A list of OpenMM custom nonbonded forces implementing
        non-electrostatic, non-bonded interactions for the given atoms.
    """
    other_atoms = (
        {i for i in range(omm_system.getNumParticles())}
        - atoms
    )
    nonbonded_forces = [
        force for force in omm_system.getForces()
        if isinstance(force, openmm.NonbondedForce)
    ]
    custom_nonbonded_forces = [
        force for force in omm_system.getForces()
        if isinstance(force, openmm.CustomNonbondedForce)
    ]
    bonded_forces = [
        force for force in omm_system.getForces()
        if (
            not isinstance(force, openmm.CustomNonbondedForce)
            and not isinstance(force, openmm.NonbondedForce)
        )
    ]
    forces = []
    if custom_nonbonded_forces:
        for force in custom_nonbonded_forces:
            new_force = force.__copy__()
            new_force.setNonbondedMethod(0)
            forces.append(new_force)
    if bonded_forces:
        for force in bonded_forces:
            forces.append(force.__copy__())
    for force in nonbonded_forces:
        new_force = openmm.CustomNonbondedForce(
            """4*epsilon*((sigma/r)^12-(sigma/r)^6);
            sigma=0.5*(sigma1+sigma2);
            epsilon=sqrt(epsilon1*epsilon2)""",
        )
        new_force.addPerParticleParameter("epsilon")
        new_force.addPerParticleParameter("sigma")
        cbf_force = openmm.CustomBondForce(
            "4*epsilon*((sigma/r)^12-(sigma/r)^6)",
        )
        cbf_force.addPerBondParameter("epsilon")
        cbf_force.addPerBondParameter("sigma")
        for atom in range(omm_system.getNumParticles()):
            _, sigma, epsilon = force.getParticleParameters(
                atom,
            )
            new_force.addParticle(
                [epsilon / kilojoule_per_mole, sigma / nanometer],
            )
        exclusions = []
        for i in range(force.getNumExceptions()):
            *p, _, s, e = force.getExceptionParameters(i)
            exclusions.append(p)
            if set(p) & atoms and e / kilojoule_per_mole:
                cbf_force.addBond(*p, [e / kilojoule_per_mole, s / nanometer])
        for x in exclusions:
            new_force.addExclusion(*x)
        new_force.addInteractionGroup(
            atoms,
            other_atoms,
        )
        if force.getNonbondedMethod() != force.NoCutoff:
            # The auxiliary context also contains an uncut Coulomb force.
            # OpenMM CPU cannot mix cut and uncut CustomNonbondedForces.
            # Explicit LJ pairs preserve the base cutoff and periodic metric
            # without changing the Coulomb subtraction's distance convention.
            cutoff = force.getCutoffDistance() / nanometer
            expression = "step(rc-r)*4*epsilon*((sigma/r)^12-(sigma/r)^6)"
            if force.getUseSwitchingFunction():
                switch = force.getSwitchingDistance() / nanometer
                expression = (expression + "*(1-10*x^3+15*x^4-6*x^5);"
                              "x=min(1,max(0,(r-rs)/(rc-rs)))")
            pairs = openmm.CustomBondForce(expression)
            pairs.addGlobalParameter("rc", cutoff)
            if force.getUseSwitchingFunction():
                pairs.addGlobalParameter("rs", switch)
            pairs.addPerBondParameter("epsilon")
            pairs.addPerBondParameter("sigma")
            pairs.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())
            excluded_pairs = {frozenset(pair) for pair in exclusions}
            for i in sorted(atoms):
                ei, si = new_force.getParticleParameters(i)
                for j in sorted(other_atoms):
                    if frozenset((i, j)) in excluded_pairs:
                        continue
                    ej, sj = new_force.getParticleParameters(j)
                    epsilon = (ei * ej) ** 0.5
                    if epsilon:
                        pairs.addBond(i, j, [epsilon, 0.5 * (si + sj)])
            forces.append(pairs)
        else:
            forces.append(new_force)
        if cbf_force.getNumBonds():
            forces.append(cbf_force)
    return forces


def _exclude_intermolecular(
        omm_system: openmm.System,
        atoms: frozenset[int],
) -> None:
    """Remove inter-molecular interactions for a set of atoms.

    Args:
        omm_system: The OpenMM representation of forces, constraints,
            and particles.
        atoms: The indices of atoms from which to remove inter-molecular
            interactions.
    """
    _exclude_electrostatic(omm_system, atoms)
    _exclude_lennard_jones(omm_system, atoms)
    _exclude_custom_nonbonded(omm_system, atoms)


def _exclude_electrostatic(
        omm_system: openmm.System,
        atoms: frozenset[int],
) -> None:
    """Remove electrostatic interactions for a set of atoms.

    Args:
        omm_system: The OpenMM representation of forces, constraints,
            and particles.
        atoms: The indices of atoms from which to remove electrostatic
            interactions.
    """
    nonbonded_forces = [
        force for force in omm_system.getForces()
        if isinstance(force, openmm.NonbondedForce)
    ]
    for force in nonbonded_forces:
        for i in atoms:
            q, s, e = force.getParticleParameters(i)
            force.setParticleParameters(i, q*0, s, e)
        for i in range(force.getNumExceptions()):
            *p, q, s, e = force.getExceptionParameters(i)
            if set(p) & atoms:
                force.setExceptionParameters(i, *p, q*0, s, e)


def _exclude_lennard_jones(
        omm_system: openmm.System,
        atoms: frozenset[int],
) -> None:
    """Remove Lennard-Jones interactions for a set of atoms.

    Args:
        omm_system: The OpenMM representation of forces, constraints,
            and particles.
        atoms: The indices of atoms from which to remove Lennard-Jones
            interactions.
    """
    nonbonded_forces = [
        force for force in omm_system.getForces()
        if isinstance(force, openmm.NonbondedForce)
    ]
    for force in nonbonded_forces:
        for i in atoms:
            q, s, e = force.getParticleParameters(i)
            force.setParticleParameters(i, q, 1.*s.unit, e*0)
        for i in range(force.getNumExceptions()):
            *p, q, s, e = force.getExceptionParameters(i)
            if set(p) & atoms:
                force.setExceptionParameters(i, *p, q, 1.*s.unit, e*0)


def _exclude_custom_nonbonded(
        omm_system: openmm.System,
        atoms: frozenset[int],
) -> None:
    """Remove user-defined non-bonded interactions for a set of atoms.

    Args:
        omm_system: The OpenMM representation of forces, constraints,
            and particles.
        atoms: The indices of atoms from which to remove user-defined
            non-bonded interactions.
    """
    custom_nonbonded_forces = [
        force for force in omm_system.getForces()
        if isinstance(force, openmm.CustomNonbondedForce)
    ]
    all_atoms = {i for i in range(omm_system.getNumParticles())}
    other_atoms = all_atoms - atoms
    for force in custom_nonbonded_forces:
        force.addInteractionGroup(
            other_atoms,
            other_atoms,
        )


def _get_non_zero_exceptions(
        force: openmm.nonbondedForce,
) -> list[tuple[int, list[Any]]]:
    """Get the non-zero exceptions for a NonbondedForce object.

    This function makes use a cache in order to ensure that the
    exceptions update step is quicker.

    Args:
        force: The OpenMM NonbondedForce with exceptions to collect.

    Returns:
        A list of tuples containing non-zero exception indices and
        their associated parameters.
    """
    if (n := force.getForceGroup()) in _EXCEPTIONS_CACHE:
        return _EXCEPTIONS_CACHE[n]
    exceptions = []
    for i in range(force.getNumExceptions()):
        params = force.getExceptionParameters(i)
        if params[2]:
            exceptions.append((i, params))
    _EXCEPTIONS_CACHE[n] = exceptions
    return exceptions


def _update_exceptions(
        force: openmm.nonbondedForce,
        new_charges: NDArray[np.float64],
) -> None:
    """Update OpenMM NonbondedForce exceptions to match new charges.

    Args:
        force: The OpenMM NonbondedForce with exceptions to update.
        new_charges: The new partial charge (:math:`e`) of the atoms.
    """
    for exception in _get_non_zero_exceptions(force):
        i, params = exception
        q0, _, _ = force.getParticleParameters(params[0])
        q1, _, _ = force.getParticleParameters(params[1])
        qprod_old = q0 * q1 / (elementary_charge**2)
        qprod_new = new_charges[params[0]] * new_charges[params[1]]
        if not qprod_old:
            if qprod_new and not params[2]:
                warn(
                    (
                        f"Either or both of atoms {params[0]} and"
                        f" {params[1]} were previously set to zero, but"
                        " both are now nonzero.  These atoms also had a"
                        " non-zero NonbondedForce exception in OpenMM"
                        " at the beginning of the simulation; however,"
                        " a non-zero exception is not recoverable once"
                        " one or both atoms have been set to zero"
                        " charge.  Regardless of charge modifications"
                        " going forward, the Coulomb interaction will"
                        " remain zero between these atoms."
                    ),
                    RuntimeWarning,
                )
        else:
            params[2] *= (qprod_new / qprod_old)
            force.setExceptionParameters(i, *params)
