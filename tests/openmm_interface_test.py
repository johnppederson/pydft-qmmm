"""Tests for energy-consistent OpenMM exclusions."""

import numpy as np
import openmm
from openmm import unit
import pytest

from pydft_qmmm.interfaces.openmm import openmm_utils


def _state(system, positions):
    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    context = openmm.Context(
        system,
        integrator,
        openmm.Platform.getPlatformByName("Reference"),
    )
    context.setPositions(positions)
    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.angstrom,
    )
    return energy, np.asarray(forces)


def _two_particle_system(*, epsilon=0.0, exception=False, periodic=False):
    system = openmm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    force = openmm.NonbondedForce()
    force.addParticle(1.0, 0.3, epsilon)
    force.addParticle(-1.0, 0.3, epsilon)
    if exception:
        force.addException(0, 1, -0.5, 0.25, epsilon / 2)
    if periodic:
        system.setDefaultPeriodicBoxVectors(
            openmm.Vec3(2, 0, 0),
            openmm.Vec3(0, 2, 0),
            openmm.Vec3(0, 0, 2),
        )
        force.setNonbondedMethod(openmm.NonbondedForce.PME)
        force.setCutoffDistance(0.9)
    else:
        force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    system.addForce(force)
    return system, force


def test_zero_charges_removes_coulomb_energy_and_both_forces():
    """Charge removal removes Coulomb energy and forces."""
    positions = np.array([[0, 0, 0], [4, 0, 0]]) * unit.angstrom
    system, _ = _two_particle_system()
    before_energy, before_forces = _state(system, positions)

    openmm_utils._exclude_electrostatic(system, frozenset({0}))
    after_energy, after_forces = _state(system, positions)

    assert abs(before_energy) > 1.0
    assert np.linalg.norm(before_forces) > 1.0
    assert after_energy == pytest.approx(0.0, abs=1e-10)
    assert after_forces == pytest.approx(0.0, abs=1e-10)


def test_zero_charges_preserves_exception_lennard_jones_parameters():
    """Charge removal preserves exception Lennard-Jones parameters."""
    system, force = _two_particle_system(epsilon=0.4, exception=True)
    _, _, _, sigma_before, epsilon_before = force.getExceptionParameters(0)

    openmm_utils._exclude_electrostatic(system, frozenset({0}))

    _, _, charge_product, sigma_after, epsilon_after = (
        force.getExceptionParameters(0)
    )
    assert charge_product.value_in_unit(unit.elementary_charge**2) == 0.0
    assert sigma_after == sigma_before
    assert epsilon_after == epsilon_before


def test_pme_forces_after_zero_charges_are_energy_derivatives():
    """PME forces remain energy derivatives after charge removal."""
    positions = np.array([[5, 5, 5], [9, 5, 5]], dtype=float)
    system, _ = _two_particle_system(periodic=True)
    openmm_utils._exclude_electrostatic(system, frozenset({0}))
    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    context = openmm.Context(
        system,
        integrator,
        openmm.Platform.getPlatformByName("Reference"),
    )

    def evaluate(coordinates):
        context.setPositions(coordinates * unit.angstrom)
        state = context.getState(getEnergy=True, getForces=True)
        return (
            state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
            np.asarray(state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.angstrom,
            )),
        )

    energy, analytical = evaluate(positions)
    step = 1e-4
    numerical = np.zeros_like(analytical)
    for atom in (0, 1):
        for axis in range(3):
            plus = positions.copy()
            minus = positions.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            e_plus, _ = evaluate(plus)
            e_minus, _ = evaluate(minus)
            numerical[atom, axis] = -(e_plus - e_minus) / (2 * step)

    # PME retains the remaining ion's background energy.
    assert np.isfinite(energy)
    assert analytical == pytest.approx(numerical, abs=1e-4)
