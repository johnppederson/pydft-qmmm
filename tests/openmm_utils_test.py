from __future__ import annotations

import openmm
import pytest

from pydft_qmmm.interfaces.openmm.openmm_utils import _exclude_lennard_jones


def _system_with_sigma(sigma: float) -> openmm.System:
    system = openmm.System()
    system.addParticle(1.0)
    force = openmm.NonbondedForce()
    force.addParticle(0.0, sigma, 0.0)
    system.addForce(force)
    return system


def test_exclude_lennard_jones_handles_zero_sigma():
    system = _system_with_sigma(0.0)
    _exclude_lennard_jones(system, frozenset({0}))
    force = system.getForce(0)
    _, sigma, epsilon = force.getParticleParameters(0)
    assert sigma._value == pytest.approx(1.0)
    assert epsilon._value == pytest.approx(0.0)


def test_exclude_lennard_jones_normalizes_nonzero_sigma():
    system = _system_with_sigma(0.3)
    _exclude_lennard_jones(system, frozenset({0}))
    force = system.getForce(0)
    _, sigma, epsilon = force.getParticleParameters(0)
    assert sigma._value == pytest.approx(1.0)
    assert epsilon._value == pytest.approx(0.0)


@pytest.mark.parametrize('switching', [False, True])
def test_restored_lj_matches_base_cutoff_and_periodic_images(switching):
    """Force restoration must preserve the base LJ distance and switching rule."""
    import numpy as np
    from pydft_qmmm.interfaces.openmm.openmm_utils import _non_electrostatic

    base = openmm.System()
    aux = openmm.System()
    box = [openmm.Vec3(3, 0, 0), openmm.Vec3(0, 3, 0), openmm.Vec3(0, 0, 3)]
    for system in (base, aux):
        system.addParticle(1.)
        system.addParticle(1.)
        system.setDefaultPeriodicBoxVectors(*box)
    force = openmm.NonbondedForce()
    force.addParticle(0., 0.3, 0.6)
    force.addParticle(0., 0.3, 0.6)
    force.setNonbondedMethod(force.CutoffPeriodic)
    force.setCutoffDistance(1.)
    force.setUseDispersionCorrection(False)
    force.setUseSwitchingFunction(switching)
    force.setSwitchingDistance(0.8)
    base.addForce(force)
    for restored in _non_electrostatic(base, frozenset({0})):
        aux.addForce(restored)
    platform = openmm.Platform.getPlatformByName('Reference')
    contexts = [openmm.Context(system, openmm.VerletIntegrator(0.001), platform)
                for system in (base, aux)]
    # An ordinary pair, switching region, beyond cutoff, and across the cell.
    for x in (0.6, 1., 1.4, 2.7):
        values = []
        for context in contexts:
            context.setPositions([[0.1, 0., 0.], [x, 0., 0.]])
            state = context.getState(getEnergy=True, getForces=True)
            values.append((state.getPotentialEnergy()._value,
                           state.getForces(asNumpy=True)._value))
        assert values[1][0] == pytest.approx(values[0][0], abs=1e-10)
        np.testing.assert_allclose(values[1][1], values[0][1], rtol=0, atol=1e-10)
