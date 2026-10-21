"""Charge notifications must preserve explicitly removed electrostatics."""
from __future__ import annotations

import numpy as np
import openmm
from openmm import unit
import pytest

from pydft_qmmm import Atom, System
from pydft_qmmm.interfaces.openmm.openmm_interface import OpenMMPotential
from pydft_qmmm.interfaces.openmm import openmm_utils


@pytest.fixture
def make_interface(monkeypatch):
    monkeypatch.setattr(openmm_utils, "_EXCEPTIONS_CACHE", {})

    def build(charges, periodic):
        positions = np.array([[2., 3., 4.], [4., 3., 5.],
                              [7., 6., 4.], [8., 7., 6.]])
        system = System([
            Atom(element="H", position=position, charge=charge, mass=1.)
            for position, charge in zip(positions, charges)
        ], np.eye(3)*12.)
        base, aux = openmm.System(), openmm.System()
        force = openmm.NonbondedForce()
        correction = openmm.CustomNonbondedForce("nn*q1*q2/r;nn=n1*n2")
        correction.addPerParticleParameter("q")
        correction.addPerParticleParameter("n")
        for charge in charges:
            base.addParticle(1.)
            aux.addParticle(1.)
            force.addParticle(charge, .2, 0.)
            correction.addParticle([charge, 1.])
        # Exercise a removed exception and a retained scaled exception.
        for i, j in [(0, 1), (2, 3)]:
            force.addException(i, j, .5*charges[i]*charges[j], .2, 0.)
        if periodic:
            base.setDefaultPeriodicBoxVectors(*np.eye(3)*1.2)
            force.setNonbondedMethod(force.PME)
            force.setCutoffDistance(.5)
            force.setPMEParameters(3.5, 24, 24, 24)
        base.addForce(force)
        aux.addForce(correction)
        contexts = []
        for omm_system in (base, aux):
            context = openmm.Context(omm_system, openmm.VerletIntegrator(.001),
                                     openmm.Platform.getPlatformByName("Reference"))
            context.setPositions(positions*unit.angstrom)
            contexts.append(context)
        return OpenMMPotential(system, *contexts, np.ones((4, 3)),
                               np.ones((4, 3)), np.ones((4, 3)))

    return build


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("action", ["unchanged", "bulk", "inplace"])
def test_charge_updates_preserve_exclusions(make_interface, periodic, action):
    charges = np.array([1., -.2, .3, -.1])
    mm = make_interface(charges, periodic)
    # Populate exception bookkeeping before removing a charge, too.
    mm.system.charges = charges.copy()
    mm.zero_charges(frozenset({0}))
    desired = charges.copy()
    if action != "unchanged":
        desired[2:] += [.02, -.02]
    if action == "inplace":
        mm.system.charges[2] = desired[2]
        mm.system.charges[3] = desired[3]
    else:
        mm.system.charges = desired.copy()

    expected = make_interface(desired, periodic)
    expected.zero_charges(frozenset({0}))
    np.testing.assert_array_equal(mm.system.charges, desired)
    force = mm.base_context.getSystem().getForce(0)
    actual = np.array([force.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                       for i in range(4)])
    np.testing.assert_allclose(actual, [0., *desired[1:]], atol=1e-14)
    assert force.getExceptionParameters(0)[2].value_in_unit(unit.elementary_charge**2) == 0.
    assert force.getExceptionParameters(1)[2].value_in_unit(unit.elementary_charge**2) == pytest.approx(
        .5*desired[2]*desired[3], abs=1e-14)
    assert mm.compute_energy() == pytest.approx(expected.compute_energy(), abs=1e-8)
    np.testing.assert_allclose(mm.compute_forces(), expected.compute_forces(), atol=1e-8)
    # Auxiliary Coulomb corrections still use the original source charges.
    correction = mm.aux_context.getSystem().getForce(0)
    np.testing.assert_allclose([correction.getParticleParameters(i)[0] for i in range(4)], desired)


def test_intermolecular_exclusions_accumulate_per_interface(make_interface):
    charges = np.array([1., -.2, .3, -.1])
    mm = make_interface(charges, False)
    other = make_interface(charges, False)
    mm.zero_intermolecular(frozenset({0}))
    mm.zero_charges(frozenset({1}))
    mm.system.charges = charges.copy()
    other.system.charges = charges.copy()
    for interface, expected in [(mm, [0., 0., .3, -.1]), (other, charges)]:
        force = interface.base_context.getSystem().getForce(0)
        actual = [force.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                  for i in range(4)]
        np.testing.assert_allclose(actual, expected, atol=1e-14)
