"""LJ mixing, NBFix overrides, and group exclusions through MMHamiltonian."""
from itertools import combinations

import numpy as np
import openmm
from openmm import unit
import pytest

from pydft_qmmm import Atom, MMHamiltonian, System
from pydft_qmmm.utils import Subsystem
from pydft_qmmm.interfaces.openmm.lj_mixing import apply_geometric_lj
from pydft_qmmm.interfaces.openmm.openmm_utils import _exclude_nonbonded


@pytest.fixture
def forcefield(tmp_path):
    path = tmp_path / 'pairs.xml'
    path.write_text('''<ForceField>
<AtomTypes>
<Type name="a" class="A" element="H" mass="1"/>
<Type name="b" class="B" element="H" mass="1"/>
</AtomTypes>
<Residues><Residue name="PAIR">
<Atom name="A" type="a"/><Atom name="B" type="b"/>
</Residue></Residues>
<NonbondedForce coulomb14scale="0.5" lj14scale="0.5">
<Atom type="a" charge="0" sigma="0.2" epsilon="1"/>
<Atom type="b" charge="0" sigma="0.8" epsilon="4"/>
</NonbondedForce></ForceField>''')
    return path


def make_system():
    return System([Atom(element='H', name=name, residue=i // 2, residue_name='PAIR',
                        position=np.array([i * 7., 0., 0.]))
                   for i, name in enumerate(['A', 'B', 'A', 'B'])])


@pytest.mark.parametrize('options,ab_sigma', [
    ({}, .4), ({'lj_lorentz_classes': ('B',)}, .5),
])
def test_geometric_mixing_and_group_exclusions(forcefield, options, ab_sigma):
    system = make_system()
    calculator = MMHamiltonian(
        forcefield=str(forcefield), nonbonded_method='NoCutoff', platform='Reference',
        subsystem=Subsystem.II, lj_mixing='geometric',
        nonbonded_exclusions=('residue 0', 'atom 0 1'), **options,
    ).build_calculator(system)
    assert all(s == Subsystem.II for s in system.subsystems)
    expected = 0.
    for i, j in combinations(range(4), 2):
        if (i, j) == (0, 1):
            continue
        sigma = (.2 if i % 2 == 0 else .8) if i % 2 == j % 2 else ab_sigma
        epsilon = (1. if i % 2 == 0 else 4.) if i % 2 == j % 2 else 2.
        r = .7 * (j - i)
        expected += 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)
    result = calculator.calculate()
    assert result.energy == pytest.approx(expected, abs=1e-10)
    # A displacement checks that the returned forces differentiate the same energy.
    h = 1e-4
    system.positions[3, 0] += h
    plus = calculator.calculate().energy
    system.positions[3, 0] -= 2 * h
    minus = calculator.calculate().energy
    assert result.forces[3, 0] == pytest.approx(-(plus - minus) / (2*h), abs=1e-7)


def test_nbfix_and_zero_classes(forcefield):
    forcefield.write_text(forcefield.read_text().replace('</NonbondedForce>',
        '<NBFixPair class1="A" class2="B" sigma="0.3" epsilon="0.7"/></NonbondedForce>'))
    system = make_system()
    calc = MMHamiltonian(forcefield=str(forcefield), platform='Reference',
                        nonbonded_method='NoCutoff', lj_mixing='geometric',
                        lj_zero_classes=('B',)).build_calculator(system)
    # Only the A-A pair survives, even though XML defines an A-B override.
    expected = 4 * ((.2 / 1.4)**12 - (.2 / 1.4)**6)
    assert calc.calculate().energy == pytest.approx(expected, abs=1e-12)
    system2 = make_system()
    calc2 = MMHamiltonian(forcefield=str(forcefield), platform='Reference',
                         nonbonded_method='NoCutoff', lj_mixing='geometric').build_calculator(system2)
    expected = 0.
    for i, j in combinations(range(4), 2):
        sigma, epsilon = ((.3, .7) if i % 2 != j % 2 else
                          ((.2, 1.) if i % 2 == 0 else (.8, 4.)))
        r = .7 * (j - i)
        expected += 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)
    assert calc2.calculate().energy == pytest.approx(expected, abs=1e-10)


def test_preserve_coulomb_and_14_scaling(forcefield):
    topology = openmm.app.Topology()
    residue = topology.addResidue('PAIR', topology.addChain())
    for name in ['A', 'B']:
        topology.addAtom(name, openmm.app.Element.getBySymbol('H'), residue)
    system = openmm.System()
    force = openmm.NonbondedForce()
    for charge, sigma, epsilon in [(1., .2, 1.), (-1., .8, 4.)]:
        system.addParticle(1.)
        force.addParticle(charge, sigma, epsilon)
    force.addException(0, 1, -.5, .5, 1.)
    system.addForce(force)
    apply_geometric_lj(system, topology, [str(forcefield)])
    p, q, charge, sigma, epsilon = force.getExceptionParameters(0)
    assert charge.value_in_unit(unit.elementary_charge**2) == -.5
    assert sigma.value_in_unit(unit.nanometer) == pytest.approx(.4)
    assert epsilon.value_in_unit(unit.kilojoule_per_mole) == 1.
    assert force.getParticleParameters(0)[0].value_in_unit(unit.elementary_charge) == 1.
    custom = system.getForce(1)
    assert custom.getNumExclusions() == 1
    _exclude_nonbonded(system, frozenset([0, 1]))
    _exclude_nonbonded(system, frozenset([0, 1]))
    assert custom.getNumExclusions() == 1


def test_default_mixing_and_region_unchanged(forcefield):
    system = make_system()
    calc = MMHamiltonian(forcefield=str(forcefield), platform='Reference',
                         nonbonded_method='NoCutoff').build_calculator(system)
    assert all(s == Subsystem.III for s in system.subsystems)
    assert not any(isinstance(f, openmm.CustomNonbondedForce)
                   for f in calc.potential.base_context.getSystem().getForces())
