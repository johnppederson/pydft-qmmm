"""Regression tests for conservative PME exclusion forces."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pydft_qmmm import Atom
from pydft_qmmm import System
from pydft_qmmm import MMHamiltonian, QMHamiltonian, QMMMHamiltonian
from pydft_qmmm.plugins import CentroidPartition
from pydft_qmmm.potentials.pme_potential import PMEExcludedPotential
from pydft_qmmm.utils import Subsystem


def _excluded_potential() -> PMEExcludedPotential:
    atoms = [
        Atom(
            position=np.array([2.0, 2.0, 2.0]),
            charge=-0.8,
            element="O",
            subsystem=Subsystem.I,
        ),
        Atom(
            position=np.array([3.0, 2.0, 2.0]),
            charge=0.4,
            element="H",
            subsystem=Subsystem.II,
        ),
        Atom(
            position=np.array([7.0, 6.0, 5.0]),
            charge=0.4,
            element="H",
            subsystem=Subsystem.III,
        ),
    ]
    system = System(atoms, box=np.diag([10.0, 10.0, 10.0]))
    return PMEExcludedPotential(system, 0.4, (20, 20, 20), 6)


def test_excluded_pme_force_differentiates_energy_on_field_source() -> None:
    """Moving a field-source atom must produce its energy derivative."""
    potential = _excluded_potential()
    atom = 2
    step = 0.001
    numerical = np.zeros(3)
    for axis in range(3):
        potential.system.positions[atom, axis] += step
        plus = potential.compute_energy()
        potential.system.positions[atom, axis] -= 2 * step
        minus = potential.compute_energy()
        potential.system.positions[atom, axis] += step
        numerical[axis] = -(plus - minus) / (2 * step)

    assert np.abs(numerical).max() > 1e-3
    assert potential.compute_forces()[atom] == pytest.approx(
        numerical, abs=1e-4,
    )


def test_force_mixing_exclusion_preserves_energy_without_extra_forces() -> None:
    """Eq. 9 energy corrections must not restore excluded classical forces."""
    differentiated = _excluded_potential()
    energy_only = replace(differentiated, include_forces=False)
    assert np.abs(differentiated.compute_forces()).max() > 1e-3
    for displacement in (0.0, 0.0025, -0.0025):
        differentiated.system.positions[0, 0] = 2.0 + displacement
        assert energy_only.compute_energy() == differentiated.compute_energy()
        np.testing.assert_array_equal(
            energy_only.compute_forces(), np.zeros((3, 3)),
        )


@pytest.mark.parametrize("atom", [1, 2])
def test_engine_exclusion_force_differentiates_energy_on_mm_atoms(atom) -> None:
    """Engine force mixing zeroes subsystem I only; MM atoms keep the gradient."""
    potential = replace(
        _excluded_potential(), include_forces=False, real_space_cutoff=7.5,
    )
    # Off the half-box plane, where the nearest image would flip mid-step.
    positions = np.asarray(potential.system.positions).copy()
    positions[2] = [6.5, 6.0, 5.0]
    potential.system.positions = positions
    step = 0.001
    numerical = np.zeros(3)
    for axis in range(3):
        potential.system.positions[atom, axis] += step
        plus = potential.compute_energy()
        potential.system.positions[atom, axis] -= 2 * step
        minus = potential.compute_energy()
        potential.system.positions[atom, axis] += step
        numerical[axis] = -(plus - minus) / (2 * step)

    forces = potential.compute_forces()
    assert np.abs(numerical).max() > 1e-3
    assert forces[atom] == pytest.approx(numerical, abs=1e-4)
    np.testing.assert_array_equal(forces[0], np.zeros(3))


@pytest.mark.parametrize("engine", ["vasp", "pyscf-pbc"])
def test_engine_coupling_uses_energy_only_exclusion(
        spce_system, tmp_path, engine,
) -> None:
    """Periodic engines must share the energy-only exclusion correction."""
    options = ({"pp_path": str(tmp_path)} if engine == "vasp" else {
        "basis": "gth-dzvp", "pseudo": "gth-pbe", "functional": "pbe",
        "charge": 0, "multiplicity": 1, "ke_cutoff": 200.,
    })
    qm = QMHamiltonian(interface=engine, **options)
    mm = MMHamiltonian(
        interface="openmm",
        forcefield=["tests/data/spce.xml", "tests/data/spce_residues.xml"],
        nonbonded_method="PME", nonbonded_cutoff=7.0,
        pme_gridnumber=(40, 40, 40), pme_alpha=5.0,
    )
    coupling = QMMMHamiltonian(
        "electrostatic", "electrostatic",
        partition=CentroidPartition("all", 6.0),
        pme_gridnumber=(40, 40, 40), pme_alpha=0.5,
    )
    calculator = (qm[:3] + mm[3:] + coupling).build_calculator(spce_system)
    exclusions = [
        c.potential for c in calculator.calculators
        if isinstance(c.potential, PMEExcludedPotential)
    ]
    assert len(exclusions) == 1
    assert exclusions[0].include_forces is False


def test_periodic_force_mixing_classical_qm_gradient(spce_system, monkeypatch):
    """Periodic self images, erfc tails, and LJ must match the masked QM force."""
    import openmm
    from pydft_qmmm.interfaces.openmm import openmm_factory

    def reference_context(omm_system, modeller):
        context = openmm.Context(omm_system, openmm.VerletIntegrator(0.001),
                                 openmm.Platform.getPlatformByName('Reference'))
        context.setPositions(modeller.positions)
        return context

    monkeypatch.setattr(openmm_factory, '_build_omm_context', reference_context)
    qm = QMHamiltonian(interface='pyscf-pbc', basis='gth-dzvp', pseudo='gth-pbe',
                       functional='pbe', charge=0, multiplicity=1, ke_cutoff=200.)
    mm = MMHamiltonian(interface='openmm',
                       forcefield=['tests/data/spce.xml', 'tests/data/spce_residues.xml'],
                       nonbonded_method='PME', nonbonded_cutoff=7.,
                       pme_gridnumber=(80, 80, 80), pme_alpha=5.)
    partition = CentroidPartition('all', 6.)
    with pytest.warns(RuntimeWarning, match='nonconservative'):
        coupling = QMMMHamiltonian('electrostatic', 'electrostatic', partition=partition,
                                   pme_gridnumber=(80, 80, 80), pme_alpha=0.5,
                                   coupling_mode='force')
    calculator = (qm[:3] + mm[3:] + coupling).build_calculator(spce_system)
    partition.generate_partition()
    classical = [sub.potential for sub in calculator.calculators[1:]]
    exclusion = next(p for p in classical if isinstance(p, PMEExcludedPotential))
    assert exclusion.real_space_cutoff == pytest.approx(7.)
    assert not exclusion.include_forces
    baseline = np.asarray(spce_system.positions).copy()
    analytic = -sum(p.compute_forces() for p in classical)[:3]
    step = 0.000625
    numerical = np.zeros((3, 3))
    for atom in range(3):
        for axis in range(3):
            pos = baseline.copy(); pos[atom, axis] += step
            spce_system.positions = pos
            plus = sum(p.compute_energy() for p in classical)
            pos = baseline.copy(); pos[atom, axis] -= step
            spce_system.positions = pos
            minus = sum(p.compute_energy() for p in classical)
            numerical[atom, axis] = (plus-minus)/(2*step)
    spce_system.positions = baseline
    np.testing.assert_allclose(analytic, numerical, rtol=0, atol=3e-4)
