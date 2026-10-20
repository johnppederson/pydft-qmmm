"""Periodic coupling assembly and classical gradients; no engine runs."""
from __future__ import annotations

import numpy as np
import openmm
import pytest

from pydft_qmmm import MMHamiltonian, QMHamiltonian, QMMMHamiltonian
from pydft_qmmm.interfaces import MMInterface, QMInterface


pytestmark = pytest.mark.usefixtures("reference_platform")


@pytest.fixture(params=["vasp", "sparc", "pyscf-pbc", "ksced"])
def qm(request, tmp_path):
    engine = request.param
    if engine == "vasp":
        options = dict(pp_path=str(tmp_path), directory=str(tmp_path))
    elif engine == "sparc":
        pytest.importorskip("sparc")
        options = dict(fd_grid=(24, 24, 24), directory=str(tmp_path))
    else:
        pytest.importorskip("pyscf")
        options = dict(basis="gth-szv", pseudo="gth-pbe", functional="pbe",
                       charge=0, multiplicity=1, mesh=(15, 15, 15))
        if engine == "ksced":
            options["ksced"] = dict(active_atoms=[0, 1, 2], frozen_atoms=[3, 4, 5])
        engine = "pyscf-pbc"
    return QMHamiltonian(interface=engine, **options)


def build(system, qm, long_range, mode="conservative", close_range="electrostatic"):
    mm = MMHamiltonian(
        interface="openmm",
        forcefield=["tests/data/spce.xml", "tests/data/spce_residues.xml"],
        nonbonded_cutoff=5.0, pme_gridnumber=30, pme_alpha=5.0,
    )
    coupling = QMMMHamiltonian(
        close_range, long_range, cutoff=2.5, pme_gridnumber=30,
        coupling_mode=mode,
    )
    calculator = (qm[:3] + mm[3:] + coupling).build_calculator(system)
    coupling.partition.generate_partition()
    potentials = [c.potential for c in calculator.calculators]
    return (calculator,
            next(p for p in potentials if isinstance(p, MMInterface)),
            next(p for p in potentials if isinstance(p, QMInterface)))


@pytest.mark.parametrize("long_range", ["none", "cutoff", "mechanical", "electrostatic"])
def test_conservative_engine_assembly(water4_system, qm, long_range):
    calculator, mm, engine = build(water4_system, qm, long_range)
    names = [c.name for c in calculator.calculators]
    assert "PMEExcluded" not in names
    assert ("PMENuclear" in names) == (
        long_range == "electrostatic" and not engine.applies_nuclear_potential()
    )
    assert len(engine.potentials) == (long_range == "electrostatic")
    np.testing.assert_array_equal(mm.base_force_mask, 1)
    assert not mm.aux_energy_group
    assert not mm.aux_forces_group
    nonbonded = next(f for f in mm.base_context.getSystem().getForces()
                     if isinstance(f, openmm.NonbondedForce))
    for atom in range(len(water4_system)):
        charge = nonbonded.getParticleParameters(atom)[0]._value
        assert water4_system.charges[atom] != 0
        assert charge == (0 if atom < 3 else water4_system.charges[atom])
    assert water4_system.select("subsystem II")
    assert water4_system.select("subsystem III")


@pytest.mark.parametrize("long_range", ["cutoff", "electrostatic"])
def test_conservative_classical_forces_match_energy(water4_system, qm, long_range):
    _, mm, _ = build(water4_system, qm, long_range)
    forces = mm.compute_forces()
    # QM oxygen has a nonzero LJ force; MM probes cover both II and III.
    assert np.linalg.norm(forces[0]) > 1e-3
    probes = [0] + [min(water4_system.select(f"subsystem {region}"))
                    for region in ("II", "III")]
    step = 1e-4
    baseline = np.asarray(water4_system.positions).copy()
    for atom in probes:
        for axis in range(3):
            pos = baseline.copy()
            pos[atom, axis] += step
            water4_system.positions = pos
            plus = mm.compute_energy()
            pos[atom, axis] -= 2 * step
            water4_system.positions = pos
            minus = mm.compute_energy()
            assert forces[atom, axis] == pytest.approx(
                -(plus - minus) / (2 * step), abs=2e-4,
            )
    water4_system.positions = baseline


def test_engine_force_mode_keeps_legacy_assembly(water4_system, qm):
    with pytest.warns(RuntimeWarning, match="nonconservative"):
        calculator, mm, _ = build(water4_system, qm, "electrostatic", "force")
    exclusion = next(c.potential for c in calculator.calculators
                     if c.name == "PMEExcluded")
    assert exclusion.include_forces is False
    assert exclusion.real_space_cutoff == 5.0
    np.testing.assert_array_equal(mm.base_force_mask[:3], 0)
    assert mm.aux_forces_group


def test_mechanical_engine_coupling_keeps_classical_charges(water4_system, qm):
    _, mm, _ = build(water4_system, qm, "mechanical", close_range="mechanical")
    nonbonded = next(f for f in mm.base_context.getSystem().getForces()
                     if isinstance(f, openmm.NonbondedForce))
    for atom in range(3):
        assert nonbonded.getParticleParameters(atom)[0]._value != 0
    np.testing.assert_array_equal(mm.base_force_mask, 1)
