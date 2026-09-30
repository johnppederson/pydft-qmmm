"""End-to-end PySCF/OpenMM QM/MM coupling tests."""
from __future__ import annotations

import numpy as np
import pytest

from pydft_qmmm import QMMMHamiltonian
from pydft_qmmm import Simulation
from pydft_qmmm import VerletIntegrator
from pydft_qmmm.utils import numerical_gradient

FORCE_TOLERANCE = 0.05
PROBE_ATOMS = (0, 3, 6)


def build(system, mm, qm, long_range, coupling_mode="conservative"):
    coupling = QMMMHamiltonian(
        "electrostatic",
        long_range,
        cutoff=6.0,
        pme_gridnumber=48,
        coupling_mode=coupling_mode,
    )
    return (mm[3:] + qm[0:3] + coupling).build_calculator(system)


def test_conservative_pme_removes_openmm_qm_charges_without_force_masks(
        pyscf_openmm_system, mm_pyscf_spce, qm_pyscf_water,
):
    """Conservative PME removes QM charges without masking forces."""
    calculator = build(
        pyscf_openmm_system, mm_pyscf_spce, qm_pyscf_water, "electrostatic",
    )
    names = [component.name for component in calculator.calculators]
    openmm = next(
        component.potential for component in calculator.calculators
        if component.name == "OpenMM"
    )
    qm_atoms = sorted(pyscf_openmm_system.select("subsystem I"))
    nonbonded = next(
        force for force in openmm.base_context.getSystem().getForces()
        if force.__class__.__name__ == "NonbondedForce"
    )

    assert "PMENuclear" in names
    assert "PMEExcluded" not in names
    assert np.all(openmm.base_force_mask == 1)
    for atom in qm_atoms:
        assert pyscf_openmm_system.charges[atom] != 0.0
        charge, _, _ = nonbonded.getParticleParameters(atom)
        assert charge._value == 0.0


def test_force_mode_preserves_historical_pme_assembly(
        pyscf_openmm_system, mm_pyscf_spce, qm_pyscf_water,
):
    """Force mode preserves the historical PME assembly."""
    with pytest.warns(RuntimeWarning, match="nonconservative"):
        calculator = build(
            pyscf_openmm_system,
            mm_pyscf_spce,
            qm_pyscf_water,
            "electrostatic",
            coupling_mode="force",
        )
    names = [component.name for component in calculator.calculators]
    openmm = next(
        component.potential for component in calculator.calculators
        if component.name == "OpenMM"
    )
    qm_atoms = sorted(pyscf_openmm_system.select("subsystem I"))

    assert "PMEExcluded" in names
    assert np.all(openmm.base_force_mask[qm_atoms] == 0)


@pytest.mark.parametrize("long_range", ["cutoff", "electrostatic"])
def test_pyscf_openmm_qmmm_force_matches_energy(
        pyscf_openmm_system, mm_pyscf_spce, qm_pyscf_water, long_range,
):
    calculator = build(
        pyscf_openmm_system, mm_pyscf_spce, qm_pyscf_water, long_range,
    )
    analytical = calculator.calculate().forces
    numerical = numerical_gradient(calculator, set(PROBE_ATOMS))
    assert -analytical[list(PROBE_ATOMS)] == pytest.approx(
        numerical, abs=FORCE_TOLERANCE,
    )


@pytest.mark.parametrize("long_range", ["cutoff", "electrostatic"])
def test_pyscf_qmmm_net_force_vanishes(
        pyscf_openmm_system, mm_pyscf_spce, qm_pyscf_water, long_range,
):
    calculator = build(
        pyscf_openmm_system, mm_pyscf_spce, qm_pyscf_water, long_range,
    )
    net = calculator.calculate().forces.sum(axis=0)
    assert net == pytest.approx(0, abs=FORCE_TOLERANCE)


def test_subsystems_are_all_populated(
        pyscf_openmm_system, mm_pyscf_spce, qm_pyscf_water,
):
    calculator = build(
        pyscf_openmm_system, mm_pyscf_spce, qm_pyscf_water, "electrostatic",
    )
    calculator.calculate()
    for subsystem in ("subsystem I", "subsystem II", "subsystem III"):
        assert pyscf_openmm_system.select(subsystem), subsystem
    assert np.count_nonzero(pyscf_openmm_system.charges) == len(
        pyscf_openmm_system.charges,
    )


def test_reciprocal_coupling_adds_no_double_counting(
        pyscf_openmm_system, mm_pyscf_spce, qm_pyscf_water,
):
    """Reciprocal coupling adds no force residual."""
    residuals = []
    for long_range in ("cutoff", "electrostatic"):
        calculator = build(
            pyscf_openmm_system, mm_pyscf_spce, qm_pyscf_water, long_range,
        )
        total = calculator.calculate().forces
        numerical = numerical_gradient(calculator, set(PROBE_ATOMS))
        residuals.append(total[list(PROBE_ATOMS)] + numerical)
    assert residuals[1] == pytest.approx(residuals[0], abs=1e-3)


def test_embedding_components_are_reported_once(
        pyscf_openmm_system, mm_pyscf_spce, qm_pyscf_water,
):
    """Embedding components are reported once."""
    calculator = build(
        pyscf_openmm_system, mm_pyscf_spce, qm_pyscf_water, "electrostatic",
    )
    total = calculator.calculate()
    pyscf = [c for c in calculator.calculators if c.name == "PySCF"][0]
    results = pyscf.calculate()
    components = results.components
    assert set(components) == {"Finite Embedding", "Reciprocal Embedding"}
    # Both terms are already in the SCF total.
    assert results.energy == pytest.approx(
        pyscf.potential.compute_energy(), abs=1e-9,
    )
    assert abs(components["Finite Embedding"]) > 1.0
    assert abs(components["Reciprocal Embedding"]) > 1.0
    per_calculator = {
        name: value for name, value in total.components.items()
        if not name.startswith(".")
    }
    assert total.energy == pytest.approx(
        sum(per_calculator.values()), abs=1e-6,
    )


def test_one_scf_serves_energy_and_forces(
        pyscf_openmm_system, mm_pyscf_spce, qm_pyscf_water,
):
    calculator = build(
        pyscf_openmm_system, mm_pyscf_spce, qm_pyscf_water, "electrostatic",
    )
    potential = [
        c.potential for c in calculator.calculators if c.name == "PySCF"
    ][0]
    potential.compute_energy()
    solver = potential.method[0]
    potential.compute_forces()
    assert potential.method[0] is solver
    potential.compute_components()
    assert potential.method[0] is solver


@pytest.mark.parametrize(
    "observable", ["positions", "charges", "box", "subsystems"],
)
def test_state_changes_invalidate_the_cache(
        pyscf_openmm_system, mm_pyscf_spce, qm_pyscf_water, observable,
):
    calculator = build(
        pyscf_openmm_system, mm_pyscf_spce, qm_pyscf_water, "electrostatic",
    )
    potential = [
        c.potential for c in calculator.calculators if c.name == "PySCF"
    ][0]
    potential.compute_energy()
    solver = potential.method[0]
    if observable == "positions":
        pyscf_openmm_system.positions[9, 0] += 0.1
    elif observable == "charges":
        pyscf_openmm_system.charges[9] *= 1.01
    elif observable == "box":
        pyscf_openmm_system.box[:] = np.eye(3) * 24.5
    else:
        pyscf_openmm_system.subsystems[9] = (
            pyscf_openmm_system.subsystems[3]
        )
    potential.compute_energy()
    assert potential.method[0] is not solver


def test_pyscf_nve_energy_drift(
        pyscf_openmm_system, mm_pyscf_spce, qm_pyscf_water, no_logging,
):
    """NVE drift must decrease when the timestep is halved."""
    initial_positions = np.asarray(pyscf_openmm_system.positions).copy()
    initial_velocities = np.asarray(pyscf_openmm_system.velocities).copy()
    drifts = []
    for timestep, steps in ((0.5, 5), (0.25, 10)):
        pyscf_openmm_system.positions[:] = initial_positions
        pyscf_openmm_system.velocities[:] = initial_velocities
        coupling = QMMMHamiltonian(
            "electrostatic",
            "electrostatic",
            cutoff=6.0,
            pme_gridnumber=48,
        )
        total = mm_pyscf_spce[3:] + qm_pyscf_water[0:3] + coupling
        simulation = Simulation(
            system=pyscf_openmm_system,
            integrator=VerletIntegrator(timestep),
            hamiltonian=total,
            **no_logging,
        )
        energies = []
        for _ in range(steps):
            simulation.run_dynamics(1)
            energies.append(simulation.energy["Total Energy"])
        drifts.append(max(energies) - min(energies))
    # Measured 0.5 fs drift is 0.177 kJ/mol on the compute-node fixture.
    assert drifts[0] < 0.2, drifts
    assert drifts[1] < drifts[0], drifts
