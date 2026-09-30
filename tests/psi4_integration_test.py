"""End-to-end conservative Psi4/OpenMM coupling tests."""

import numpy as np
import pytest

from pydft_qmmm import QMMMHamiltonian
from pydft_qmmm import QMHamiltonian
from pydft_qmmm import Simulation
from pydft_qmmm import VerletIntegrator
from pydft_qmmm.utils import numerical_gradient


def _build(system, mm):
    system.box[:] = np.eye(3) * 24.0
    qm = QMHamiltonian(
        interface="psi4",
        basis="sto-3g",
        functional="pbe",
        charge=0,
        multiplicity=1,
        output_file="/dev/null",
        guess="sad",
        scf_type="pk",
        e_convergence=1e-9,
        d_convergence=1e-9,
        dft_radial_points=99,
        dft_spherical_points=590,
    )
    coupling = QMMMHamiltonian(
        "electrostatic",
        "electrostatic",
        partition=None,
        pme_gridnumber=48,
    )
    return (mm[3:] + qm[0:3] + coupling).build_calculator(system)


def test_psi4_openmm_total_forces_match_total_energy(
        pyscf_pme_system, mm_pyscf_spce,
):
    """Psi4 total forces match total-energy derivatives."""
    calculator = _build(pyscf_pme_system, mm_pyscf_spce)
    analytical = calculator.calculate().forces
    numerical = numerical_gradient(calculator, {0, 6})
    assert -analytical[[0, 6]] == pytest.approx(numerical, abs=0.5)


def test_psi4_openmm_total_force_is_translationally_invariant(
        pyscf_pme_system, mm_pyscf_spce,
):
    """Psi4 total forces are translationally invariant."""
    calculator = _build(pyscf_pme_system, mm_pyscf_spce)
    net_force = calculator.calculate().forces.sum(axis=0)
    assert net_force == pytest.approx(np.zeros(3), abs=0.05)


def test_psi4_nve_drift_decreases_with_timestep(
        pyscf_pme_system, mm_pyscf_spce, no_logging,
):
    """Psi4 NVE drift decreases with the timestep."""
    initial_positions = np.asarray(pyscf_pme_system.positions).copy()
    initial_velocities = np.asarray(pyscf_pme_system.velocities).copy()
    drifts = []
    for timestep, steps in ((0.5, 3), (0.25, 6)):
        pyscf_pme_system.positions[:] = initial_positions
        pyscf_pme_system.velocities[:] = initial_velocities
        simulation = Simulation(
            system=pyscf_pme_system,
            integrator=VerletIntegrator(timestep),
            calculator=_build(pyscf_pme_system, mm_pyscf_spce),
            **no_logging,
        )
        energies = []
        for _ in range(steps):
            simulation.run_dynamics(1)
            energies.append(simulation.energy["Total Energy"])
        drifts.append(max(energies) - min(energies))
    assert drifts[0] < 0.5
    assert drifts[1] < drifts[0]
