from __future__ import annotations

import numpy
import pytest

from pydft_qmmm import QMMMHamiltonian
from pydft_qmmm import set_interfaces
from pydft_qmmm import Simulation

set_interfaces("psi4", "openmm")


def test_simulation_system_centering(
        spce_qmmm_system,
        spce_system,
        mm_spce,
        qm_water,
        verlet,
):
    qmmm = QMMMHamiltonian("electrostatic", "electrostatic")
    total = mm_spce[3:] + qm_water[0:3] + qmmm
    simulation = Simulation(
        system=spce_system,
        integrator=verlet,
        hamiltonian=total,
    )
    assert numpy.allclose(spce_system.positions, spce_qmmm_system.positions)
