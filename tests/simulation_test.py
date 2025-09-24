from __future__ import annotations

import numpy

from pydft_qmmm import QMMMHamiltonian
from pydft_qmmm import Simulation


def test_simulation_system_centering(
        spce_qmmm_system,
        spce_system,
        mm_spce,
        qm_water,
        verlet,
        no_logging,
):
    qmmm = QMMMHamiltonian("electrostatic", "electrostatic")
    total = mm_spce[3:] + qm_water[0:3] + qmmm
    _ = Simulation(
        system=spce_system,
        integrator=verlet,
        hamiltonian=total,
        **no_logging,
    )
    assert numpy.allclose(spce_system.positions, spce_qmmm_system.positions)
