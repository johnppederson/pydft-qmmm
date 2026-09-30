"""Tests for reciprocal source forces in the Psi4 interface."""

import numpy as np
import pytest
import psi4

from pydft_qmmm.interfaces.psi4.psi4_factory import psi4_interface_factory


class RecordingPotential:
    """Record source-force inputs."""

    def __init__(self, size):
        self.size = size
        self.calls = []

    def compute_potential(self, coordinates):
        return np.zeros((len(coordinates), 1))

    def compute_source_forces(self, coordinates, weights):
        self.calls.append((coordinates.copy(), weights.copy()))
        return np.zeros((self.size, 3))


def test_psi4_factory_resets_an_inherited_read_guess(pyscf_water_system):
    """The factory does not inherit a stale orbital guess."""
    psi4.set_options({"guess": "read"})

    psi4_interface_factory(
        pyscf_water_system,
        basis="sto-3g",
        functional="pbe",
        charge=0,
        multiplicity=1,
        output_file="/dev/null",
    )

    assert psi4.core.get_global_option("GUESS") == "SAD"


@pytest.mark.parametrize(
    ("system_fixture", "multiplicity"),
    (("pyscf_pme_system", 1), ("pyscf_triplet_system", 3)),
)
def test_psi4_reuses_embpot_density_for_source_forces(
        request, system_fixture, multiplicity,
):
    """Psi4 passes its EMBPOT density to source forces."""
    system = request.getfixturevalue(system_fixture)
    potential = psi4_interface_factory(
        system,
        basis="sto-3g",
        functional="pbe",
        charge=0,
        multiplicity=multiplicity,
        output_file="/dev/null",
        scf_type="pk",
    )
    recorder = RecordingPotential(len(system.positions))
    potential.add_electronic_potential(recorder)

    potential.compute_forces()
    wfn = potential._generate_wavefunction()

    assert len(recorder.calls) == 1
    coordinates, source_charges = recorder.calls[0]
    assert coordinates.shape == (len(source_charges), 3)
    assert source_charges.sum() == pytest.approx(
        -(wfn.nalpha() + wfn.nbeta()), abs=2e-3,
    )
