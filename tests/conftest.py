from __future__ import annotations

import json

import numpy as np
import pytest

from pydft_qmmm import MMHamiltonian
from pydft_qmmm import QMHamiltonian
from pydft_qmmm import System
from pydft_qmmm import VerletIntegrator
from pydft_qmmm.plugins import SETTLE
from pydft_qmmm.utils import Subsystem


@pytest.fixture
def spce_system():
    return System.load(
        "tests/data/spce_qmmm.pdb",
    )


@pytest.fixture
def spce_qmmm_system(spce_system):
    with open("tests/data/spce_qmmm_region_ii.json") as fh:
        embedding_list = json.load(fh)
    for atom in embedding_list:
        spce_system.subsystems[atom] = Subsystem.II
    return spce_system


@pytest.fixture
def qm_water():
    return QMHamiltonian(
        basis="def2-SVP",
        functional="PBE",
        charge=0,
        multiplicity=1,
        guess="read",
    )


@pytest.fixture
def mm_spce():
    return MMHamiltonian(
        forcefield=[
            "tests/data/spce.xml",
            "tests/data/spce_residues.xml",
        ],
        pme_gridnumber=30,
        pme_alpha=5.0,
    )


@pytest.fixture
def mm_spce_no_lj():
    return MMHamiltonian(
        forcefield=[
            "tests/data/spce_no_lj.xml",
            "tests/data/spce_residues.xml",
        ],
        pme_gridnumber=30,
        pme_alpha=5.0,
    )


@pytest.fixture
def spce_plugins():
    return [SETTLE()]


@pytest.fixture
def verlet():
    return VerletIntegrator(1)


@pytest.fixture
def no_logging():
    return {
        "log_write": False,
        "csv_write": False,
        "dcd_write": False,
    }


@pytest.fixture
def reference_platform(monkeypatch):
    """Run OpenMM on the Reference (double-precision) platform."""
    import openmm
    from pydft_qmmm.interfaces.openmm import openmm_factory

    def context(system, modeller, platform="CPU", precision=None):
        result = openmm.Context(
            system, openmm.VerletIntegrator(0.001),
            openmm.Platform.getPlatformByName("Reference"),
        )
        result.setPositions(modeller.positions)
        return result

    monkeypatch.setattr(openmm_factory, "_build_omm_context", context)


@pytest.fixture
def water4_system():
    """Four SPC/E waters in a 12 A cubic cell.

    Small enough for fast periodic QM/MM coupling tests.
    """
    return System.load("tests/data/water4.pdb")


# PySCF fixtures.

# SPC/E charges from tests/data/spce_no_lj.xml.
SPCE_CHARGES = {"O": -0.8476, "H": 0.4238}

# SPC/E water at the origin: r(OH) = 1 A, HOH = 109.47 deg.
WATER_GEOMETRY = np.array([
    [0.00000, 0.0, 0.00000],
    [0.81650, 0.0, 0.57735],
    [-0.81650, 0.0, 0.57735],
])


def build_water_system(placements, box_length=12.0):
    """Build SPC/E waters with assigned subsystems.

    Args:
        placements: Pairs of an origin for the oxygen and the subsystem
            the whole molecule belongs to.
        box_length: The edge length (Angstrom) of the cubic box.

    Returns:
        The system, with SPC/E charges and masses assigned.
    """
    from pydft_qmmm import Atom
    from pydft_qmmm.utils import ELEMENT_TO_MASS
    atoms = []
    for residue, (origin, subsystem) in enumerate(placements):
        # Match atom names in tests/data/spce_residues.xml.
        for element, name, offset in zip(
                ("O", "H", "H"), ("O", "H1", "H2"), WATER_GEOMETRY,
        ):
            atoms.append(
                Atom(
                    position=np.asarray(origin, dtype=float) + offset,
                    element=element,
                    name=name,
                    residue=residue,
                    residue_name="HOH",
                    chain="A",
                    charge=SPCE_CHARGES[element],
                    mass=ELEMENT_TO_MASS[element],
                    subsystem=subsystem,
                ),
            )
    system = System(atoms, np.eye(3) * box_length)
    assert np.count_nonzero(system.charges) == len(system.charges)
    return system


@pytest.fixture
def pyscf_water_system():
    """A single QM water in a cubic box."""
    return build_water_system([((6.0, 6.0, 6.0), Subsystem.I)])


@pytest.fixture
def pyscf_triplet_system():
    """Molecular oxygen, whose ground state is a triplet."""
    from pydft_qmmm import Atom
    from pydft_qmmm.utils import ELEMENT_TO_MASS
    atoms = [
        Atom(
            position=np.array([6.0, 6.0, 6.0 + z]),
            element="O",
            name="O",
            residue=0,
            residue_name="OXY",
            charge=0.0,
            mass=ELEMENT_TO_MASS["O"],
            subsystem=Subsystem.I,
        )
        for z in (-0.604, 0.604)
    ]
    return System(atoms, np.eye(3) * 12.0)


@pytest.fixture
def pyscf_embedded_water():
    """A QM water and an equivalent bare PySCF embedding."""
    from pyscf import dft, gto, qmmm
    from pydft_qmmm.interfaces.pyscf.pyscf_factory import (
        pyscf_interface_factory,
    )
    system = build_water_system([
        ((6.0, 6.0, 6.0), Subsystem.I),
        ((6.4, 7.1, 8.7), Subsystem.II),
    ])
    potential = pyscf_interface_factory(
        system,
        basis="sto-3g",
        functional="PBE",
        charge=0,
        multiplicity=1,
        conv_tol=1e-11,
    )
    qm_indices = sorted(system.select("subsystem I"))
    mm_indices = sorted(system.select("subsystem II"))
    mol = gto.M(
        atom=[
            (str(system.elements[i]), tuple(system.positions[i]))
            for i in qm_indices
        ],
        unit="Angstrom",
        basis="sto-3g",
        charge=0,
        spin=0,
        verbose=0,
    )
    method = dft.RKS(mol, xc="PBE")
    method.conv_tol = 1e-11
    method.grids.level = 3
    method = qmmm.add_mm_charges(
        method,
        np.asarray(system.positions)[mm_indices],
        np.asarray(system.charges)[mm_indices],
        unit="Angstrom",
    )
    return potential, method, qm_indices, mm_indices


@pytest.fixture
def pyscf_pme_system():
    """Waters spanning all three subsystems in a small box."""
    system = build_water_system(
        [
            ((4.0, 4.0, 4.0), Subsystem.I),
            ((4.5, 5.2, 6.3), Subsystem.II),
            ((7.4, 2.1, 8.0), Subsystem.III),
            ((1.6, 7.7, 2.4), Subsystem.III),
        ],
        box_length=10.0,
    )
    assert len(system.select("subsystem I")) > 0
    assert len(system.select("subsystem II")) > 0
    assert len(system.select("subsystem III")) > 0
    return system


@pytest.fixture
def pyscf_iodide_system():
    """An iodide ion for effective-core-potential tests."""
    from pydft_qmmm import Atom
    from pydft_qmmm.utils import ELEMENT_TO_MASS
    atoms = [
        Atom(
            position=np.array([6.0, 6.0, 6.0]),
            element="I",
            name="I",
            residue=0,
            residue_name="IOD",
            charge=-1.0,
            mass=ELEMENT_TO_MASS["I"],
            subsystem=Subsystem.I,
        ),
    ]
    return System(atoms, np.eye(3) * 12.0)
