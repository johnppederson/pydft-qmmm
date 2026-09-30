from __future__ import annotations

import json
import os
import pathlib

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


# @pytest.fixture
# def spce_dimer_system():
#    return System.load(
#        "tests/data/hoh_dimer.pdb",
#    )


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


VASP_PP_LIBRARY = os.environ.get(
    "VASP_PP_PATH",
    "/storage/project/r-jmcdaniel43-0/cshao48/install/vasp/potpaw_PBE.64",
)


@pytest.fixture
def vasp_pp_library():
    """The POTCAR library used by the VASP interface tests."""
    return VASP_PP_LIBRARY


@pytest.fixture
def vasp_workdir(tmp_path, request):
    """Where VASP-backed tests should put their run directories.

    Defaults to pytest's tmp_path, which keeps the unit tests hermetic.
    On a compute node tmp_path lives in node-local /tmp and evaporates
    when the job ends, taking the OUTCAR with it -- so setting
    PYDFT_QMMM_VASP_TESTDIR redirects runs somewhere durable and makes
    failures diagnosable after the fact.
    """
    root = os.environ.get("PYDFT_QMMM_VASP_TESTDIR")
    if not root:
        return tmp_path
    path = os.path.join(root, request.node.name)
    os.makedirs(path, exist_ok=True)
    return pathlib.Path(path)


@pytest.fixture
def vasp_qmmm_system(spce_system):
    """An SPC/E system with an explicit QM region.

    The spce_qmmm_system fixture only assigns subsystem II; subsystem I
    is normally set when a Hamiltonian builds its calculator.  The VASP
    interface tests exercise the interface directly, so they need the QM
    region assigned up front.
    """
    with open("tests/data/spce_qmmm_region_ii.json") as fh:
        embedding_list = json.load(fh)
    for atom in embedding_list:
        spce_system.subsystems[atom] = Subsystem.II
    # One water molecule as the QM subsystem.
    for atom in range(3):
        spce_system.subsystems[atom] = Subsystem.I
    # System.load() leaves every charge at zero -- charges are normally
    # assigned when an MM Hamiltonian builds its calculator.  Without
    # this the embedding tests build an IDENTICALLY ZERO V_ext and pass
    # while testing no physics whatsoever.  Values are the SPC/E
    # charges from tests/data/spce_no_lj.xml.
    charges = np.asarray(spce_system.charges)
    for atom, element in enumerate(spce_system.elements):
        charges[atom] = -0.8476 if str(element) == "O" else 0.4238
    assert np.count_nonzero(charges) == len(charges), "charges not applied"
    return spce_system


@pytest.fixture
def vasp_three_subsystem_system(vasp_qmmm_system):
    """vasp_qmmm_system with a genuinely populated subsystem III.

    The stock fixture assigns 3 atoms to subsystem I and 1149 to
    subsystem II, leaving subsystem III EMPTY.  That makes any assertion
    of the form "VASP contributes nothing to subsystem III" vacuously
    true, since it ranges over a zero-length array.  Demote the outer
    half of subsystem II so such assertions have atoms to range over.
    """
    near = sorted(vasp_qmmm_system.select("subsystem II"))
    for atom in near[len(near) // 2:]:
        vasp_qmmm_system.subsystems[atom] = Subsystem.III
    assert len(vasp_qmmm_system.select("subsystem III")) > 0
    assert len(vasp_qmmm_system.select("subsystem II")) > 0
    return vasp_qmmm_system


@pytest.fixture
def vasp_embedded(vasp_qmmm_system, vasp_workdir):
    """A VASP potential with electrostatic embedding switched on."""
    from pydft_qmmm.interfaces.vasp.vasp_factory import vasp_interface_factory
    return vasp_interface_factory(
        vasp_qmmm_system,
        directory=str(vasp_workdir / "vasp"),
        pp_path=VASP_PP_LIBRARY,
        embedding=True,
        # The QM region is one water, but the VASP cell must match the
        # MM box (29.9 A) so the embedded charges land in the right
        # place -- it cannot be shrunk.  At the default ENCUT=400 that
        # is a ~196**3 fine grid, which OOM-killed a 12 GB job.  These
        # tests ask "does the plugin fire and is E consistent with F",
        # not "is the energy converged", so a lighter cutoff is right.
        incar={"ENCUT": 250},
    )


@pytest.fixture
def h_constant_field_system():
    """H atom in a constant field.

    One hydrogen at the centre of a 10 x 10 x 30 cell (subsystem I) and
    two oppositely charged sheets at z = 20 and z = 0 (subsystem II).

    The atom is charge neutral and barely polarizable, so the TOTAL
    force must be ~0.  VASP alone reports 1e * E_ext, because it omits
    the core-field interaction; Eq. 4 supplies exactly that.  Unlike a
    QM/MM system, the correction here IS the entire signal.
    """
    system = System.load("tests/data/h_constant_field.pdb")
    SURFACE_CHARGE = 0.001                      # e / Angstrom**2
    sheet = [i for i, e in enumerate(system.elements) if str(e) == "Ne"]
    per_sheet = len(sheet) // 2
    per_site = SURFACE_CHARGE * 100.0 / per_sheet
    charges = np.asarray(system.charges)
    for n, atom in enumerate(sheet):
        charges[atom] = per_site if n < per_sheet else -per_site
    system.subsystems[0] = Subsystem.I
    for atom in sheet:
        system.subsystems[atom] = Subsystem.II
    assert abs(charges.sum()) < 1e-9, "sheets must be net neutral"
    return system


@pytest.fixture
def sparc_workdir(tmp_path, request):
    """A scratch directory for SPARC runs.

    Defaults to pytest's tmp_path, which keeps the unit tests hermetic.
    Under srun, ranks land on compute nodes whose node-local /tmp does
    not contain a login/build node's tmp_path, and every rank fails to
    chdir into it, so SPARC dies immediately.  Passing
    --basetemp=<shared storage path> on the pytest command line fixes
    this for the whole run; PYDFT_QMMM_SPARC_TESTDIR is a second,
    fixture-level guarantee for anyone who forgets that flag.
    """
    root = os.environ.get("PYDFT_QMMM_SPARC_TESTDIR")
    if root:
        directory = pathlib.Path(root) / request.node.name / "sparc_workdir"
        directory.mkdir(parents=True, exist_ok=True)
    else:
        directory = tmp_path / "sparc_workdir"
        directory.mkdir()
    return str(directory)


@pytest.fixture
def sparc_plain(spce_qmmm_system, sparc_workdir):
    """A SPARC potential with embedding switched off."""
    from pydft_qmmm.interfaces.sparc.sparc_factory import (
        sparc_interface_factory,
    )
    return sparc_interface_factory(
        spce_qmmm_system,
        directory=sparc_workdir,
        xc="pbe",
        fd_grid=(48, 48, 48),
    )


@pytest.fixture
def sparc_embedded(spce_qmmm_system, sparc_workdir):
    """A SPARC potential with embedding switched on."""
    from pydft_qmmm.interfaces.sparc.sparc_factory import (
        sparc_interface_factory,
    )
    return sparc_interface_factory(
        spce_qmmm_system,
        directory=sparc_workdir,
        xc="pbe",
        fd_grid=(48, 48, 48),
        embedding=True,
    )


@pytest.fixture
def mm_spce_water4():
    """MM Hamiltonian sized for the 12 A water4 box.

    mm_spce_no_lj relies on the defaults in openmm_factory.py
    (nonbonded_method="PME", nonbonded_cutoff=14 A), which assume the
    29.9 A spce_qmmm_system.  OpenMM refuses a cutoff greater than half
    the box size, so on the 12 A water4 box that raises immediately;
    5.0 A leaves headroom under the 6 A half-box limit.  Leaving
    nonbonded_method at its "PME" default (rather than the
    "CutoffPeriodic" the sparc_electrostatic example uses) matters here:
    sparc_pme_total's QM/MM/PME coupling reads the MM interface's own
    PME alpha/gridnumber via get_pme_parameters(), which requires
    exactly one PME NonbondedForce to be present.
    """
    return MMHamiltonian(
        forcefield=[
            "tests/data/spce_no_lj.xml",
            "tests/data/spce_residues.xml",
        ],
        nonbonded_cutoff=5.0,
        pme_gridnumber=30,
        pme_alpha=5.0,
    )


@pytest.fixture
def water4_system():
    """A 4-water cluster in a 12 A cell.

    Small enough (12 atoms, 12 A box) for the slow SPARC embedding
    tests to finish a real SCF cycle in scheduler time, unlike the
    1152-atom 29.9 A spce_qmmm_system, which never got past one SCF
    iteration on a 48**3 grid before hitting a wall-clock limit.
    """
    return System.load("tests/data/water4.pdb")


@pytest.fixture
def sparc_qmmm_system(water4_system):
    """water4_system with the first water assigned to subsystem I.

    Subsystem II/III membership for the rest is deliberately left
    unset here: MMHamiltonian.build_calculator stamps every MM atom to
    Subsystem.III when the calculator is built, and only a *live*
    partition plugin -- not partition=None -- promotes any of them
    back to II at calculate() time.  Assigning II by hand in this
    fixture would just be silently overwritten the moment the
    calculator is built, which is exactly the bug this fixture used to
    hide (see F1 in the review).
    """
    system = water4_system
    for i in range(3):
        system.subsystems[i] = Subsystem.I
    return system


@pytest.fixture
def sparc_qm_embedded(sparc_workdir):
    """A SPARC QM Hamiltonian with embedding on.

    The factory fixtures above return a potential, which is what the
    configuration tests poke at.  Composing a QM/MM calculator needs a
    Hamiltonian instead.
    """
    from pydft_qmmm import QMHamiltonian
    fd_points = int(os.environ.get("PYDFT_QMMM_SPARC_FD_POINTS", "151"))
    return QMHamiltonian(
        interface="sparc",
        charge=0,
        xc="pbe",
        # 12 A cell / 151 points = 0.15 Bohr mesh spacing. embedding_sigma
        # (0.3 A) must comfortably exceed that or Gaussian spreading
        # aliases onto the grid -- see QMMM.md's FD_GRID guidance.  A
        # coarser 48**3 grid (0.25 A spacing here) was measured to
        # carry an unembedded net force of ~190 kJ/mol/A on this exact
        # 4-water geometry, which would swamp the physics assertions
        # below. PYDFT_QMMM_SPARC_FD_POINTS can override this for mesh
        # convergence runs.
        fd_grid=(fd_points, fd_points, fd_points),
        embedding_sigma=0.3,
        tol_scf=1e-6,
        directory=sparc_workdir,
        embedding=True,
    )


@pytest.fixture
def sparc_embedded_total(sparc_qmmm_system, sparc_qm_embedded, mm_spce_water4):
    """SPARC QM + SPCE MM, electrostatic close range, MM long range.

    long_range="mechanical" puts I-III at TheoryLevel.MM, which is
    SYMMETRIC with III-I (see _LONG_EMBEDDING).  "cutoff" is (NO, MM):
    subsystem III feels the QM atoms but the QM atoms are masked from
    feeling III (openmm_interface.zero_forces), so the total energy is
    not the potential of the total force and a full-energy finite
    difference cannot match the analytic force.  Upstream works around
    that by restricting numerical_gradient to components=["Psi4"];
    using a conservative scheme instead lets the gradient tests below
    check the whole energy, which is the stronger statement.

    Deliberately shares sparc_pme_total's 2.5 A partition cutoff so
    that the two fixtures differ in exactly one variable: the
    long-range treatment ("cutoff" here, "electrostatic" there).  The
    default CentroidPartition("all", 14.) would put all nine MM atoms
    in subsystem II and leave III empty, and then
    test_pme_changes_the_energy would be comparing a 9-Gaussian V_ext
    against a 3-Gaussian V_ext -- it would pass even if the PME
    reciprocal sum returned identically zero, which is no evidence
    about PME at all.

    Both subsystem assertions are checks, not assumptions: an empty II
    means V_ext is identically zero and the physics tests downstream
    verify nothing (F1); an empty III means the long-range term has
    nothing to act on.
    """
    from pydft_qmmm import QMMMHamiltonian
    qmmm = QMMMHamiltonian("electrostatic", "mechanical", cutoff=2.5)
    total = mm_spce_water4[3:] + sparc_qm_embedded[0:3] + qmmm
    total.build_calculator(sparc_qmmm_system)
    qmmm.partition.generate_partition()
    assert len(sparc_qmmm_system.select("subsystem II")) > 0, (
        "partition left subsystem II empty -- V_ext would be "
        "identically zero and the physics tests would verify nothing"
    )
    assert len(sparc_qmmm_system.select("subsystem III")) > 0, (
        "partition left subsystem III empty -- the long-range term "
        "would have no atoms to act on, and the PME comparison "
        "against this fixture would be vacuous"
    )
    return total


@pytest.fixture
def sparc_newton_total(sparc_qmmm_system, sparc_qm_embedded, mm_spce_water4):
    """Balanced electrostatic I-II coupling with no subsystem III.

    This fixture isolates the QM/MM action-reaction pair.  The cutoff
    fixture below deliberately leaves a subsystem III, whose ordinary
    II-III MM forces make a sum over only I and II non-conservative.
    """
    from pydft_qmmm import QMMMHamiltonian
    qmmm = QMMMHamiltonian("electrostatic", "none", cutoff=14.0)
    total = mm_spce_water4[3:] + sparc_qm_embedded[0:3] + qmmm
    total.build_calculator(sparc_qmmm_system)
    qmmm.partition.generate_partition()
    assert len(sparc_qmmm_system.select("subsystem II")) > 0
    assert len(sparc_qmmm_system.select("subsystem III")) == 0
    return total


@pytest.fixture
def sparc_cutoff_total(sparc_qmmm_system, sparc_qm_embedded, mm_spce_water4):
    """The matched control for the PME comparisons.

    sparc_embedded_total uses long_range="mechanical" so that the
    gradient tests can difference the whole energy.  That is the wrong
    control for "does PME change anything", because "mechanical" leaves
    base_force_mask[QM] == 1 while "electrostatic" zeroes it -- the two
    would differ in the QM forces even if the PME reciprocal sum
    returned zero.  "cutoff" shares "electrostatic"'s mask (both are
    asymmetric, both zero it), so the ONLY difference against
    sparc_pme_total is whether subsystem III reaches the QM engine.
    """
    from pydft_qmmm import QMMMHamiltonian
    qmmm = QMMMHamiltonian("electrostatic", "cutoff", cutoff=2.5)
    total = mm_spce_water4[3:] + sparc_qm_embedded[0:3] + qmmm
    total.build_calculator(sparc_qmmm_system)
    qmmm.partition.generate_partition()
    assert len(sparc_qmmm_system.select("subsystem II")) > 0
    assert len(sparc_qmmm_system.select("subsystem III")) > 0
    return total


@pytest.fixture
def sparc_pme_total(sparc_qmmm_system, sparc_qm_embedded, mm_spce_water4):
    """SPARC QM + SPCE MM under full QM/MM/PME coupling.

    A 2.5 A partition cutoff splits the three MM waters: the nearest
    (~2.3 A from the QM centroid) lands in subsystem II and is handled
    by the near-field Gaussian embedding; the other two (~2.7 A) land
    in subsystem III, which is what the PME reciprocal sum in
    _write_pme_data actually sums over (it excludes "not subsystem
    III").  Without this split subsystem III would be empty and the
    PME test would compare an energy against itself.
    """
    from pydft_qmmm import QMMMHamiltonian
    qmmm = QMMMHamiltonian(
        "electrostatic", "electrostatic",
        cutoff=2.5,
    )
    total = mm_spce_water4[3:] + sparc_qm_embedded[0:3] + qmmm
    total.build_calculator(sparc_qmmm_system)
    qmmm.partition.generate_partition()
    assert len(sparc_qmmm_system.select("subsystem II")) > 0, (
        "partition left subsystem II empty -- V_ext would be "
        "identically zero and the physics tests would verify nothing"
    )
    assert len(sparc_qmmm_system.select("subsystem III")) > 0, (
        "partition left subsystem III empty -- PME would sum over no "
        "atoms and the PME-vs-plain energy comparison would be vacuous"
    )
    return total


@pytest.fixture
def vasp_pme_system(vasp_qmmm_system):
    """QM/MM partition with a real subsystem III, for the PME path.

    vasp_qmmm_system leaves every atom in I or II, so "not subsystem
    III" covers the whole system and the PME reciprocal sum is almost
    entirely cancelled by its own real-space adjustment.  Assigning the
    remainder to III gives PME something to do: I is the QM region, II
    is handled by the real-space cutoff, and III is what PME sums.
    """
    assigned = set(vasp_qmmm_system.select("subsystem I")) | set(
        vasp_qmmm_system.select("subsystem II"),
    )
    for atom in range(len(vasp_qmmm_system.elements)):
        if atom not in assigned:
            vasp_qmmm_system.subsystems[atom] = Subsystem.III
    assert len(vasp_qmmm_system.select("subsystem III")) > 0
    return vasp_qmmm_system


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
def pyscf_pbc_system():
    """A QM water in an 8 Å periodic box."""
    return build_water_system(
        [((4.0, 4.0, 4.0), Subsystem.I)], box_length=8.0,
    )


def build_pbc_interface(system, device="cpu", **overrides):
    """Build periodic PySCF with optional device and factory overrides."""
    from pydft_qmmm.interfaces.pyscf_pbc.pbc_factory import (
        pyscf_pbc_interface_factory,
    )
    # This cutoff resolves forces; energy convergence alone is insufficient.
    options = dict(
        basis="gth-dzvp",
        pseudo="gth-pbe",
        functional="pbe",
        charge=0,
        multiplicity=1,
        ke_cutoff=200.0,
        device=device,
        conv_tol=1e-10,
    )
    options.update(overrides)
    return pyscf_pbc_interface_factory(system, **options)


def build_pbc_embedded_system():
    """QM and region II waters in an 8 Å box."""
    return build_water_system(
        [
            ((4.0, 4.0, 4.0), Subsystem.I),
            ((4.3, 5.9, 6.1), Subsystem.II),
        ],
        box_length=8.0,
    )


@pytest.fixture
def pyscf_pbc_embedded_factory():
    """Build the same embedded water system on either device."""
    def build(device="cpu"):
        interface = build_pbc_interface(
            build_pbc_embedded_system(), device=device,
        )
        interface.configure_electrostatic_embedding(True)
        return interface
    return build


@pytest.fixture
def pyscf_pbc_pme_interface():
    """Build an embedded periodic system with region III and live PME."""
    from pydft_qmmm.potentials.pme_potential import PMEElectronicPotential
    system = build_water_system(
        [
            ((4.0, 4.0, 4.0), Subsystem.I),
            ((4.3, 5.9, 6.1), Subsystem.II),
            ((9.2, 9.6, 9.1), Subsystem.III),
            ((2.1, 8.2, 3.4), Subsystem.III),
        ],
        box_length=12.0,
    )
    interface = build_pbc_interface(system)
    interface.configure_electrostatic_embedding(True)
    interface.add_electronic_potential(
        PMEElectronicPotential(system, 5.0, (24, 24, 24), 6),
    )
    return interface


@pytest.fixture
def pyscf_pbc_embedded_interface(pyscf_pbc_embedded_factory):
    """Return an embedded interface and its separately computed bare energy."""
    interface = pyscf_pbc_embedded_factory("cpu")
    bare = build_pbc_interface(interface.system).compute_energy()
    return interface, bare


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
def pyscf_pme_adapter(pyscf_pme_system):
    """A PySCF potential carrying a reciprocal PME electronic potential."""
    from pydft_qmmm.potentials.pme_potential import PMEElectronicPotential
    from pydft_qmmm.interfaces.pyscf.pyscf_factory import (
        pyscf_interface_factory,
    )
    potential = pyscf_interface_factory(
        pyscf_pme_system,
        basis="sto-3g",
        functional="PBE",
        charge=0,
        multiplicity=1,
        conv_tol=1e-11,
    )
    potential.add_electronic_potential(
        PMEElectronicPotential(pyscf_pme_system, 0.4, (20, 20, 20), 6),
    )
    return potential


@pytest.fixture
def pyscf_openmm_system():
    """Four SPC/E waters spanning all subsystems in a cubic box."""
    system = build_water_system(
        [
            ((12.0, 12.0, 12.0), Subsystem.I),
            ((12.4, 13.1, 14.7), Subsystem.II),
            ((4.0, 19.0, 5.0), Subsystem.III),
            ((19.5, 4.5, 19.5), Subsystem.III),
        ],
        box_length=24.0,
    )
    return system


@pytest.fixture
def mm_pyscf_spce():
    return MMHamiltonian(
        forcefield=[
            "tests/data/spce_no_lj.xml",
            "tests/data/spce_residues.xml",
        ],
        nonbonded_cutoff=9.0,
        pme_gridnumber=48,
        pme_alpha=3.5,
    )


@pytest.fixture
def qm_pyscf_water():
    return QMHamiltonian(
        interface="pyscf-mol",
        basis="sto-3g",
        functional="PBE",
        charge=0,
        multiplicity=1,
        conv_tol=1e-10,
        # Level 5 limits moving-grid force error to about
        # 0.003 kJ/mol/Angstrom for this fixture.
        grid_level=5,
    )


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
