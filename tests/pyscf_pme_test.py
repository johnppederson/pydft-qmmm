"""Tests for analytic PME source forces and the PySCF PME operator."""
from __future__ import annotations

import numpy as np
import pytest

from pydft_qmmm.potentials.pme_potential import PMEElectronicPotential
from pydft_qmmm.utils import BOHR_PER_ANGSTROM
from pydft_qmmm.utils import KJMOL_PER_EH

# helPME works in Angstrom, so alpha is in inverse Angstrom -- the
# OpenMM interface divides its nm^-1 value by ten before handing it
# over.  A 10 A box with 0.5 A grid spacing is the coarsest setting
# QMMMHamiltonian considers advisable.
PME_ALPHA = 0.4
PME_GRIDNUMBER = (20, 20, 20)
PME_SPLINE_ORDER = 6

# Two arbitrary points inside the fixture's 10 A box, carrying the sort
# of small negative charges an electron-density quadrature produces.
SOURCE_POINTS = np.array([[4.1, 5.2, 6.3], [7.0, 3.0, 2.0]])
SOURCE_WEIGHTS = np.array([-0.7, -0.3])


def make_potential(system):
    return PMEElectronicPotential(
        system,
        PME_ALPHA,
        PME_GRIDNUMBER,
        PME_SPLINE_ORDER,
    )


def source_energy(potential, points, weights):
    r"""The energy (kJ/mol) that ``compute_source_forces`` differentiates.

    ``compute_potential`` returns the potential energy of one electron
    in Hartree, so recovering the electrostatic potential in
    :math:`\mathrm{kJ\;mol^{-1}\;e^{-1}}` takes a sign flip and a unit
    conversion.  The source energy is then that potential contracted
    with the source charges.
    """
    v = np.asarray(potential.compute_potential(points)).reshape(-1)
    return float(weights @ (-KJMOL_PER_EH * v))


def finite_difference_source_energy(
        potential, points, weights, atom, step=0.001,
):
    """Central-difference forces on one atom, in kJ/mol/Angstrom."""
    gradient = np.zeros(3)
    for axis in range(3):
        potential.system.positions[atom, axis] += step
        plus = source_energy(potential, points, weights)
        potential.system.positions[atom, axis] -= 2 * step
        minus = source_energy(potential, points, weights)
        potential.system.positions[atom, axis] += step
        gradient[axis] = (plus - minus) / (2 * step)
    return -gradient


def test_base_electronic_potential_rejects_source_forces():
    from pydft_qmmm.potentials import ElectronicPotential

    class Constant(ElectronicPotential):
        def compute_potential(self, coordinates):
            return np.zeros((len(coordinates), 1))

    with pytest.raises(NotImplementedError, match="source forces"):
        Constant().compute_source_forces(SOURCE_POINTS, SOURCE_WEIGHTS)


@pytest.mark.parametrize("atom", [6, 0, 3])
def test_pme_source_forces_match_finite_difference(pyscf_pme_system, atom):
    potential = make_potential(pyscf_pme_system)
    analytical = potential.compute_source_forces(
        SOURCE_POINTS, SOURCE_WEIGHTS,
    )
    numerical = finite_difference_source_energy(
        potential, SOURCE_POINTS, SOURCE_WEIGHTS, atom,
    )
    assert analytical[atom] == pytest.approx(numerical, abs=1e-4)


def test_pme_source_forces_are_not_trivially_zero(pyscf_pme_system):
    potential = make_potential(pyscf_pme_system)
    forces = potential.compute_source_forces(SOURCE_POINTS, SOURCE_WEIGHTS)
    assert np.abs(forces).max() > 1e-3


def test_pme_reciprocity_of_the_reciprocal_operator(pyscf_pme_system):
    """Swapping sources and targets must not change the pair energy."""
    potential = make_potential(pyscf_pme_system)
    # Energy of the source charges in the field of the system charges.
    forward = source_energy(potential, SOURCE_POINTS, SOURCE_WEIGHTS)
    # The same energy, evaluated with the system charges as targets of
    # the field the source charges make.  This is the identity that
    # makes the adjoint force contraction legitimate.
    reverse = potential.compute_source_potential(
        SOURCE_POINTS, SOURCE_WEIGHTS,
    )
    charges = np.asarray(pyscf_pme_system.charges)
    assert float(charges @ reverse) == pytest.approx(forward, rel=1e-9)


def test_pme_source_forces_are_translationally_invariant(pyscf_pme_system):
    """Sliding everything through the box must not change the forces.

    Invariance is exact for the Ewald sum but only approximate for its
    B-spline mesh representation, so the residual is bounded at the
    working mesh and required to shrink when the mesh is refined.
    """
    shift = np.array([0.37, -0.21, 0.64])
    residuals = []
    for gridnumber, order in ((PME_GRIDNUMBER, PME_SPLINE_ORDER),
                              ((40, 40, 40), 8)):
        potential = PMEElectronicPotential(
            pyscf_pme_system, PME_ALPHA, gridnumber, order,
        )
        before = potential.compute_source_forces(
            SOURCE_POINTS, SOURCE_WEIGHTS,
        )
        potential.system.positions[:] = (
            np.asarray(potential.system.positions) + shift
        )
        after = potential.compute_source_forces(
            SOURCE_POINTS + shift, SOURCE_WEIGHTS,
        )
        potential.system.positions[:] = (
            np.asarray(potential.system.positions) - shift
        )
        residuals.append(np.abs(after - before).max())
    assert np.abs(before).max() > 1.0
    assert residuals[0] == pytest.approx(0, abs=2e-4)
    assert residuals[1] < residuals[0] / 100


# ---------------------------------------------------------------------
# Fixed-mesh PySCF PME operator
# ---------------------------------------------------------------------


def spin_sum(dm):
    dm = np.asarray(dm)
    return dm.sum(axis=0) if dm.ndim == 3 else dm


def test_pme_ao_operator_is_symmetric(pyscf_pme_adapter):
    matrix = pyscf_pme_adapter.pme_matrix
    assert matrix == pytest.approx(matrix.T, abs=1e-12)


def test_pme_energy_is_density_contraction(pyscf_pme_adapter):
    energy = pyscf_pme_adapter.pme_energy
    dm = pyscf_pme_adapter.method[0].make_rdm1()
    expected = np.einsum("ij,ji->", spin_sum(dm), pyscf_pme_adapter.pme_matrix)
    assert energy == pytest.approx(expected, abs=1e-10)


def test_pme_operator_shifts_the_scf_energy(pyscf_pme_system,
                                            pyscf_pme_adapter):
    from pydft_qmmm.interfaces.pyscf.pyscf_factory import (
        pyscf_interface_factory,
    )
    bare = pyscf_interface_factory(
        pyscf_pme_system,
        basis="sto-3g",
        functional="PBE",
        charge=0,
        multiplicity=1,
        conv_tol=1e-11,
    )
    # Without the reciprocal operator the QM energy must differ, or the
    # operator is not reaching the Hamiltonian at all.
    assert pyscf_pme_adapter.compute_energy() != pytest.approx(
        bare.compute_energy(), abs=1e-6,
    )
    assert abs(pyscf_pme_adapter.pme_energy) > 1e-8


def _displaced_energy(potential, atom, axis, step):
    """The adapter's own energy with one atom displaced, in kJ/mol."""
    potential.system.positions[atom, axis] += step
    energy = potential.compute_energy()
    potential.system.positions[atom, axis] -= step
    return energy


def _force_residuals(potential, atom, step=0.002):
    """Analytic minus central-difference force, in kJ/mol/Angstrom."""
    forces = potential.compute_forces()
    residuals = np.zeros(3)
    for axis in range(3):
        plus = _displaced_energy(potential, atom, axis, step)
        minus = _displaced_energy(potential, atom, axis, -step)
        residuals[axis] = forces[atom, axis] + (plus - minus) / (2 * step)
    return residuals


# The quadrature grid moves with the QM atoms, and the terms that
# generates are omitted here for the same reason PySCF omits them from
# the exchange-correlation gradient by default: they are artifacts of a
# finite grid, not part of the exact derivative.  What is left is a
# residual that shrinks as the grid is refined, measured at 0.07
# kJ/mol/Angstrom for grid_level 3 and 0.002 for grid_level 5 on this
# fixture.  MM sites carry no grid, so their forces are exact.
QM_FORCE_TOLERANCE = 0.1
MM_FORCE_TOLERANCE = 1e-3


@pytest.mark.parametrize("atom", [0, 1])
def test_pme_qm_force_matches_central_differences(pyscf_pme_adapter, atom):
    """The QM-atom force differentiates the SCF's own embedding energy."""
    residuals = _force_residuals(pyscf_pme_adapter, atom)
    assert residuals == pytest.approx(0, abs=QM_FORCE_TOLERANCE)


def test_pme_qm_force_residual_shrinks_with_the_grid(pyscf_pme_system):
    """Refining the quadrature must tighten the force consistency."""
    from pydft_qmmm.potentials.pme_potential import PMEElectronicPotential
    from pydft_qmmm.interfaces.pyscf.pyscf_factory import (
        pyscf_interface_factory,
    )
    residuals = []
    for grid_level in (3, 5):
        potential = pyscf_interface_factory(
            pyscf_pme_system,
            basis="sto-3g",
            functional="PBE",
            charge=0,
            multiplicity=1,
            conv_tol=1e-11,
            grid_level=grid_level,
        )
        potential.add_electronic_potential(
            PMEElectronicPotential(
                pyscf_pme_system, PME_ALPHA, PME_GRIDNUMBER,
                PME_SPLINE_ORDER,
            ),
        )
        residuals.append(np.abs(_force_residuals(potential, 0)).max())
    assert residuals[0] < QM_FORCE_TOLERANCE
    assert residuals[1] < residuals[0] / 5


@pytest.mark.parametrize("atom", [3, 6])
def test_pme_mm_force_matches_central_differences(pyscf_pme_adapter, atom):
    """Subsystem II and III sites feel the electron density in return."""
    residuals = _force_residuals(pyscf_pme_adapter, atom)
    assert residuals == pytest.approx(0, abs=MM_FORCE_TOLERANCE)


def test_pme_mm_force_is_not_trivially_zero(pyscf_pme_adapter):
    forces = pyscf_pme_adapter.compute_forces()
    subsystem_iii = sorted(
        pyscf_pme_adapter.system.select("subsystem III"),
    )
    assert np.abs(forces[subsystem_iii]).max() > 1.0


def test_pme_force_survives_a_periodic_boundary(pyscf_pme_adapter):
    """A subsystem III site straddling the boundary is still consistent."""
    pyscf_pme_adapter.system.positions[6] = np.array([9.98, 0.02, 5.0])
    residuals = _force_residuals(pyscf_pme_adapter, 6)
    assert residuals == pytest.approx(0, abs=MM_FORCE_TOLERANCE)


def test_pme_energy_is_physically_sized(pyscf_pme_adapter):
    """The reciprocal embedding energy must not be a quadrature artifact.

    A uniform mesh at PME resolution integrates this fixture's density
    to 170 electrons instead of 10, inflating this term by more than an
    order of magnitude.  The molecular quadrature integrates it to
    about a part in a million, so the term stays the few-hundred
    kJ/mol interaction it physically is.
    """
    from pyscf.dft import numint
    state = pyscf_pme_adapter._run()
    ao = numint.eval_ao(state.mol, state.quadrature.coordinates
                        * BOHR_PER_ANGSTROM)
    density = numint.eval_rho(state.mol, ao, spin_sum(state.dm))
    electrons = float(density @ state.quadrature.weights)
    assert electrons == pytest.approx(state.mol.nelectron, abs=1e-4)
    assert abs(pyscf_pme_adapter.pme_energy * KJMOL_PER_EH) < 1000.0


# ---------------------------------------------------------------------
# Nuclear charges under an effective core potential
# ---------------------------------------------------------------------


def make_nuclear_potential(system, qm_interface=None):
    from pydft_qmmm.potentials.pme_potential import PMENuclearPotential
    return PMENuclearPotential(
        system, PME_ALPHA, PME_GRIDNUMBER, PME_SPLINE_ORDER, qm_interface,
    )


def test_pme_nuclear_charge_defaults_to_the_atomic_number(pyscf_pme_system):
    """An interface that says nothing keeps the all-electron behaviour."""
    potential = make_nuclear_potential(pyscf_pme_system)
    assert potential.source_charges().tolist() == [8.0, 1.0, 1.0]


def test_all_electron_interface_reports_atomic_numbers(pyscf_pme_system):
    from pydft_qmmm.interfaces.pyscf.pyscf_factory import (
        pyscf_interface_factory,
    )
    qm = pyscf_interface_factory(
        pyscf_pme_system, basis="sto-3g", functional="PBE",
        charge=0, multiplicity=1,
    )
    potential = make_nuclear_potential(pyscf_pme_system, qm)
    assert potential.source_charges().tolist() == [8.0, 1.0, 1.0]


def test_ecp_reduces_the_pme_nuclear_charge(pyscf_iodide_system):
    """The PME field must see the charge the wavefunction leaves behind.

    Iodine's def2-ECP replaces 28 core electrons, so the solver carries
    26 electrons against a core charge of 25 -- a net -1, as iodide
    should be.  Taking the atomic number instead would put +53 against
    those 26 electrons and enter the reciprocal sum as +27.
    """
    from pydft_qmmm.interfaces.pyscf.pyscf_factory import (
        pyscf_interface_factory,
    )
    qm = pyscf_interface_factory(
        pyscf_iodide_system, basis="def2-svp", ecp="def2-svp",
        functional="PBE", charge=-1, multiplicity=1,
    )
    charges = make_nuclear_potential(pyscf_iodide_system, qm).source_charges()
    assert charges.tolist() == [25.0]
    electrons = qm._build_molecule()[1].nelectron
    assert charges.sum() - electrons == pytest.approx(-1.0)
    # The uncorrected answer is the bug this guards against.
    bare = make_nuclear_potential(pyscf_iodide_system).source_charges()
    assert bare.tolist() == [53.0]


def test_psi4_reports_the_same_ecp_nuclear_charge(pyscf_iodide_system):
    """Both engines must agree on what the PME field sees."""
    psi4 = pytest.importorskip("psi4")
    from pydft_qmmm.interfaces.psi4.psi4_factory import psi4_interface_factory
    psi4.core.be_quiet()
    qm = psi4_interface_factory(
        pyscf_iodide_system, basis="def2-SVP", functional="PBE",
        charge=-1, multiplicity=1, output_file="/dev/null",
    )
    assert qm.nuclear_charges().tolist() == [25.0]
