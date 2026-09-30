"""Periodic PySCF unit tests without SCF iterations."""
from __future__ import annotations

import numpy as np
import pytest

from pydft_qmmm.interfaces.pyscf_pbc.pbc_factory import (
    pyscf_pbc_interface_factory,
)


def _kwargs(**overrides):
    base = dict(
        basis="gth-szv",
        pseudo="gth-pbe",
        charge=0,
        multiplicity=1,
        functional="pbe",
        ke_cutoff=80.0,
    )
    base.update(overrides)
    return base


def test_pyscf_pbc_is_bundled():
    from pydft_qmmm.interfaces import interfaces
    from pydft_qmmm.utils import TheoryLevel
    assert interfaces["pyscf-pbc"][0] is TheoryLevel.QM
    assert "pyscf_pbc" not in interfaces


def test_pseudo_is_mandatory(pyscf_pbc_system):
    with pytest.raises(ValueError, match="pseudo"):
        pyscf_pbc_interface_factory(pyscf_pbc_system, **_kwargs(pseudo=None))


def test_ecp_is_rejected(pyscf_pbc_system):
    with pytest.raises(ValueError, match="ecp"):
        pyscf_pbc_interface_factory(
            pyscf_pbc_system, **_kwargs(ecp="def2-svp"),
        )


def test_ke_cutoff_and_mesh_together_raise(pyscf_pbc_system):
    with pytest.raises(ValueError, match="ke_cutoff.*mesh|mesh.*ke_cutoff"):
        pyscf_pbc_interface_factory(
            pyscf_pbc_system, **_kwargs(mesh=(24, 24, 24)),
        )


def test_one_of_ke_cutoff_or_mesh_is_required(pyscf_pbc_system):
    with pytest.raises(ValueError, match="ke_cutoff|mesh"):
        pyscf_pbc_interface_factory(
            pyscf_pbc_system, **_kwargs(ke_cutoff=None),
        )


def test_embedding_sigma_must_be_positive(pyscf_pbc_system):
    with pytest.raises(ValueError, match="embedding_sigma"):
        pyscf_pbc_interface_factory(
            pyscf_pbc_system, **_kwargs(embedding_sigma=0.0),
        )


def test_a_valid_configuration_builds(pyscf_pbc_system):
    interface = pyscf_pbc_interface_factory(pyscf_pbc_system, **_kwargs())
    assert interface.pseudo == "gth-pbe"
    assert interface.ke_cutoff == 80.0
    assert interface.embedding_sigma == 0.3


def test_cell_lattice_matches_the_system_box(pyscf_pbc_system):
    from pydft_qmmm.interfaces.pyscf_pbc.pbc_cell import build_cell
    cell, qm_indices = build_cell(
        pyscf_pbc_system, "gth-szv", "gth-pbe",
        80.0, None, 0, 1, 0,
    )
    np.testing.assert_allclose(np.asarray(cell.a), pyscf_pbc_system.box)
    assert qm_indices == tuple(sorted(pyscf_pbc_system.select("subsystem I")))


def test_valence_charges_are_not_atomic_numbers(pyscf_pbc_system):
    from pydft_qmmm.interfaces.pyscf_pbc.pbc_cell import (
        build_cell, valence_charges,
    )
    cell, _ = build_cell(
        pyscf_pbc_system, "gth-szv", "gth-pbe",
        80.0, None, 0, 1, 0,
    )
    charges = valence_charges(cell)
    assert charges.sum() == pytest.approx(cell.nelectron)
    # GTH-PBE oxygen has valence charge 6.
    assert charges.max() == pytest.approx(6.0)


def test_near_potential_matches_the_analytic_smeared_coulomb():
    """A neutral Gaussian pair approaches the isolated erf(r/(√2σ))/r field.

    Neutrality avoids the G=0 background; finite-box images set the tolerance.
    """
    from scipy.special import erf
    from pydft_qmmm.interfaces.vasp.grid_potential import interpolate_at
    from pydft_qmmm.interfaces.pyscf_pbc.pbc_embedding import (
        _smeared_potential_grid,
    )
    box = np.eye(3) * 20.0
    mesh = (80, 80, 80)
    sigma = 0.5
    positions = np.array([[9.0, 10.0, 10.0], [11.0, 10.0, 10.0]])
    charges = np.array([1.0, -1.0])
    phi = _smeared_potential_grid(positions, charges, mesh, box, sigma)
    axis = np.linspace(1.2, 3.0, 7)
    probe = np.stack(
        [10.0 + axis, np.full_like(axis, 11.5), np.full_like(axis, 10.0)],
        axis=-1,
    )
    got = interpolate_at(phi, box, probe)
    expected = np.zeros(len(probe))
    for position, charge in zip(positions, charges):
        distance = np.linalg.norm(probe - position, axis=1)
        expected += charge * erf(
            distance / (np.sqrt(2.0) * sigma),
        ) / distance * 14.399645
    np.testing.assert_allclose(got, expected, rtol=3e-2)


def test_grid_weights_sum_to_the_cell_volume(pyscf_pbc_system):
    from pydft_qmmm.interfaces.pyscf_pbc.pbc_cell import build_cell
    from pydft_qmmm.interfaces.pyscf_pbc.pbc_embedding import (
        grid_coordinates,
    )
    cell, _ = build_cell(
        pyscf_pbc_system, "gth-szv", "gth-pbe", 80.0, None, 0, 1, 0,
    )
    coords, weights = grid_coordinates(cell)
    assert coords.shape == (weights.size, 3)
    assert weights.sum() == pytest.approx(cell.vol)


def test_gaussian_contraction_is_the_adjoint_of_spreading():
    """Gaussian reaction forces must differentiate the spread-charge energy."""
    from pydft_qmmm.interfaces.vasp.grid_potential import (
        contract_gaussian_gradient, spread_gaussian,
    )
    box = np.eye(3) * 12.0
    mesh = (48, 48, 48)
    sigma = 0.5
    position = np.array([[5.3, 6.1, 5.8]])
    charge = np.array([0.7])
    axes = [np.arange(n) / n for n in mesh]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    field = np.sin(2.0 * np.pi * grid[..., 0]) * np.cos(
        2.0 * np.pi * grid[..., 1],
    )
    analytic = contract_gaussian_gradient(
        field, position, charge, mesh, box, sigma,
    )[0]
    volume_element = abs(np.linalg.det(box)) / np.prod(mesh)
    step = 1e-3
    numeric = np.zeros(3)
    for axis in range(3):
        shifted = position.copy()
        shifted[0, axis] += step
        plus = np.sum(
            spread_gaussian(shifted, charge, mesh, box, sigma) * field,
        ) * volume_element
        shifted[0, axis] -= 2.0 * step
        minus = np.sum(
            spread_gaussian(shifted, charge, mesh, box, sigma) * field,
        ) * volume_element
        numeric[axis] = -(plus - minus) / (2.0 * step)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-3, atol=1e-6)


def test_constant_potential_recovers_the_electron_count(pyscf_pbc_system):
    """A constant electron potential gives V_ao = V × overlap."""
    import pyscf
    from pyscf import pbc as cpu_pbc
    from pydft_qmmm.interfaces.pyscf_pbc.pbc_cell import build_cell
    from pydft_qmmm.interfaces.pyscf_pbc.pbc_embedding import (
        ao_operator, grid_coordinates,
    )
    cell, _ = build_cell(
        pyscf_pbc_system, "gth-szv", "gth-pbe", 80.0, None, 0, 1, 0,
    )
    kpts = cell.make_kpts([1, 1, 1])
    coords, weights = grid_coordinates(cell)
    constant = 0.25
    potential = np.full(len(coords), constant)
    matrix = ao_operator(pyscf, cell, kpts, coords, weights, potential)
    mf = cpu_pbc.dft.KRKS(cell, kpts=kpts, xc="pbe")
    overlap = mf.get_ovlp()
    # Analytic overlap isolates quadrature error from guess normalization.
    np.testing.assert_allclose(
        matrix, constant * overlap, rtol=0, atol=1e-4,
    )
    dm = mf.get_init_guess()
    trace = np.einsum("kij,kji->", dm, matrix).real
    reference = constant * np.einsum("kij,kji->", dm, overlap).real
    assert trace == pytest.approx(reference, rel=1e-4)


def test_reciprocal_potential_matches_the_vasp_construction(tmp_path):
    """PySCF and VASP must give the same region III-only reciprocal field."""
    from pydft_qmmm.interfaces.vasp.pme_external import (
        build_pme_potential, grid_coordinates as vasp_grid,
    )
    from pydft_qmmm.interfaces.vasp import vasp_utils
    from pydft_qmmm.interfaces.pyscf_pbc.pbc_pme import (
        PeriodicPMEElectronicPotential,
    )
    from pydft_qmmm.utils import KJMOL_PER_EH, KJMOL_PER_EV, Subsystem
    from tests.conftest import build_water_system

    system = build_water_system(
        [
            ((4.0, 4.0, 4.0), Subsystem.I),
            ((4.3, 5.9, 6.1), Subsystem.II),
            ((9.0, 9.5, 9.2), Subsystem.III),
            ((2.0, 8.0, 3.0), Subsystem.III),
        ],
        box_length=12.0,
    )
    alpha, mesh, spline = 5.0, (30, 30, 30), 6
    box = np.asarray(system.box, dtype=np.float64)

    potential = PeriodicPMEElectronicPotential(system, alpha, mesh, spline)
    ours = np.asarray(
        potential.compute_potential(vasp_grid(mesh, box)),
    ).reshape(-1)

    path = str(tmp_path / "PME_DATA")
    charges = np.array(system.charges, copy=True)
    charges[sorted(system.select("not subsystem III"))] = 0.0
    vasp_utils.write_pme_data(
        path,
        np.asarray(system.positions),
        charges,
        [],
        alpha, mesh, spline, 0,
    )
    theirs = build_pme_potential(path, mesh, box).reshape(-1)
    theirs = theirs * KJMOL_PER_EV / KJMOL_PER_EH

    assert np.max(np.abs(ours)) > 1e-6
    np.testing.assert_allclose(ours, theirs, rtol=1e-8, atol=1e-10)
