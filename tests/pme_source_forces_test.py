"""The reciprocal QM->MM force on PME sources is the energy gradient."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pydft_qmmm import Atom
from pydft_qmmm import System
from pydft_qmmm.potentials.embedding.grid_potential import poisson_fft
from pydft_qmmm.potentials.embedding.grid_potential import spline_value_and_gradient
from pydft_qmmm.potentials.embedding.grid_potential import spread_gaussian
from pydft_qmmm.potentials.embedding.pme_grid import build_pme_potential
from pydft_qmmm.potentials.embedding.pme_grid import charge_from_potential
from pydft_qmmm.potentials.embedding.pme_grid import KJMOL_PER_EV
from pydft_qmmm.potentials.embedding.pme_grid import pme_source_forces
from pydft_qmmm.potentials.embedding.pme_grid import write_pme_data
from pydft_qmmm.utils import Subsystem

CELL = np.diag([10., 11., 12.])
SHAPE = (30, 32, 36)
PME = (0.5, (24, 24, 24), 5)
NUCLEI = np.array([[5., 5., 6.], [5.8, 5.5, 6.]])
VALENCE = np.array([6., 1.])


def _electrons():
    """A Gaussian electron cloud and its potential (V)."""
    rho = spread_gaussian(
        np.array([[5.2, 5.2, 6.]]), np.array([-7.]), SHAPE, CELL, 0.5,
    )
    return rho, poisson_fft(rho, CELL)


def _sources():
    rng = np.random.default_rng(1)
    positions = rng.uniform(0, 10, (6, 3))
    charges = rng.normal(size=6)
    charges[[0, 1]] = 0.   # removed from the PME sources, as I and II are
    return positions, charges


def _field(path):
    """The PME electrostatic potential (kJ/mol/e) on the grid."""
    return -build_pme_potential(path, SHAPE, CELL) * KJMOL_PER_EV


def _central_difference(energy, positions, atoms, step=1e-4):
    forces = np.zeros((len(positions), 3))
    for atom in atoms:
        for axis in range(3):
            plus, minus = positions.copy(), positions.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            forces[atom, axis] = -(energy(plus) - energy(minus)) / (2 * step)
    return forces


def test_charge_from_potential_inverts_poisson():
    rho, phi = _electrons()
    volume = abs(np.linalg.det(CELL)) / rho.size
    np.testing.assert_allclose(
        charge_from_potential(phi, CELL), (rho - rho.mean()) * volume,
        atol=1e-12,
    )


@pytest.mark.parametrize("count,excluded", [(0, ()), (6, ()), (6, (2,))])
def test_zero_sources_skip_pme_and_grid_setup(tmp_path, monkeypatch, count, excluded):
    import helpme_py
    from pydft_qmmm.potentials.embedding import pme_grid

    def unexpected_work(*args, **kwargs):
        raise AssertionError("Zero sources must skip PME and grid setup")

    monkeypatch.setattr(helpme_py, "PMEInstanceD", unexpected_work)
    monkeypatch.setattr(pme_grid, "grid_coordinates", unexpected_work)
    path = tmp_path / "PME_DATA"
    write_pme_data(path, np.zeros((count, 3)), np.zeros(count), excluded, *PME, 7)
    field = build_pme_potential(path, SHAPE, CELL, expect_step=7)
    forces = pme_source_forces(
        path, np.ones(SHAPE), SHAPE, CELL,
        points=NUCLEI, point_charges=VALENCE, expect_step=7,
    )
    np.testing.assert_array_equal(field, np.zeros(SHAPE))
    np.testing.assert_array_equal(forces, np.zeros((count, 3)))
    # Skipping zero fields must not allow stale persistent frames.
    with pytest.raises(ValueError, match="expected 8"):
        build_pme_potential(path, SHAPE, CELL, expect_step=8)
    with pytest.raises(ValueError, match="expected 8"):
        pme_source_forces(path, np.ones(SHAPE), SHAPE, CELL, expect_step=8)


@pytest.mark.parametrize("matched_grid", [False, True])
@pytest.mark.parametrize("excluded", [(), (2,)])
def test_grid_source_forces_are_the_energy_gradient(
        tmp_path, excluded, matched_grid,
):
    path = tmp_path / "PME_DATA"
    positions, charges = _sources()
    _, phi = _electrons()
    grid = charge_from_potential(phi, CELL)
    parameters = (PME[0], SHAPE, 5) if matched_grid else PME

    def energy(moved):
        write_pme_data(path, moved, charges, excluded, *parameters, 0)
        return float((grid * _field(path)).sum())

    write_pme_data(path, positions, charges, excluded, *parameters, 0)
    forces = pme_source_forces(path, grid, SHAPE, CELL)
    expected = _central_difference(energy, positions, range(2, 6))
    np.testing.assert_allclose(forces, expected, atol=1e-6)
    assert np.abs(forces[:2]).max() == 0.
    assert np.abs(forces).max() > 1.


def test_native_grid_matches_point_evaluation(tmp_path, monkeypatch):
    import helpme_py
    from pydft_qmmm.potentials.embedding import pme_grid
    if (
            not hasattr(helpme_py.PMEInstanceD, "compute_P_rec_grid")
            or np.lib.NumpyVersion(helpme_py.__version__) < "0.3.1"
    ):
        pytest.skip("Native grid evaluation requires helpme-py >= 0.3.1")
    positions, charges = _sources()
    path = tmp_path / "PME_DATA"
    write_pme_data(path, positions, charges, [], PME[0], SHAPE, 5, 0)
    with monkeypatch.context() as patch:
        def no_point_grid(*args):
            raise AssertionError("Matched grids should use compute_P_rec_grid")
        patch.setattr(pme_grid, "grid_coordinates", no_point_grid)
        native = _field(path)
    # 0.3.0 exposed the method but had the convolution bug; use the
    # established point path for that release and compare its output.
    monkeypatch.setattr(helpme_py, "__version__", "0.3.0")
    points = _field(path)
    np.testing.assert_allclose(native, points, rtol=1e-11, atol=1e-10)


@pytest.mark.parametrize("order", [4, 5, 6, 10])
@pytest.mark.parametrize("shape", [(12, 14, 16), (15, 15, 15)])
def test_native_grid_force_transpose(tmp_path, monkeypatch, order, shape):
    """Charged, nonsmooth densities exercise phases and Nyquist modes."""
    import helpme_py
    from pydft_qmmm.potentials.embedding import pme_grid
    if np.lib.NumpyVersion(helpme_py.__version__) < "0.3.1":
        pytest.skip("Native grid evaluation requires helpme-py >= 0.3.1")
    rng = np.random.default_rng(13)
    positions = np.array([[0., 11., -12.], [1.13, 3.72, 6.39], [2., 3., 4.]])
    charges = np.array([1., -.3, 0.])
    density = rng.normal(size=shape)
    path = tmp_path / "PME_DATA"
    write_pme_data(path, positions, charges, [], 1.2, shape, order, 8)
    with monkeypatch.context() as patch:
        def no_particles(*args):
            raise AssertionError("Native forces must not allocate grid-point particles")
        patch.setattr(pme_grid, "grid_coordinates", no_particles)
        actual = pme_source_forces(path, density, shape, CELL, expect_step=8)
    with monkeypatch.context() as patch:
        patch.setattr(helpme_py, "__version__", "0.3.0")
        reference = pme_source_forces(path, density, shape, CELL, expect_step=8)
    np.testing.assert_allclose(actual, reference, rtol=1e-10, atol=1e-8)
    np.testing.assert_array_equal(actual[2], 0.)

    def energy(moved):
        write_pme_data(path, moved, charges, [], 1.2, shape, order, 8)
        return -KJMOL_PER_EV * np.sum(density * build_pme_potential(path, shape, CELL))

    expected = _central_difference(energy, positions, [1], step=1e-5)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-6, atol=1e-5)


@pytest.mark.parametrize("option", [
    "grid", "cell", "origin", "excluded", "points", "version", "skew", "order",
])
def test_native_force_fallback(tmp_path, monkeypatch, option):
    import helpme_py
    from pydft_qmmm.potentials.embedding import pme_grid_forces

    def unexpected_fast_path(*args):
        raise AssertionError("This geometry requires the point evaluator")

    monkeypatch.setattr(pme_grid_forces, "native_grid_source_forces", unexpected_fast_path)
    positions, charges = _sources()
    path = tmp_path / "PME_DATA"
    grid = (24, 24, 24) if option == "grid" else SHAPE
    write_pme_data(
        path, positions, charges, [2] if option == "excluded" else [],
        .5, grid, 3 if option == "order" else 6, 0,
    )
    kwargs = {}
    cell = CELL.copy()
    if option == "cell":
        kwargs["source_cell"] = CELL * 1.1
    elif option == "origin":
        kwargs["origin"] = np.array([.1, 0., 0.])
    elif option == "points":
        kwargs.update(points=NUCLEI, point_charges=VALENCE)
    elif option == "version":
        monkeypatch.setattr(helpme_py, "__version__", "0.3.0")
    elif option == "skew":
        cell[1, 0] = .5
    result = pme_source_forces(path, np.ones(SHAPE) / np.prod(SHAPE), SHAPE, cell, **kwargs)
    assert np.isfinite(result).all()


def test_padded_mesh_uses_point_evaluation(tmp_path, monkeypatch):
    """helPME pads 17 to 18; requested equality does not mean equal meshes."""
    import helpme_py
    from pydft_qmmm.potentials.embedding import pme_grid_forces

    def unexpected_grid(*args):
        raise AssertionError("A padded mesh must use the point evaluator")

    monkeypatch.setattr(
        helpme_py.PMEInstanceD, "compute_P_rec_grid", unexpected_grid,
        raising=False,
    )
    monkeypatch.setattr(pme_grid_forces, "native_grid_source_forces", unexpected_grid)
    shape = (17, 18, 20)
    positions, charges = _sources()
    density = np.random.default_rng(2).normal(size=shape)
    path = tmp_path / "PME_DATA"

    def energy(moved):
        write_pme_data(path, moved, charges, [], .5, shape, 6, 0)
        return -KJMOL_PER_EV * np.sum(density * build_pme_potential(path, shape, CELL))

    energy(positions)
    forces = pme_source_forces(path, density, shape, CELL)
    expected = _central_difference(energy, positions, [2])
    np.testing.assert_allclose(forces[2], expected[2], rtol=1e-6, atol=1e-5)


def test_vasp_plugin_writes_region_iii_reaction(tmp_path, monkeypatch):
    """Electrons via the final density, nuclei via the ZVAL spline."""
    from pydft_qmmm.interfaces.vasp import vasp_plugin
    from pydft_qmmm.interfaces.vasp import vasp_utils
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(vasp_plugin.ERFC_NEAR_ENV, raising=False)
    monkeypatch.delenv(vasp_plugin.NUCLEAR_FIELD_ENV, raising=False)
    positions, charges = _sources()
    _, phi = _electrons()
    grid = charge_from_potential(phi, CELL)
    write_pme_data("PME_DATA", positions, charges, [], *PME, 3)
    vasp_utils.write_mm_charges(
        "MM_CHARGES", np.zeros((0, 3)), np.zeros(0), 3, 0.3,
    )
    vasp_plugin.reset_cache()
    # VASP's charge_density is N_grid * electrons per point.
    density = (0.01 - grid) * np.prod(SHAPE)
    constants = SimpleNamespace(
        lattice_vectors=CELL, shape_grid=SHAPE,
        positions=NUCLEI @ np.linalg.inv(CELL), ZVAL=VALENCE,
        ion_types=[0, 1], forces=np.zeros((2, 3)), charge_density=density,
    )
    additions = SimpleNamespace(total_energy=0., forces=np.zeros((2, 3)))
    vasp_plugin.force_and_stress(constants, additions)
    forces, step = vasp_utils.read_mm_forces("PME_FORCES")
    assert step == 3

    def energy(moved):
        write_pme_data("PME_DATA", moved, charges, [], *PME, 3)
        field = _field("PME_DATA")
        nuclear, _ = spline_value_and_gradient(field, CELL, NUCLEI)
        # The uniform part is not exactly G=0 once spread onto the PME grid.
        return float(((grid - 0.01) * field).sum() + VALENCE @ nuclear)

    expected = _central_difference(energy, positions, range(2, 6))
    np.testing.assert_allclose(forces, expected, atol=1e-6)
    assert np.abs(forces).max() > 1.


@pytest.mark.parametrize("native", [False, True])
def test_sparc_reads_region_iii_reaction_from_phi(tmp_path, monkeypatch, native):
    pytest.importorskip("sparc")
    from pydft_qmmm.interfaces.sparc import sparc_grid
    from pydft_qmmm.interfaces.sparc import sparc_utils
    from pydft_qmmm.interfaces.sparc.sparc_interface import PHI_FILE
    from pydft_qmmm.interfaces.sparc.sparc_interface import SPARCInterface
    system = System([
        Atom(position=np.array([5., 5., 6.]), charge=-0.8,
             element="O", subsystem=Subsystem.I),
        Atom(position=np.array([6., 5., 6.]), charge=0.4,
             element="H", subsystem=Subsystem.II),
        Atom(position=np.array([1., 2., 9.]), charge=0.4,
             element="H", subsystem=Subsystem.III),
    ], box=CELL)
    interface = SPARCInterface(
        system=system, charge=0, directory=str(tmp_path), calculator=None,
        fd_grid=SHAPE, embedding=True,
    )
    parameters = (PME[0], SHAPE, 6) if native else PME
    if native:
        from pydft_qmmm.potentials.embedding import pme_grid
        def unexpected_points(*args, **kwargs):
            raise AssertionError("SPARC matched grids must use native PME")
        monkeypatch.setattr(pme_grid, "grid_coordinates", unexpected_points)
    interface.potentials.append(SimpleNamespace(
        pme_alpha=parameters[0], pme_gridnumber=parameters[1],
        pme_spline_order=parameters[2],
    ))
    interface.frame[0] = 4
    interface.vext_step[0] = 4
    path = interface._write_pme_data()
    rho, phi = _electrons()
    # SPARC writes the electron potential energy in Hartree.
    sparc_utils.write_grid_file(
        str(tmp_path / PHI_FILE), -phi / sparc_grid.EV_PER_HARTREE,
        sparc_grid.cell_to_bohr(CELL), 4,
    )
    forces = interface._read_pme_forces()
    volume = abs(np.linalg.det(CELL)) / rho.size

    def energy(moved):
        write_pme_data(path, moved, system.charges * [0, 0, 1], [], *parameters, 4)
        return float((rho * volume * _field(path)).sum())

    expected = _central_difference(
        energy, np.asarray(system.positions), [2],
    )
    np.testing.assert_allclose(forces, expected, atol=1e-6)
    assert np.abs(forces[2]).max() > 0.1


def test_runtime_grid_matches_explicit_native_grid(tmp_path, monkeypatch):
    from pydft_qmmm.potentials.embedding import pme_grid
    path = tmp_path/'PME_DATA'
    positions, charges = _sources()
    q = np.random.default_rng(20).normal(size=SHAPE)
    def evaluate(grid):
        write_pme_data(path, positions, charges, [], .5, grid, 6, 0)
        return (build_pme_potential(path, SHAPE, CELL),
                pme_source_forces(path, q, SHAPE, CELL))
    def unexpected_points(*args, **kwargs):
        raise AssertionError('Default PME must not construct point coordinates')
    monkeypatch.setattr(pme_grid, 'grid_coordinates', unexpected_points)
    explicit = evaluate(SHAPE)
    automatic = evaluate((0, 0, 0))
    for actual, expected in zip(automatic, explicit):
        np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match='matching QM/MM cells'):
        build_pme_potential(path, SHAPE, CELL, source_cell=CELL*2)
