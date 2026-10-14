"""The reciprocal QM->MM force on PME sources is the energy gradient."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pydft_qmmm import Atom
from pydft_qmmm import System
from pydft_qmmm.embedding.grid_potential import poisson_fft
from pydft_qmmm.embedding.grid_potential import spline_value_and_gradient
from pydft_qmmm.embedding.grid_potential import spread_gaussian
from pydft_qmmm.embedding.pme_grid import build_pme_potential
from pydft_qmmm.embedding.pme_grid import charge_from_potential
from pydft_qmmm.embedding.pme_grid import KJMOL_PER_EV
from pydft_qmmm.embedding.pme_grid import pme_source_forces
from pydft_qmmm.embedding.pme_grid import write_pme_data
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


@pytest.mark.parametrize("excluded", [(), (2,)])
def test_grid_source_forces_are_the_energy_gradient(tmp_path, excluded):
    path = tmp_path / "PME_DATA"
    positions, charges = _sources()
    _, phi = _electrons()
    grid = charge_from_potential(phi, CELL)

    def energy(moved):
        write_pme_data(path, moved, charges, excluded, *PME, 0)
        return float((grid * _field(path)).sum())

    write_pme_data(path, positions, charges, excluded, *PME, 0)
    forces = pme_source_forces(path, grid, SHAPE, CELL)
    expected = _central_difference(energy, positions, range(2, 6))
    np.testing.assert_allclose(forces, expected, atol=1e-6)
    assert np.abs(forces[:2]).max() == 0.
    assert np.abs(forces).max() > 1.


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


def test_sparc_reads_region_iii_reaction_from_phi(tmp_path):
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
    interface.potentials.append(SimpleNamespace(
        pme_alpha=PME[0], pme_gridnumber=PME[1], pme_spline_order=PME[2],
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
        write_pme_data(path, moved, system.charges * [0, 0, 1], [], *PME, 4)
        return float((rho * volume * _field(path)).sum())

    expected = _central_difference(
        energy, np.asarray(system.positions), [2],
    )
    np.testing.assert_allclose(forces, expected, atol=1e-6)
    assert np.abs(forces[2]).max() > 0.1
