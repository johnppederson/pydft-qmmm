"""Regression tests for the VASP periodic embedding source split."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pydft_qmmm import Atom
from pydft_qmmm import System
from pydft_qmmm.embedding.grid_potential import read_mm_charges
from pydft_qmmm.embedding.pme_grid import build_pme_potential
from pydft_qmmm.embedding.pme_grid import read_pme_data
from pydft_qmmm.interfaces.vasp.vasp_interface import VaspInterface
from pydft_qmmm.interfaces.vasp.vasp_plugin import ERFC_NEAR_ENV
from pydft_qmmm.utils import Subsystem


@pytest.fixture
def interface(tmp_path, monkeypatch):
    monkeypatch.delenv(ERFC_NEAR_ENV, raising=False)
    system = System([
        Atom(position=np.array([2., 2., 2.]), charge=-0.8,
             element="O", subsystem=Subsystem.I),
        Atom(position=np.array([3., 2., 2.]), charge=0.4,
             element="H", subsystem=Subsystem.II),
        Atom(position=np.array([7., 6., 5.]), charge=0.4,
             element="H", subsystem=Subsystem.III),
    ], box=np.eye(3) * 10.)
    result = VaspInterface(
        system=system, charge=0, directory=str(tmp_path), command="",
        incar={}, kpts=(1, 1, 1), pp_path="", potcar_map={}, embedding=True,
    )
    result.potentials.append(SimpleNamespace(
        pme_alpha=0.5, pme_gridnumber=(20, 20, 20), pme_spline_order=5,
    ))
    return result


@pytest.mark.parametrize("near", [[], [1], [1, 2]])
def test_fft_exports_only_region_iii_without_mutating_charges(
        interface, tmp_path, near,
):
    """Cover empty II, all three regions, and empty III (infinite cutoff)."""
    system = interface.system
    system.subsystems[1:] = Subsystem.III
    for i in near:
        system.subsystems[i] = Subsystem.II
    physical_charges = np.array(system.charges, copy=True)
    positions = np.array(system.positions, copy=True)
    assert interface._write_pme_data() == len(system)
    points, charges, excluded, alpha, grid, order, step = read_pme_data(
        tmp_path / "PME_DATA",
    )
    expected = physical_charges.copy()
    expected[[0, *near]] = 0.
    np.testing.assert_array_equal(charges, expected)
    np.testing.assert_array_equal(points, positions)
    assert excluded.size == 0
    assert (alpha, grid, order, step) == (0.5, (20, 20, 20), 5, 0)
    np.testing.assert_array_equal(system.charges, physical_charges)
    assert interface._write_mm_charges() == len(near)
    near_points, near_charges, _, _ = read_mm_charges(
        tmp_path / "MM_CHARGES",
    )
    np.testing.assert_array_equal(near_charges, physical_charges[near])
    np.testing.assert_array_equal(near_points, positions[near])


def test_erfc_keeps_region_ii_in_reciprocal_sum(
        interface, tmp_path, monkeypatch,
):
    monkeypatch.setenv(ERFC_NEAR_ENV, "1")
    interface._write_pme_data()
    _, charges, excluded, *_ = read_pme_data(tmp_path / "PME_DATA")
    np.testing.assert_array_equal(charges, interface.system.charges)
    np.testing.assert_array_equal(excluded, [0])


def test_pme_source_selection_updates_with_partition(interface, tmp_path):
    interface._write_pme_data()
    interface.system.subsystems[1] = Subsystem.III
    interface.system.subsystems[2] = Subsystem.II
    interface.frame[0] = 7
    interface._write_pme_data()
    _, charges, excluded, *_, step = read_pme_data(tmp_path / "PME_DATA")
    np.testing.assert_array_equal(charges, [0., 0.4, 0.])
    assert excluded.size == 0
    assert step == 7


def test_empty_iii_produces_zero_pme_without_exclusion_call(
        interface, tmp_path, monkeypatch,
):
    import helpme_py
    original = helpme_py.PMEInstanceD

    class NoExclusionPME:
        def __init__(self):
            self.pme = original()

        def __getattr__(self, name):
            return getattr(self.pme, name)

        def compute_P_adj(self, *args):
            raise AssertionError("Empty exclusion sets must not be subtracted")

    monkeypatch.setattr(helpme_py, "PMEInstanceD", NoExclusionPME)
    interface.system.subsystems[2] = Subsystem.II
    interface._write_pme_data()
    field = build_pme_potential(
        tmp_path / "PME_DATA", (8, 8, 8), interface.system.box,
    )
    np.testing.assert_array_equal(field, np.zeros((8, 8, 8)))


def test_removed_sources_cannot_change_the_pme_field(interface, tmp_path):
    interface._write_pme_data()
    field = build_pme_potential(
        tmp_path / "PME_DATA", (8, 8, 8), interface.system.box,
    )
    assert np.max(np.abs(field)) > 0.01
    interface.system.positions[:2] += np.array([0.3, -0.2, 0.1])
    interface.system.charges[:2] *= 2
    interface._write_pme_data()
    updated = build_pme_potential(
        tmp_path / "PME_DATA", (8, 8, 8), interface.system.box,
    )
    np.testing.assert_allclose(updated, field, atol=1e-12, rtol=0)
