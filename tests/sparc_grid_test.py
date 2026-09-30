from __future__ import annotations

import numpy as np
import pytest

from pydft_qmmm.interfaces.sparc import sparc_grid


def test_vext_to_sparc_converts_ev_to_hartree_without_flipping_sign():
    v_ext = np.array([[[1.0, -2.0]]])
    result = sparc_grid.vext_to_sparc(v_ext)
    assert result[0, 0, 0] == pytest.approx(1.0 / 27.211386245988)
    # The sign MUST survive: both sides use electron potential energy.
    assert result[0, 0, 1] < 0.0


def test_phi_from_sparc_converts_hartree_to_volts_and_negates():
    phi = np.array([[[1.0, -2.0]]])
    result = sparc_grid.phi_from_sparc(phi)
    assert result[0, 0, 0] == pytest.approx(-27.211386245988)
    assert result[0, 0, 1] == pytest.approx(2.0 * 27.211386245988)


def test_round_trip_recovers_the_field_up_to_the_documented_negation():
    rng = np.random.default_rng(20260830)
    field_ev = rng.normal(size=(5, 4, 3))
    hartree = sparc_grid.vext_to_sparc(field_ev)
    volts = sparc_grid.phi_from_sparc(hartree)
    assert volts == pytest.approx(-field_ev)


def test_cell_to_bohr():
    cell = np.diag([1.0, 2.0, 3.0])
    result = sparc_grid.cell_to_bohr(cell)
    assert result[1, 1] == pytest.approx(2.0 * 1.8897261246257702)


def test_shape_is_preserved():
    field = np.zeros((7, 5, 3))
    assert sparc_grid.vext_to_sparc(field).shape == (7, 5, 3)
    assert sparc_grid.phi_from_sparc(field).shape == (7, 5, 3)
