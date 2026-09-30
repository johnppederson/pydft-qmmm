"""Tests for QM/MM coupling assembly."""

import pytest

from pydft_qmmm import QMMMHamiltonian


def test_conservative_coupling_is_default():
    """Conservative coupling is the default."""
    coupling = QMMMHamiltonian("electrostatic", "electrostatic")
    assert coupling.coupling_mode == "conservative"


def test_unknown_coupling_mode_is_rejected():
    """Unknown coupling modes are rejected."""
    with pytest.raises(ValueError, match="conservative.*force"):
        QMMMHamiltonian(coupling_mode="unknown")


def test_force_mode_warns_that_energy_and_forces_are_inconsistent():
    """Legacy force mixing emits a warning."""
    with pytest.warns(RuntimeWarning, match="nonconservative"):
        coupling = QMMMHamiltonian(coupling_mode="force")
    assert coupling.coupling_mode == "force"
