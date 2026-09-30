"""Tests for shared software-interface contracts."""

import pytest

from pydft_qmmm.interfaces import ElectrostaticCouplingMode
from pydft_qmmm.interfaces.interface import QMInterface
from pydft_qmmm.interfaces.psi4.psi4_interface import Psi4Interface
from pydft_qmmm.interfaces.pyscf.pyscf_interface import PySCFInterface
from pydft_qmmm.interfaces.vasp.vasp_interface import VaspInterface


def test_unspecified_qm_engine_rejects_electrostatic_coupling():
    """Unspecified engines do not support electrostatic coupling."""
    assert QMInterface.electrostatic_coupling_mode(None) is (
        ElectrostaticCouplingMode.UNSUPPORTED
    )


def test_qm_engines_declare_their_electrostatic_owner():
    """Each engine declares its electrostatic owner."""
    assert PySCFInterface.electrostatic_coupling_mode(None) is (
        ElectrostaticCouplingMode.MOLECULAR
    )
    assert Psi4Interface.electrostatic_coupling_mode(None) is (
        ElectrostaticCouplingMode.MOLECULAR
    )
    assert VaspInterface.electrostatic_coupling_mode(None) is (
        ElectrostaticCouplingMode.ENGINE
    )


def test_sparc_owns_electrostatic_coupling():
    """The SPARC fork owns electronic and nuclear embedding terms."""
    pytest.importorskip("sparc")
    from pydft_qmmm.interfaces.sparc.sparc_interface import SPARCInterface

    assert SPARCInterface.electrostatic_coupling_mode(None) is (
        ElectrostaticCouplingMode.ENGINE
    )
