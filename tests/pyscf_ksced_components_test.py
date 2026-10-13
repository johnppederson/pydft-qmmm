"""Additive KSCED energy reporting and its full-system force contract."""
from __future__ import annotations

import os

import numpy as np
import pytest

from pyscf_ksced_embedding_test import _calculator, _host, _fd_force, FORCE_SCALE
from pyscf_ksced_test import make_interface
from pydft_qmmm.interfaces.pyscf_pbc.pbc_embedding import ao_operator
from pydft_qmmm.interfaces.pyscf_pbc.pbc_interface import _nuclear_coupling
from pydft_qmmm.interfaces.pyscf.pyscf_backend import load_backend
from pydft_qmmm.utils import KJMOL_PER_EH, Subsystem

pytestmark = pytest.mark.slow

KEYS = {
    "KSCED Active Energy", "A-B Nuclear Interaction",
    "A-B Electron-Nuclear Interaction", "A-B Hartree Interaction",
    "Non-additive XC", "Non-additive Kinetic", "Electronic Entropy Correction",
    "A-C Near Electronic Embedding", "A-C Near Nuclear Embedding",
    "A-C Reciprocal Electronic Embedding", "A-C Reciprocal Nuclear Embedding",
}


@pytest.mark.parametrize("case", ["off", "near", "pme", "combined", "uks",
                                 "frozen_uks", "smearing", "gga"])
def test_component_contract(case, monkeypatch):
    changes = {}
    if case == "uks":
        changes.update(charge=1, multiplicity=2)
    if case == "smearing":
        changes.update(sigma=.15, smearing_method="fermi")
    if case in ("frozen_uks", "gga"):
        changes["ksced"] = dict(active_atoms=[1, 3], frozen_atoms=[0],
                                t_nad="GGA_K_APBE" if case == "gga" else "LDA_K_TF")
        if case == "frozen_uks":
            changes["ksced"].update(frozen_charge=1, frozen_multiplicity=2)
    if case == "off":
        interface = make_interface(device=os.environ.get("KSCED_BACKEND", "cpu"))
    else:
        mode = case if case in ("near", "pme") else "combined"
        interface = _calculator(mode, **changes).potential
    system = interface.system
    energy = interface.compute_energy()
    forces = interface.compute_forces()
    state = interface._scf_state()
    summary = dict(state.method.scf_summary)
    core = _host(state.method.get_hcore()).copy()
    frozen = interface.frozen_method[0]
    frozen_dm = _host(frozen.make_rdm1()).copy()

    def forbidden(*args, **kwargs):
        pytest.fail("Component reporting reran an SCF")

    with monkeypatch.context() as patch:
        patch.setattr(state.method, "kernel", forbidden)
        patch.setattr(frozen, "kernel", forbidden)
        components = interface.compute_components()
        assert set(components) == KEYS
        assert all(type(value) is float and np.isfinite(value)
                   for value in components.values())
        assert sum(components.values()) == pytest.approx(energy, abs=2e-8)
        assert interface._scf_state() is state
        # No expensive XC or AO evaluation on a repeat request.
        nr_name = "nr_uks" if state.dm.ndim == 3 else "nr_rks"
        patch.setattr(state.method._numint, nr_name, forbidden)
        copy = interface.compute_components()
        copy["KSCED Active Energy"] = 123.
        assert interface.compute_components() == components
    np.testing.assert_array_equal(_host(state.method.get_hcore()), core)
    for key, value in summary.items():
        np.testing.assert_array_equal(_host(state.method.scf_summary[key]), _host(value))
    np.testing.assert_array_equal(_host(frozen.make_rdm1()), frozen_dm)
    np.testing.assert_allclose(interface.compute_forces(), forces, atol=1e-8, rtol=0)
    np.testing.assert_array_equal(forces[0], np.zeros(3))

    # Independent plain-A functional evaluated at the embedded density.
    plain = interface._method(state.cell, interface.multiplicity, False)
    dm = state.dm
    veff = plain.get_veff(state.cell, dm)
    own = float(plain.energy_tot(dm=dm, h1e=plain.get_hcore(), vhf=veff))
    assert components["KSCED Active Energy"] == pytest.approx(own*KJMOL_PER_EH, abs=2e-6)
    expected_nuc = (state.method.mol_ab.energy_nuc() - state.cell.energy_nuc()
                    - frozen.mol.energy_nuc()) * KJMOL_PER_EH
    assert components["A-B Nuclear Interaction"] == pytest.approx(expected_nuc, abs=1e-8)
    assert components["Non-additive Kinetic"] == pytest.approx(
        float(_host(state.method.e_tnad))*KJMOL_PER_EH, abs=1e-8)
    correction = (state.method.energy_potential()-state.method.e_tot)*KJMOL_PER_EH
    assert components["Electronic Entropy Correction"] == pytest.approx(correction, abs=1e-8)
    if case == "smearing":
        assert correction < -1e-5
    else:
        assert correction == 0

    if case == "off":
        assert all(value == 0 for key, value in components.items() if key.startswith("A-C"))
    else:
        # Compute the reciprocal operator directly, rather than by subtraction.
        emb = state.embedding_state
        field = sum((p.compute_potential(emb.coords).reshape(-1) for p in emb.potentials),
                    np.zeros(len(emb.coords)))
        matrix = ao_operator(load_backend(interface.device), state.cell, emb.kpts,
                             emb.coords, emb.weights, field)[0]
        total_dm = _host(dm)
        if total_dm.ndim == 3:
            total_dm = total_dm.sum(axis=0)
        rec_e = float(np.einsum('ij,ji->', total_dm, matrix).real)*KJMOL_PER_EH
        rec_n = _nuclear_coupling(state.cell, field, system.box)*KJMOL_PER_EH
        assert components["A-C Reciprocal Electronic Embedding"] == pytest.approx(rec_e, abs=1e-8)
        assert components["A-C Reciprocal Nuclear Embedding"] == pytest.approx(rec_n, abs=1e-8)
        for kind, total in (("Electronic", state.electronic_energy),
                            ("Nuclear", state.nuclear_energy)):
            assert (components[f"A-C Near {kind} Embedding"]
                    + components[f"A-C Reciprocal {kind} Embedding"]) == pytest.approx(
                        total*KJMOL_PER_EH, abs=1e-8)
        if case == "pme":
            assert abs(components["A-C Near Electronic Embedding"]) < 1e-12
        if case == "near":
            assert components["A-C Reciprocal Electronic Embedding"] == 0

    # Component queries must not change the conservative force contract.
    for atom in ((1,) if case == "off" else (1, 2, 4)):
        numeric = _fd_force(interface, atom, 0, 5e-4)
        error = abs(forces[atom, 0]-numeric)/FORCE_SCALE
        print(f"components FD {case} atom={atom} error_Ha_Bohr={error:.8g}", flush=True)
        assert error < (1e-5 if atom == 1 else 1e-6)

    if case == "combined":
        original = interface.compute_components()
        system.charges[0] = -.83
        system.subsystems[0] = Subsystem.III
        changed_b = interface.compute_components()
        for key in KEYS:
            assert changed_b[key] == pytest.approx(original[key], abs=2e-6)
        before = interface._cached_components()
        positions = np.asarray(system.positions).copy()
        positions[2, 0] += .03
        system.positions = positions
        moved = interface.compute_components()
        assert interface._cached_components() is not before
        assert sum(moved.values()) == pytest.approx(interface.compute_energy(), abs=2e-8)
        assert abs(moved["A-C Near Electronic Embedding"] - changed_b["A-C Near Electronic Embedding"]) > 1e-6
        system.charges[4] *= 1.2
        charged = interface.compute_components()
        assert sum(charged.values()) == pytest.approx(interface.compute_energy(), abs=2e-8)
        assert abs(charged["A-C Reciprocal Electronic Embedding"] - moved["A-C Reciprocal Electronic Embedding"]) > 1e-6
        assert interface.frozen_method[0] is frozen
        np.testing.assert_array_equal(_host(frozen.make_rdm1()), frozen_dm)
        # Cached reporting must still reject movement of frozen B.
        positions = np.asarray(system.positions).copy()
        positions[0, 0] += .01
        system.positions = positions
        with pytest.raises(ValueError, match="frozen atom"):
            interface.compute_components()
