"""SCF integration tests; submit ../pbc-runs/integration.slurm on a compute node."""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.slow


def test_embedding_shifts_the_energy(pyscf_pbc_embedded_interface):
    """Embedding must shift the energy without a quadrature blowup."""
    interface, bare = pyscf_pbc_embedded_interface
    embedded = interface.compute_energy()
    assert embedded != pytest.approx(bare, abs=1e-6)
    # A large shift signals unresolved density quadrature.
    assert abs(embedded - bare) < 1000.0


def test_components_sum_to_the_energy(pyscf_pbc_embedded_interface):
    interface, _ = pyscf_pbc_embedded_interface
    components = interface.compute_components()
    assert sum(components.values()) == pytest.approx(
        interface.compute_energy(), rel=1e-9,
    )


def _central_difference(interface, atom, step):
    """The numerical force on one atom, by central differences."""
    system = interface.system
    numeric = np.zeros(3)
    for axis in range(3):
        original = float(system.positions[atom, axis])
        positions = np.asarray(system.positions).copy()
        positions[atom, axis] = original + step
        system.positions = positions
        plus = interface.compute_energy()
        positions = np.asarray(system.positions).copy()
        positions[atom, axis] = original - step
        system.positions = positions
        minus = interface.compute_energy()
        positions = np.asarray(system.positions).copy()
        positions[atom, axis] = original
        system.positions = positions
        numeric[axis] = -(plus - minus) / (2.0 * step)
    return numeric


def test_qm_forces_match_central_differences(pyscf_pbc_embedded_interface):
    """Every QM force channel must differentiate the reported energy."""
    interface, _ = pyscf_pbc_embedded_interface
    atom = sorted(interface.system.select("subsystem I"))[0]
    analytic = interface.compute_forces()[atom]
    numeric = _central_difference(interface, atom, 1e-3)
    np.testing.assert_allclose(analytic, numeric, rtol=5e-3, atol=1.0)


def test_pulay_term_is_live(pyscf_pbc_embedded_interface, monkeypatch):
    """Omitting the AO Pulay term must break force/energy agreement."""
    from pydft_qmmm.interfaces.pyscf_pbc import pbc_forces
    interface, _ = pyscf_pbc_embedded_interface
    atom = sorted(interface.system.select("subsystem I"))[0]
    numeric = _central_difference(interface, atom, 1e-3)
    monkeypatch.setattr(
        pbc_forces,
        "pulay_forces",
        lambda backend, state, natoms: np.zeros((natoms, 3)),
    )
    crippled = interface.compute_forces()[atom]
    assert not np.allclose(crippled, numeric, rtol=5e-3, atol=1.0)


def test_mm_forces_match_central_differences(pyscf_pbc_embedded_interface):
    """The reaction on a static MM charge must differentiate the energy."""
    interface, _ = pyscf_pbc_embedded_interface
    atom = sorted(interface.system.select("subsystem II"))[0]
    analytic = interface.compute_forces()[atom]
    numeric = _central_difference(interface, atom, 2e-3)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-2, atol=1.0)


def test_embedding_channels_conserve_momentum(pyscf_pbc_embedded_interface):
    """Embedding forces must cancel within grid error.

    Exclude PySCF's bare gradient, which has its own grid-dependent residual.
    """
    from pydft_qmmm.interfaces.pyscf_pbc import pbc_forces
    from pydft_qmmm.interfaces.pyscf.pyscf_backend import load_backend
    from pydft_qmmm.utils import BOHR_PER_ANGSTROM, KJMOL_PER_EH
    interface, _ = pyscf_pbc_embedded_interface
    state = interface._scf_state()
    backend = load_backend(interface.device)
    natoms = len(interface.system.positions)
    scale = KJMOL_PER_EH * BOHR_PER_ANGSTROM
    embedding = (
        pbc_forces.pulay_forces(backend, state, natoms)
        + pbc_forces.nuclear_forces(state, interface.system.box, natoms)
        + pbc_forces.mm_forces(
            backend, state, interface.system,
            interface.embedding_sigma, natoms,
        )
    ) * scale
    np.testing.assert_allclose(
        embedding.sum(axis=0), np.zeros(3), atol=0.5,
    )


def test_total_force_is_pyscf_s_residual(pyscf_pbc_embedded_interface):
    """The total force residual must be small at the fixture's FFT cutoff."""
    interface, _ = pyscf_pbc_embedded_interface
    total = interface.compute_forces().sum(axis=0)
    np.testing.assert_allclose(total, np.zeros(3), atol=5.0)


def test_cpu_and_gpu_agree(pyscf_pbc_embedded_factory):
    """CPU/GPU energies and forces must agree within backend grid tolerances."""
    # A CPU node may have GPU4PySCF installed but lack CUDA libraries.
    try:
        import gpu4pyscf                                     # noqa: F401
    except ImportError as error:
        pytest.skip(f"GPU4PySCF unavailable on this node: {error}")
    cpu = pyscf_pbc_embedded_factory("cpu")
    gpu = pyscf_pbc_embedded_factory("gpu")
    assert gpu.compute_energy() == pytest.approx(
        cpu.compute_energy(), abs=2.0,
    )
    np.testing.assert_allclose(
        gpu.compute_forces(), cpu.compute_forces(), rtol=1e-2, atol=1.0,
    )


def test_reciprocal_reaction_on_subsystem_iii_is_exact(
        pyscf_pbc_pme_interface,
):
    """Region III reaction forces must differentiate the full embedding energy."""
    interface = pyscf_pbc_pme_interface
    assert interface.potentials, "fixture must register a PME potential"
    system = interface.system
    atom = sorted(system.select("subsystem III"))[0]
    analytic = interface.compute_forces()[atom]
    numeric = _central_difference(interface, atom, 2e-3)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-3)


def test_pme_qm_force_matches_central_differences(pyscf_pbc_pme_interface):
    """The QM force must differentiate the energy when PME is active."""
    interface = pyscf_pbc_pme_interface
    atom = sorted(interface.system.select("subsystem I"))[0]
    analytic = interface.compute_forces()[atom]
    numeric = _central_difference(interface, atom, 2e-3)
    np.testing.assert_allclose(analytic, numeric, rtol=5e-3, atol=1.0)
