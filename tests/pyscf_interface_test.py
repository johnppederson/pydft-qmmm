"""Tests for the bundled PySCF interface."""
from __future__ import annotations

import numpy as np
import pytest

from pydft_qmmm.utils import BOHR_PER_ANGSTROM
from pydft_qmmm.utils import KJMOL_PER_EH


def test_pyscf_is_bundled():
    from pydft_qmmm.interfaces import interfaces
    from pydft_qmmm.utils import TheoryLevel
    assert interfaces["pyscf-mol"][0] is TheoryLevel.QM
    assert "pyscf" not in interfaces


def test_missing_pyscf_does_not_break_package(monkeypatch):
    import pydft_qmmm.interfaces.interface_manager as manager
    real_import = manager.importlib.import_module

    def missing(name):
        if name.endswith(".pyscf"):
            raise ImportError("test missing pyscf")
        return real_import(name)

    monkeypatch.setattr(manager.importlib, "import_module", missing)
    assert manager._load_bundled("pyscf") is None
    assert manager.UNAVAILABLE_INTERFACES["pyscf"] == "test missing pyscf"


# ---------------------------------------------------------------------
# Molecular RKS / UKS lifecycle
# ---------------------------------------------------------------------


def _factory(system, **kwargs):
    from pydft_qmmm.interfaces.pyscf.pyscf_factory import (
        pyscf_interface_factory,
    )
    options = dict(
        basis="sto-3g",
        functional="PBE",
        charge=0,
        multiplicity=1,
    )
    options.update(kwargs)
    return pyscf_interface_factory(system, **options)


def _direct_method(system, spin=0, charge=0, functional="PBE",
                   basis="sto-3g", grid_level=3, conv_tol=1e-9):
    """Build the equivalent bare PySCF solver for the QM subsystem."""
    from pyscf import dft, gto
    from pydft_qmmm.utils import Subsystem
    qm = [
        i for i, s in enumerate(system.subsystems)
        if s is Subsystem.I
    ]
    mol = gto.M(
        atom=[
            (str(system.elements[i]), tuple(system.positions[i]))
            for i in qm
        ],
        unit="Angstrom",
        basis=basis,
        charge=charge,
        spin=spin,
        verbose=0,
    )
    method = dft.RKS(mol, xc=functional) if spin == 0 else dft.UKS(
        mol, xc=functional,
    )
    method.conv_tol = conv_tol
    method.grids.level = grid_level
    return qm, method


def test_singlet_builds_rks(pyscf_water_system):
    potential = _factory(pyscf_water_system)
    potential.compute_energy()
    assert potential.method[0].__class__.__name__ == "RKS"


def test_triplet_builds_uks(pyscf_triplet_system):
    potential = _factory(pyscf_triplet_system, multiplicity=3)
    potential.compute_energy()
    assert potential.method[0].__class__.__name__ == "UKS"


def test_unconverged_scf_raises(pyscf_water_system):
    potential = _factory(pyscf_water_system, max_cycle=0)
    with pytest.raises(RuntimeError, match="did not converge"):
        potential.compute_energy()


def test_incompatible_multiplicity_raises(pyscf_water_system):
    potential = _factory(pyscf_water_system, multiplicity=2)
    with pytest.raises(ValueError, match="incompatible with multiplicity"):
        potential.compute_energy()


def test_molecular_energy_matches_direct_pyscf(pyscf_water_system):
    potential = _factory(pyscf_water_system)
    _, method = _direct_method(pyscf_water_system)
    assert potential.compute_energy() / KJMOL_PER_EH == pytest.approx(
        method.kernel(), abs=1e-8,
    )


def test_molecular_uks_energy_matches_direct_pyscf(pyscf_triplet_system):
    potential = _factory(pyscf_triplet_system, multiplicity=3)
    _, method = _direct_method(pyscf_triplet_system, spin=2)
    assert potential.compute_energy() / KJMOL_PER_EH == pytest.approx(
        method.kernel(), abs=1e-8,
    )


def test_molecular_forces_match_direct_pyscf(pyscf_water_system):
    potential = _factory(pyscf_water_system)
    forces = potential.compute_forces()
    qm, method = _direct_method(pyscf_water_system)
    method.kernel()
    expected = -method.nuc_grad_method().kernel()
    assert forces[qm] / (KJMOL_PER_EH * BOHR_PER_ANGSTROM) == pytest.approx(
        expected, abs=1e-7,
    )
    others = [i for i in range(len(pyscf_water_system)) if i not in qm]
    assert np.allclose(forces[others], 0)


def test_molecular_energy_is_cached(pyscf_water_system):
    potential = _factory(pyscf_water_system)
    potential.compute_energy()
    first = potential.method[0]
    potential.compute_energy()
    assert potential.method[0] is first


def test_moving_an_atom_invalidates_the_cache(pyscf_water_system):
    potential = _factory(pyscf_water_system)
    energy = potential.compute_energy()
    first = potential.method[0]
    pyscf_water_system.positions[1, 0] += 0.05
    moved = potential.compute_energy()
    assert potential.method[0] is not first
    assert moved != energy


def test_density_guess_is_reused(pyscf_water_system):
    potential = _factory(pyscf_water_system)
    potential.compute_energy()
    guess = potential.density_guess[0]
    assert guess is not None
    # The stored matrix is handed straight back for a compatible solver.
    assert potential._density_guess(potential.method[0].mol) is guess


def test_density_guess_is_discarded_on_shape_change(pyscf_water_system):
    potential = _factory(pyscf_water_system)
    potential.compute_energy()
    mol = potential.method[0].mol
    potential.density_guess[0] = np.zeros((2, 2))
    assert potential._density_guess(mol) is None
    # A stale guess must not break the next evaluation either.
    pyscf_water_system.positions[1, 0] += 0.01
    potential.compute_energy()
    assert potential.density_guess[0].shape[-1] == mol.nao


def test_restricted_guess_is_rejected_by_an_open_shell_solver(
        pyscf_water_system,
        pyscf_triplet_system,
):
    restricted = _factory(pyscf_water_system)
    restricted.compute_energy()
    unrestricted = _factory(pyscf_triplet_system, multiplicity=3)
    unrestricted.compute_energy()
    # A spin-summed matrix has the wrong rank for UKS, whatever its
    # AO dimension.
    unrestricted.density_guess[0] = restricted.density_guess[0]
    assert unrestricted._density_guess(unrestricted.method[0].mol) is None


# ---------------------------------------------------------------------
# Finite point-charge embedding
# ---------------------------------------------------------------------


def test_finite_embedding_matches_pyscf_qmmm(pyscf_embedded_water):
    potential, direct_method, qm_indices, mm_indices = pyscf_embedded_water
    result_energy = potential.compute_energy() / KJMOL_PER_EH
    result_forces = potential.compute_forces() / (
        KJMOL_PER_EH * BOHR_PER_ANGSTROM
    )
    direct_energy = direct_method.kernel()
    grad = direct_method.nuc_grad_method()
    direct_qm = -grad.kernel()
    direct_mm = -(
        grad.grad_hcore_mm(direct_method.make_rdm1()) + grad.grad_nuc_mm()
    )
    assert result_energy == pytest.approx(direct_energy, abs=1e-9)
    assert result_forces[qm_indices] == pytest.approx(direct_qm, abs=1e-7)
    assert result_forces[mm_indices] == pytest.approx(direct_mm, abs=1e-7)


def test_finite_embedding_changes_the_energy(pyscf_embedded_water):
    potential, _, _, mm_indices = pyscf_embedded_water
    embedded = potential.compute_energy()
    # Zeroing the environment charges must move the energy, or the test
    # above would be comparing two calculations that both embed nothing.
    potential.system.charges[mm_indices] = 0.0
    assert potential.compute_energy() != pytest.approx(embedded, abs=1e-6)


def test_finite_gradient_matches_central_differences(pyscf_embedded_water):
    potential, _, qm_indices, mm_indices = pyscf_embedded_water
    forces = potential.compute_forces() / (KJMOL_PER_EH * BOHR_PER_ANGSTROM)
    step = 0.002
    for atom in (qm_indices[0], mm_indices[0]):
        for axis in range(3):
            potential.system.positions[atom, axis] += step
            plus = potential.compute_energy() / KJMOL_PER_EH
            potential.system.positions[atom, axis] -= 2 * step
            minus = potential.compute_energy() / KJMOL_PER_EH
            potential.system.positions[atom, axis] += step
            gradient = (plus - minus) / (2 * step * BOHR_PER_ANGSTROM)
            assert forces[atom, axis] == pytest.approx(-gradient, abs=1e-4)


def test_mechanical_coupling_disables_finite_embedding(pyscf_embedded_water):
    potential, _, _, mm_indices = pyscf_embedded_water
    embedded = potential.compute_energy()
    potential.configure_electrostatic_embedding(False)
    bare = potential.compute_energy()
    assert bare != pytest.approx(embedded, abs=1e-6)
    potential.system.charges[mm_indices] = 0.0
    assert potential.compute_energy() == pytest.approx(bare, abs=1e-9)


# ---------------------------------------------------------------------
# Effective core potentials
# ---------------------------------------------------------------------


def test_ecp_basis_without_an_ecp_is_refused(pyscf_iodide_system):
    """A valence basis with no ECP would converge to a wrong answer.

    PySCF keeps ``basis`` and ``ecp`` independent, so asking for
    def2-SVP alone puts all 54 of iodide's electrons into a 26-function
    valence basis and the SCF still converges.  Psi4 applies the ECP
    automatically, so accepting this silently would make the two engines
    disagree by hundreds of Hartree with nothing to show for it.
    """
    potential = _factory(
        pyscf_iodide_system, basis="def2-svp", charge=-1, multiplicity=1,
    )
    with pytest.raises(ValueError, match="effective core potential"):
        potential.compute_energy()


def test_ecp_is_applied_when_requested(pyscf_iodide_system):
    potential = _factory(
        pyscf_iodide_system,
        basis="def2-svp",
        ecp="def2-svp",
        charge=-1,
        multiplicity=1,
    )
    potential.compute_energy()
    mol = potential.method[0].mol
    assert mol.has_ecp()
    # 53 protons, one extra electron, 28 replaced by the ECP.
    assert mol.nelectron == 26


def test_ecp_energy_matches_direct_pyscf(pyscf_iodide_system):
    from pyscf import dft, gto
    potential = _factory(
        pyscf_iodide_system,
        basis="def2-svp",
        ecp="def2-svp",
        charge=-1,
        multiplicity=1,
    )
    mol = gto.M(
        atom=[("I", (6.0, 6.0, 6.0))],
        unit="Angstrom",
        basis="def2-svp",
        ecp="def2-svp",
        charge=-1,
        spin=0,
        verbose=0,
    )
    method = dft.RKS(mol, xc="PBE")
    method.conv_tol = 1e-9
    method.grids.level = 3
    assert potential.compute_energy() / KJMOL_PER_EH == pytest.approx(
        method.kernel(), abs=1e-8,
    )


def test_all_electron_basis_needs_no_ecp(pyscf_water_system):
    """The guard must not fire for a basis that defines no ECP."""
    potential = _factory(pyscf_water_system, basis="def2-svp")
    potential.compute_energy()
    assert not potential.method[0].mol.has_ecp()


def test_ecp_core_electrons_are_excluded_from_the_spin_check():
    from pydft_qmmm.interfaces.pyscf.pyscf_utils import core_electrons
    from pydft_qmmm.interfaces.pyscf.pyscf_utils import validate_spin
    core = core_electrons("def2-svp", ["I"], "def2-svp")
    assert core == 28
    electrons, spin = validate_spin(["I"], -1, 1, core)
    assert (electrons, spin) == (26, 0)
    # Without the ECP the count is the all-electron one.
    assert validate_spin(["I"], -1, 1, 0)[0] == 54


# ---------------------------------------------------------------------
# Method selection and density fitting
# ---------------------------------------------------------------------


def _hf_factory(system, **kwargs):
    """Build a potential with no functional, as Hartree-Fock needs."""
    from pydft_qmmm.interfaces.pyscf.pyscf_factory import (
        pyscf_interface_factory,
    )
    options = dict(basis="sto-3g", charge=0, multiplicity=1)
    options.update(kwargs)
    return pyscf_interface_factory(system, **options)


@pytest.mark.parametrize(
    "method,expected",
    [("rhf", "RHF"), ("rks", "RKS")],
)
def test_closed_shell_method_selection(pyscf_water_system, method, expected):
    kwargs = {"functional": "PBE"} if method == "rks" else {}
    potential = _hf_factory(pyscf_water_system, method=method, **kwargs)
    potential.compute_energy()
    assert type(potential.method[0]).__name__ == expected


@pytest.mark.parametrize(
    "method,expected",
    [("uhf", "UHF"), ("rohf", "ROHF"), ("uks", "UKS"), ("roks", "ROKS")],
)
def test_open_shell_method_selection(pyscf_triplet_system, method, expected):
    kwargs = {"functional": "PBE"} if method.endswith("ks") else {}
    potential = _hf_factory(
        pyscf_triplet_system, charge=0, multiplicity=3, method=method,
        **kwargs,
    )
    potential.compute_energy()
    assert type(potential.method[0]).__name__ == expected


def test_method_defaults_to_hartree_fock_without_a_functional(
        pyscf_water_system,
):
    potential = _hf_factory(pyscf_water_system)
    potential.compute_energy()
    assert type(potential.method[0]).__name__ == "RHF"


def test_method_defaults_to_kohn_sham_with_a_functional(pyscf_water_system):
    potential = _factory(pyscf_water_system)
    potential.compute_energy()
    assert type(potential.method[0]).__name__ == "RKS"


def test_open_shell_default_is_unrestricted(pyscf_triplet_system):
    potential = _hf_factory(
        pyscf_triplet_system, charge=0, multiplicity=3,
    )
    potential.compute_energy()
    assert type(potential.method[0]).__name__ == "UHF"


def test_hartree_fock_energy_matches_direct_pyscf(pyscf_water_system):
    from pyscf import gto, scf
    potential = _hf_factory(pyscf_water_system, method="rhf", conv_tol=1e-11)
    mol = gto.M(
        atom=[
            (str(pyscf_water_system.elements[i]),
             tuple(pyscf_water_system.positions[i]))
            for i in range(3)
        ],
        unit="Angstrom", basis="sto-3g", verbose=0,
    )
    method = scf.RHF(mol)
    method.conv_tol = 1e-11
    assert potential.compute_energy() / KJMOL_PER_EH == pytest.approx(
        method.kernel(), abs=1e-9,
    )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"method": "rks"}, "needs a functional"),
        ({"method": "rhf", "functional": "PBE"}, "takes no functional"),
        ({"method": "nonsense"}, "unknown method"),
        ({"auxbasis": "def2-universal-jfit"}, "auxbasis only applies"),
        ({"device": "tpu"}, "unknown device"),
    ],
)
def test_invalid_method_options_are_refused(
        pyscf_water_system, kwargs, match,
):
    with pytest.raises(ValueError, match=match):
        _hf_factory(pyscf_water_system, **kwargs)


def test_restricted_method_is_refused_for_an_open_shell(
        pyscf_triplet_system,
):
    with pytest.raises(ValueError, match="restricted to a closed shell"):
        _hf_factory(
            pyscf_triplet_system, charge=0, multiplicity=3, method="rhf",
        )


def test_density_fitting_changes_the_solver_and_the_energy(
        pyscf_water_system,
):
    exact = _factory(pyscf_water_system, conv_tol=1e-11)
    fitted = _factory(pyscf_water_system, conv_tol=1e-11, density_fit=True)
    exact_energy = exact.compute_energy()
    fitted_energy = fitted.compute_energy()
    assert "DF" in type(fitted.method[0]).__name__
    # The fitting error is small but has to be real, or the option is
    # not reaching the solver.
    difference = abs(fitted_energy - exact_energy) / KJMOL_PER_EH
    assert 1e-9 < difference < 1e-2


def test_density_fitted_forces_match_central_differences(
        pyscf_embedded_water,
):
    """Fitting must not break the analytic gradient."""
    from pydft_qmmm.interfaces.pyscf.pyscf_factory import (
        pyscf_interface_factory,
    )
    system = pyscf_embedded_water[0].system
    potential = pyscf_interface_factory(
        system, basis="sto-3g", functional="PBE", charge=0, multiplicity=1,
        conv_tol=1e-11, density_fit=True,
    )
    forces = potential.compute_forces() / (KJMOL_PER_EH * BOHR_PER_ANGSTROM)
    step = 0.002
    for atom in (0, 3):
        for axis in range(3):
            system.positions[atom, axis] += step
            plus = potential.compute_energy() / KJMOL_PER_EH
            system.positions[atom, axis] -= 2 * step
            minus = potential.compute_energy() / KJMOL_PER_EH
            system.positions[atom, axis] += step
            gradient = (plus - minus) / (2 * step * BOHR_PER_ANGSTROM)
            assert forces[atom, axis] == pytest.approx(-gradient, abs=1e-4)


def test_hartree_fock_builds_its_own_quadrature_grid(pyscf_pme_system):
    """An HF solver carries no grid, so the embedding has to make one."""
    from pydft_qmmm.potentials.pme_potential import PMEElectronicPotential
    potential = _hf_factory(
        pyscf_pme_system, method="rhf", conv_tol=1e-11, grid_level=4,
    )
    potential.add_electronic_potential(
        PMEElectronicPotential(pyscf_pme_system, 0.4, (20, 20, 20), 6),
    )
    potential.compute_energy()
    assert not hasattr(potential.method[0], "grids")
    quadrature = potential._run().quadrature
    assert len(quadrature.weights) > 1000
    assert abs(potential.pme_energy) > 1e-8


# ---------------------------------------------------------------------
# GPU4PySCF
# ---------------------------------------------------------------------


def _gpu_available() -> bool:
    """Whether a GPU and GPU4PySCF are both actually usable here."""
    try:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() < 1:
            return False
        import gpu4pyscf  # noqa: F401
    except Exception:
        return False
    return True


requires_gpu = pytest.mark.skipif(
    not _gpu_available(),
    reason="needs a GPU and GPU4PySCF",
)

# GPU4PySCF builds the same integrals with different algorithms, so the
# two devices agree to SCF precision rather than to the last bit.
GPU_ENERGY_TOLERANCE = 0.01           # kJ/mol
GPU_FORCE_TOLERANCE = 0.05            # kJ/mol/Angstrom


@requires_gpu
def test_gpu_matches_cpu_for_a_molecular_energy(pyscf_water_system):
    cpu = _factory(pyscf_water_system, conv_tol=1e-11)
    gpu = _factory(pyscf_water_system, conv_tol=1e-11, device="gpu")
    assert gpu.compute_energy() == pytest.approx(
        cpu.compute_energy(), abs=GPU_ENERGY_TOLERANCE,
    )


@requires_gpu
def test_gpu_selects_the_gpu4pyscf_solver(pyscf_water_system):
    gpu = _factory(pyscf_water_system, device="gpu")
    gpu.compute_energy()
    assert type(gpu.method[0]).__module__.startswith("gpu4pyscf")


@requires_gpu
@pytest.mark.parametrize("method", ["rhf", "uhf"])
def test_gpu_supports_method_selection(pyscf_water_system, method):
    system = pyscf_water_system
    multiplicity = 1 if method == "rhf" else 1
    cpu = _hf_factory(
        system, method=method, multiplicity=multiplicity, conv_tol=1e-11,
    )
    gpu = _hf_factory(
        system, method=method, multiplicity=multiplicity, conv_tol=1e-11,
        device="gpu",
    )
    assert gpu.compute_energy() == pytest.approx(
        cpu.compute_energy(), abs=GPU_ENERGY_TOLERANCE,
    )


@requires_gpu
def test_gpu_supports_density_fitting(pyscf_water_system):
    cpu = _factory(pyscf_water_system, conv_tol=1e-11, density_fit=True)
    gpu = _factory(
        pyscf_water_system, conv_tol=1e-11, density_fit=True, device="gpu",
    )
    assert gpu.compute_energy() == pytest.approx(
        cpu.compute_energy(), abs=GPU_ENERGY_TOLERANCE,
    )


@requires_gpu
def test_gpu_matches_cpu_for_finite_embedding(pyscf_embedded_water):
    """The environment charges have to reach the GPU solver too."""
    from pydft_qmmm.interfaces.pyscf.pyscf_factory import (
        pyscf_interface_factory,
    )
    system = pyscf_embedded_water[0].system
    options = dict(
        basis="sto-3g", functional="PBE", charge=0, multiplicity=1,
        conv_tol=1e-11,
    )
    cpu = pyscf_interface_factory(system, **options)
    gpu = pyscf_interface_factory(system, device="gpu", **options)
    assert gpu.compute_energy() == pytest.approx(
        cpu.compute_energy(), abs=GPU_ENERGY_TOLERANCE,
    )
    assert gpu.compute_forces() == pytest.approx(
        cpu.compute_forces(), abs=GPU_FORCE_TOLERANCE,
    )


@requires_gpu
def test_gpu_matches_cpu_for_pme_embedding(pyscf_pme_system):
    """The reciprocal operator is built on the host and used on the device."""
    from pydft_qmmm.potentials.pme_potential import PMEElectronicPotential
    from pydft_qmmm.interfaces.pyscf.pyscf_factory import (
        pyscf_interface_factory,
    )
    options = dict(
        basis="sto-3g", functional="PBE", charge=0, multiplicity=1,
        conv_tol=1e-11, grid_level=4,
    )
    potentials = []
    for device in ("cpu", "gpu"):
        potential = pyscf_interface_factory(
            pyscf_pme_system, device=device, **options
        )
        potential.add_electronic_potential(
            PMEElectronicPotential(pyscf_pme_system, 0.4, (20, 20, 20), 6),
        )
        potentials.append(potential)
    cpu, gpu = potentials
    assert gpu.compute_energy() == pytest.approx(
        cpu.compute_energy(), abs=GPU_ENERGY_TOLERANCE,
    )
    assert gpu.pme_energy == pytest.approx(cpu.pme_energy, abs=1e-6)
    assert gpu.compute_forces() == pytest.approx(
        cpu.compute_forces(), abs=GPU_FORCE_TOLERANCE,
    )
