from __future__ import annotations

import os

import numpy as np
import pytest

from pydft_qmmm import QMMMHamiltonian
from pydft_qmmm.interfaces.sparc import sparc_utils
from pydft_qmmm.interfaces.sparc.sparc_interface import PHI_FILE
from pydft_qmmm.interfaces.sparc.sparc_interface import VEXT_FILE


class TestSparcEmbeddingConfiguration:
    """Configuration guards. No SPARC binary required."""

    def test_pme_exclusion_uses_resolved_override_parameters(
            self, monkeypatch, sparc_qmmm_system, sparc_qm_embedded,
            mm_spce_water4,
    ):
        """Every PME term must use the same explicitly selected mesh."""
        from pydft_qmmm.potentials import pme_potential

        requested = (0.37, (40, 42, 44), 5)
        captured = []
        original = pme_potential.PMEExcludedPotential

        def record(system, alpha, gridnumber, spline_order, **kwargs):
            captured.append((alpha, gridnumber, spline_order))
            assert kwargs["include_forces"] is False
            return original(system, alpha, gridnumber, spline_order, **kwargs)

        monkeypatch.setattr(
            pme_potential, "PMEExcludedPotential", record,
        )
        coupling = QMMMHamiltonian(
            "electrostatic",
            "electrostatic",
            cutoff=2.5,
            pme_alpha=requested[0],
            pme_gridnumber=requested[1],
            pme_spline_order=requested[2],
        )
        total = mm_spce_water4[3:] + sparc_qm_embedded[0:3] + coupling
        total.build_calculator(sparc_qmmm_system)

        assert captured == [requested]

    def test_mechanical_coupling_rejects_manual_embedding(
            self, sparc_embedded,
    ):
        with pytest.raises(ValueError, match="double-count|embedding"):
            sparc_embedded.configure_electrostatic_embedding(False)

    def test_embedding_owns_the_nuclear_term(self, sparc_embedded):
        sparc_embedded.configure_electrostatic_embedding(True)
        assert sparc_embedded.applies_nuclear_potential() is True

    def test_unembedded_does_not_own_the_nuclear_term(self, sparc_plain):
        assert sparc_plain.applies_nuclear_potential() is False

    def test_unembedded_rejects_electronic_potentials(self, sparc_plain):
        with pytest.raises(NotImplementedError, match="embedding"):
            sparc_plain.add_electronic_potential(object())


class TestSparcRunInvalidation:
    """SPARC must re-run when V_ext changes but the QM geometry does not."""

    def test_moving_only_mm_atoms_still_reruns_sparc(
            self, sparc_embedded, monkeypatch,
    ):
        """ASE caches on the QM-only Atoms object.

        numerical_gradient over an MM atom leaves every subsystem I
        coordinate untouched, so ase's check_state() reports no change
        and hands back the previous energy -- while _write_vext has just
        written a different external potential.  Without an explicit
        reset SPARC is never invoked again, the stale energy is returned,
        and the SPARC term of an MM-displacement gradient is identically
        zero.
        """
        order = []

        class StubCalculator:
            def __init__(self):
                self.results = {"energy": 1.0}

            def set(self, **kwargs):
                return {}

            def reset(self):
                self.results = {}
                order.append("reset")

            def get_potential_energy(self, atoms=None):
                order.append(("energy", dict(self.results)))
                return 0.0

            def get_forces(self, atoms=None):
                return np.zeros((len(sparc_embedded.system.select(
                    "subsystem I",
                )), 3))

        # SPARCPotential is a frozen dataclass, so setattr is refused.
        object.__setattr__(sparc_embedded, "calculator", StubCalculator())
        object.__setattr__(sparc_embedded, "_write_vext", lambda: None)
        monkeypatch.setattr(
            sparc_utils, "assert_scf_converged", lambda *a, **k: None,
        )
        monkeypatch.setattr(
            sparc_utils, "assert_embedding_capable", lambda *a, **k: None,
        )

        sparc_embedded._run()

        assert "reset" in order, (
            "_run never reset the ASE calculator, so a V_ext change with "
            "unchanged QM coordinates would return a cached energy"
        )
        assert order.index("reset") < next(
            i for i, e in enumerate(order) if e != "reset"
        ), "the reset must happen before the energy is requested"
        assert order[-1][1] == {}, (
            "the cached ASE results were still present when the energy "
            "was requested"
        )


class TestSparcEmbeddingStalenessGuard:
    """Regression coverage for the frame-vs-vext_step off-by-one.

    ``compute_forces`` -> ``_run`` -> ``_write_vext`` (stamps V_ext with
    ``frame[0]``) -> ... SPARC runs and echoes that step into the PHI
    file ... -> ``_run`` increments ``frame[0]`` (cache miss only) ->
    ``compute_forces`` calls ``_read_mm_forces``.  By the time
    ``_read_mm_forces`` checks the PHI file's step, ``frame[0]`` may
    have moved on from the value that was actually written.  These
    tests pin that exact ordering -- including the increment landing
    strictly between the write and the read -- without needing the
    SPARC binary: SPARC's echo is simulated by hand-writing a PHI file.
    """

    def test_phi_matching_the_written_vext_step_is_accepted(
            self, sparc_embedded,
    ):
        # Mirrors what a cache-MISS _run() does: write V_ext, then
        # (after the elided SPARC call) bump frame before returning.
        sparc_embedded._write_vext()
        sparc_embedded.frame[0] += 1
        vext_path = os.path.join(sparc_embedded.directory, VEXT_FILE)
        # Read back what was actually written rather than assuming it.
        _, cell, written_step = sparc_utils.read_grid_file(vext_path)
        phi = np.zeros(sparc_embedded.fd_grid)
        sparc_utils.write_grid_file(
            os.path.join(sparc_embedded.directory, PHI_FILE),
            phi, cell, written_step,
        )
        # Must not raise: this is SPARC faithfully echoing the step it
        # was given, which is exactly what the pre-fix code rejected.
        sparc_embedded._read_mm_forces()

    def test_phi_matching_the_written_vext_step_is_accepted_on_cache_hit(
            self, sparc_embedded,
    ):
        # Mirrors a cache HIT: _write_vext() is never called again and
        # frame[0] never advances, so vext_step must still describe the
        # PHI file left over from the last real evaluation.
        sparc_embedded._write_vext()
        sparc_embedded.frame[0] += 1
        vext_path = os.path.join(sparc_embedded.directory, VEXT_FILE)
        _, cell, written_step = sparc_utils.read_grid_file(vext_path)
        phi = np.zeros(sparc_embedded.fd_grid)
        sparc_utils.write_grid_file(
            os.path.join(sparc_embedded.directory, PHI_FILE),
            phi, cell, written_step,
        )
        sparc_embedded._read_mm_forces()
        # A second call, with nothing re-written (as on a cache hit),
        # must still succeed.
        sparc_embedded._read_mm_forces()

    def test_phi_left_over_from_a_previous_evaluation_is_rejected(
            self, sparc_embedded,
    ):
        vext_path = os.path.join(sparc_embedded.directory, VEXT_FILE)
        phi_path = os.path.join(sparc_embedded.directory, PHI_FILE)
        phi = np.zeros(sparc_embedded.fd_grid)

        # First evaluation: V_ext written at step 0, PHI answers it.
        sparc_embedded._write_vext()
        sparc_embedded.frame[0] += 1
        _, cell, first_step = sparc_utils.read_grid_file(vext_path)
        sparc_utils.write_grid_file(phi_path, phi, cell, first_step)
        sparc_embedded._read_mm_forces()

        # Second evaluation: geometry moved on, so a new V_ext is
        # written at the new step -- but the PHI file on disk is still
        # the first evaluation's.  This is the actual staleness the
        # guard exists to catch, and must still raise after the fix.
        sparc_embedded._write_vext()
        with pytest.raises(ValueError, match="stale"):
            sparc_embedded._read_mm_forces()


class TestSparcBackReactionSign:
    """The sign of the MM back-reaction, without an SCF.

    The slow physics tests below cannot settle this.  Their finite
    differences sit under a noise floor of roughly
    ``tol_scf * n_qm_atoms / dist`` -- about 1600 kJ/mol/A at
    ``tol_scf=1e-6`` and the 0.005 A central-difference denominator --
    so they cannot distinguish a correct force from an inverted one.

    These tests can, because they remove the SCF entirely.  A potential
    that is exactly linear in one Cartesian direction is a uniform
    field, and the force on a normalised Gaussian charge sitting in a
    uniform field is exactly ``q * E``, independent of the Gaussian
    width, the mesh and the position.  Feeding such a field in through
    the real production path -- ``phi_from_sparc`` (which negates) and
    then ``contract_gaussian_gradient`` (whose three flips are the rest
    of the chain) -- reproduces ``q * E`` to ~1e-8 relative, and drops
    any one of those four sign operations and it comes back inverted.

    What this does NOT verify is SPARC's own convention: that
    ``elecstPotential`` really holds electron potential *energy* in
    Hartree with the sign ``phi_from_sparc`` assumes.  That link is
    established at the C source level and by the constant-field
    validation in the fork's ``QMMM.md``, not here.
    """

    L = 12.0
    E0 = 0.05          # V / Angstrom
    SIGMA = 0.5        # Angstrom
    CHARGE = 0.5       # e

    def _force(self, axis, charge, shape=None, cell=None, position=None):
        """Force on one Gaussian in a uniform field along ``axis``."""
        from pydft_qmmm.embedding.grid_potential import (
            contract_gaussian_gradient,
        )
        from pydft_qmmm.interfaces.sparc.sparc_grid import EV_PER_HARTREE
        from pydft_qmmm.interfaces.sparc.sparc_grid import phi_from_sparc
        cell = np.eye(3) * self.L if cell is None else cell
        shape = (60, 60, 60) if shape is None else shape
        position = (
            np.diag(cell) / 2.0 if position is None else np.asarray(position)
        )
        # Grid coordinate along `axis`, broadcast over the other two.
        length = np.linalg.norm(cell[axis])
        ramp = (np.arange(shape[axis]) * (length / shape[axis])).reshape(
            [-1 if a == axis else 1 for a in range(3)],
        )
        # We want phi_volts = -E0 * r_axis, so that E = -grad phi points
        # along +axis.  SPARC's file holds electron potential ENERGY in
        # Hartree, which is the negative of that over EV_PER_HARTREE --
        # so the array we hand phi_from_sparc is +E0 * r / EV_PER_HARTREE.
        phi_hartree = (self.E0 * ramp / EV_PER_HARTREE) * np.ones(shape)
        return contract_gaussian_gradient(
            phi_from_sparc(phi_hartree),
            position.reshape(1, 3), np.array([charge]),
            shape, cell, self.SIGMA,
        )[0]

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_uniform_field_gives_exactly_q_times_E(self, axis):
        force = self._force(axis, self.CHARGE)
        expected = np.zeros(3)
        expected[axis] = self.CHARGE * self.E0
        # 1e-8 relative is the 6 sigma Gaussian truncation and nothing
        # else: it does not shrink with the mesh, because the field is
        # exactly linear and so carries no discretisation error.
        assert force == pytest.approx(expected, abs=1e-9)

    def test_reversing_the_charge_reverses_the_force(self):
        positive = self._force(0, +self.CHARGE)
        negative = self._force(0, -self.CHARGE)
        assert positive[0] > 0
        assert negative == pytest.approx(-positive, abs=1e-15)

    def test_anisotropic_cell_and_grid_do_not_mix_axes(self):
        """Catches a transposed grid or a swapped lattice vector."""
        cell = np.diag([10.0, 12.0, 14.0])
        force = self._force(
            1, self.CHARGE, shape=(50, 60, 56), cell=cell,
            position=[5.0, 6.0, 7.0],
        )
        assert force == pytest.approx(
            [0.0, self.CHARGE * self.E0, 0.0], abs=1e-9,
        )


@pytest.mark.slow
class TestSparcEmbeddingPhysics:
    """Rungs 4 and 5. Requires the forked SPARC binary."""

    def test_newtons_third_law_across_subsystems(
            self, sparc_qmmm_system, sparc_newton_total,
    ):
        calculator = sparc_newton_total.build_calculator(
            sparc_qmmm_system,
        )
        forces = calculator.calculate().forces
        # At 151**3 (h ~= 0.15 Bohr), the remaining SPARC egg-box
        # residual for this geometry is about 0.050 kJ/mol/A.
        assert forces.sum(axis=0) == pytest.approx(0, abs=6e-2)

    def test_qm_atom_gradient(
            self, sparc_qmmm_system, sparc_embedded_total,
    ):
        from pydft_qmmm.utils import numerical_gradient
        calculator = sparc_embedded_total.build_calculator(
            sparc_qmmm_system,
        )
        analytical = -calculator.calculate().forces[0]
        numerical = numerical_gradient(calculator, {0})
        assert analytical - numerical == pytest.approx(0, abs=2.0)

    def test_mm_atom_gradient(
            self, sparc_qmmm_system, sparc_embedded_total,
    ):
        from pydft_qmmm.utils import numerical_gradient
        calculator = sparc_embedded_total.build_calculator(
            sparc_qmmm_system,
        )
        analytical = -calculator.calculate().forces[3]
        numerical = numerical_gradient(calculator, {3})
        assert analytical - numerical == pytest.approx(0, abs=2.0)


@pytest.mark.slow
class TestSparcPME:

    def test_pme_changes_the_energy(
            self, sparc_qmmm_system, sparc_cutoff_total,
            sparc_pme_total,
    ):
        plain = sparc_cutoff_total.build_calculator(sparc_qmmm_system)
        pme = sparc_pme_total.build_calculator(sparc_qmmm_system)
        assert plain.calculate().energy != pytest.approx(
            pme.calculate().energy, abs=1e-6,
        )

    def test_pme_changes_the_force_on_the_qm_region(
            self, sparc_qmmm_system, sparc_cutoff_total,
            sparc_pme_total,
    ):
        """Subsystem III's reciprocal field must act on subsystem I."""
        plain = sparc_cutoff_total.build_calculator(sparc_qmmm_system)
        pme = sparc_pme_total.build_calculator(sparc_qmmm_system)
        qm = sorted(sparc_qmmm_system.select("subsystem I"))
        plain_qm = plain.calculate().forces[qm]
        pme_qm = pme.calculate().forces[qm]
        assert pme_qm != pytest.approx(plain_qm, abs=1e-3)

    def test_pme_qm_atom_gradient(
            self, sparc_qmmm_system, sparc_pme_total,
    ):
        from pydft_qmmm.utils import numerical_gradient
        calculator = sparc_pme_total.build_calculator(sparc_qmmm_system)
        analytical = -calculator.calculate().forces[0]
        # Engine coupling supplies the QM embedding forces itself.
        # PMEExcluded corrects OpenMM's energy only, so its derivative
        # must not enter this force-mixed gradient comparison.
        numerical = numerical_gradient(
            calculator, {0}, components=["SPARC"],
        )
        assert analytical - numerical == pytest.approx(0, abs=2.0)


class TestSparcCharge:

    def test_integer_charge_sets_net_charge(
            self, spce_qmmm_system, sparc_workdir,
    ):
        from pydft_qmmm.interfaces.sparc.sparc_factory import (
            sparc_interface_factory,
        )
        potential = sparc_interface_factory(
            spce_qmmm_system, charge=-1, directory=sparc_workdir,
            xc="pbe", fd_grid=(48, 48, 48),
        )
        assert potential.calculator.valid_params["NET_CHARGE"] == -1

    def test_zero_charge_omits_net_charge(
            self, spce_qmmm_system, sparc_workdir,
    ):
        from pydft_qmmm.interfaces.sparc.sparc_factory import (
            sparc_interface_factory,
        )
        potential = sparc_interface_factory(
            spce_qmmm_system, charge=0, directory=sparc_workdir,
            xc="pbe", fd_grid=(48, 48, 48),
        )
        assert "NET_CHARGE" not in potential.calculator.valid_params

    def test_fractional_charge_is_rejected(
            self, spce_qmmm_system, sparc_workdir,
    ):
        from pydft_qmmm.interfaces.sparc.sparc_factory import (
            sparc_interface_factory,
        )
        with pytest.raises(ValueError, match="integer"):
            sparc_interface_factory(
                spce_qmmm_system, charge=0.5, directory=sparc_workdir,
                xc="pbe", fd_grid=(48, 48, 48),
            )
