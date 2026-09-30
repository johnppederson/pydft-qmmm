"""The SPARC software interface and potential.

This module wraps the SPARC ASE calculator from SPARC-X-API and exposes
it as a QM potential within PyDFT-QMMM.  Mechanical embedding is always
available; electrostatic embedding of subsystem II point charges is
additionally supported when ``embedding=True`` and the calculator is
built from a SPARC binary forked to accept the QM/MM tags.
"""
from __future__ import annotations

import glob
import os
import warnings
from dataclasses import dataclass
from dataclasses import field
from typing import ClassVar
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms

from pydft_qmmm.embedding.grid_potential import build_external_potential
from pydft_qmmm.embedding.grid_potential import contract_gaussian_gradient
from pydft_qmmm.interfaces import ElectrostaticCouplingMode
from pydft_qmmm.interfaces import QMInterface
from pydft_qmmm.interfaces.engine_embedding import EngineEmbeddingMixin
from pydft_qmmm.potentials import AtomicPotential
from pydft_qmmm.utils import KJMOL_PER_EV
from pydft_qmmm.utils import system_cache

from . import sparc_grid
from . import sparc_utils

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from sparc.calculator import SPARC
    from pydft_qmmm.potentials import ElectronicPotential
    from pydft_qmmm import System  # noqa: F401

VEXT_FILE = "QMMM_VEXT.bin"
PHI_FILE = "QMMM_PHI.bin"
PME_FILE = "PME_DATA"


@dataclass(frozen=True)
class SPARCInterface(EngineEmbeddingMixin, QMInterface):
    r"""A mix-in for storing and manipulating SPARC data types.

    Args:
        system: The system that will inform the interface to SPARC.
        calculator: A configured ``sparc.calculator.SPARC`` instance.
        charge: The net charge (:math:`e`) of the QM subsystem.
        directory: The working directory for SPARC calculations.
        fd_grid: The finite-difference grid, pinned so that the driver
            knows the grid before SPARC runs.
        embedding: Whether to electrostatically embed subsystem II.
        embedding_sigma: The Gaussian width
            (:math:`\mathrm{\mathring{A}}`) representing MM point
            charges on the grid.

    Attributes:
        potentials: Electronic potentials to fold into V_ext.
        frame: The number of calculations performed so far.
        vext_step: The step stamp actually written into the most recent
            V_ext file, so the staleness guard on the PHI file compares
            against what was written rather than against ``frame``,
            which may have advanced (on a cache miss) or not (on a
            cache hit) since that write.
    """
    _embedding_conflict_message: ClassVar[str] = (
        "SPARC embedding=True conflicts with this "
        "QMMMHamiltonian: no QM/MM electrostatic interaction is "
        "assigned to the QM level of theory.  Disable embedding "
        "or select electrostatic coupling to avoid "
        "double-counting electrostatics."
    )
    _embedding_unavailable_message: ClassVar[str] = (
        "PME embedding needs the SPARC QM/MM fork.  Build the "
        "potential with embedding=True and point the command "
        "keyword at a sparc built from the qmmm-embedding "
        "branch."
    )

    calculator: SPARC
    charge: int
    directory: str
    fd_grid: tuple[int, int, int]
    embedding: bool = False
    embedding_sigma: float = 0.3
    potentials: list[ElectronicPotential] = field(
        default_factory=list, init=False,
    )
    frame: list[int] = field(default_factory=lambda: [0], init=False)
    vext_step: list[int | None] = field(
        default_factory=lambda: [None], init=False,
    )

    def electrostatic_coupling_mode(self) -> ElectrostaticCouplingMode:
        """The SPARC embedding fork owns electronic and nuclear coupling."""
        return ElectrostaticCouplingMode.ENGINE

    def applies_nuclear_potential(self) -> bool:
        """The fork couples V_ext to the pseudocharge itself.

        ``energy.c`` adds ``∫b·V_ext`` and ``forces.c`` differentiates
        ``phi + V_ext`` against ``b_J``, so the coupling Hamiltonian
        must not add its own nuclear term on top.

        Returns:
            Whether the nuclear term is already applied.
        """
        return self.embedding

    def _cell_angstrom(self) -> NDArray[np.float64]:
        """
        Returns:
            The lattice vectors as rows
            (:math:`\\mathrm{\\mathring{A}}`).
        """
        # PyDFT-QMMM stores lattice vectors as columns; ASE and SPARC
        # want rows.
        return np.asarray(self.system.box).T

    def _write_pme_data(self) -> str:
        r"""Write region III PME sources and Ewald parameters.

        The Gaussian field already includes region II and its periodic
        images, and SPARC treats region I periodically. Zero their PME
        source charges instead of subtracting single-image exclusions.

        Returns:
            The path written.
        """
        from pydft_qmmm.embedding.pme_grid import write_pme_data
        potential = self.potentials[0]
        charges = np.array(self.system.charges, copy=True)
        charges[sorted(self.system.select("not subsystem III"))] = 0.
        os.makedirs(self.directory, exist_ok=True)
        path = os.path.join(self.directory, PME_FILE)
        if os.path.isfile(path):
            os.remove(path)
        write_pme_data(
            path,
            np.asarray(self.system.positions),
            charges,
            [],
            potential.pme_alpha,
            tuple(potential.pme_gridnumber),
            potential.pme_spline_order,
            self.frame[0],
        )
        return path

    def _build_vext(self) -> NDArray[np.float64]:
        r"""Build the external potential on SPARC's grid.

        The near field comes from subsystem II spread as Gaussians; the
        long-range term, when QM/MM/PME is active, comes from helPME's
        reciprocal sum sourced only by subsystem III, with no exclusion
        subtraction. Both are returned as electron potential ENERGY, the
        convention SPARC's Veff uses.

        Returns:
            The external potential (:math:`\mathrm{eV}`), shaped like
            ``fd_grid``.
        """
        indices = sorted(self.system.select("subsystem II"))
        if not indices and not self.potentials:
            warnings.warn(
                "embedding=True but subsystem II is empty: V_ext will "
                "be identically zero and electrostatic embedding will "
                "have no effect.  Check that a partition plugin (not "
                "partition=None) is assigned to the QMMMHamiltonian, "
                "and that it actually promotes some MM atoms to "
                "subsystem II for this geometry.",
                RuntimeWarning,
                stacklevel=2,
            )
        positions = np.asarray(self.system.positions)[indices]
        charges = np.asarray(self.system.charges)[indices]
        cell = self._cell_angstrom()
        v_ext = build_external_potential(
            positions, charges, self.fd_grid, cell, self.embedding_sigma,
        )
        if self.potentials:
            from pydft_qmmm.embedding.pme_grid import build_pme_potential
            path = self._write_pme_data()
            v_ext = v_ext + build_pme_potential(
                path, self.fd_grid, cell, expect_step=self.frame[0],
            )
        return v_ext

    def _write_vext(self) -> None:
        """Write the external potential for the SPARC fork to read."""
        v_ext = self._build_vext()
        step = self.frame[0]
        sparc_utils.write_grid_file(
            os.path.join(self.directory, VEXT_FILE),
            sparc_grid.vext_to_sparc(v_ext),
            sparc_grid.cell_to_bohr(self._cell_angstrom()),
            step,
        )
        # Recorded here, at write time, rather than derived from
        # ``frame`` at read time: ``frame`` is incremented at the end of
        # ``_run`` on a cache miss and not at all on a cache hit, so by
        # the time ``_read_mm_forces`` runs it no longer necessarily
        # equals the step this V_ext file (and thus the PHI file SPARC
        # echoes it into) actually carries.
        self.vext_step[0] = step

    def _read_mm_forces(self) -> NDArray[np.float64]:
        r"""Contract SPARC's potential against the MM Gaussians.

        Returns:
            An Nx3 array of forces
            (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`) ordered to
            match ``sorted(system.select("subsystem II"))``.

        Raises:
            SparcExecutionError: If SPARC wrote no potential, so the
                back-reaction was never computed and momentum would not
                be conserved.
        """
        path = os.path.join(self.directory, PHI_FILE)
        if not os.path.isfile(path):
            raise sparc_utils.SparcExecutionError(
                self.directory,
                f"{PHI_FILE} is absent, so the QM->MM back-reaction was "
                "never computed and momentum will not be conserved.",
            )
        phi_hartree, _, _ = sparc_utils.read_grid_file(
            path, expect_shape=self.fd_grid, expect_step=self.vext_step[0],
        )
        phi = sparc_grid.phi_from_sparc(phi_hartree)
        indices = sorted(self.system.select("subsystem II"))
        positions = np.asarray(self.system.positions)[indices]
        charges = np.asarray(self.system.charges)[indices]
        forces = contract_gaussian_gradient(
            phi, positions, charges, self.fd_grid,
            self._cell_angstrom(), self.embedding_sigma,
        )
        return forces * KJMOL_PER_EV

    @system_cache("positions", "elements", "subsystems", "box", "charges")
    def _run(self) -> tuple[float, NDArray[np.float64]]:
        r"""Run SPARC on the QM subsystem.

        Returns:
            The energy (:math:`\mathrm{kJ\;mol^{-1}}`) and an Nx3 force
            array (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`) for
            the QM atoms, ordered to match
            ``sorted(system.select("subsystem I"))``.
        """
        os.makedirs(self.directory, exist_ok=True)
        # SPARC renames an existing SPARC.out to SPARC.out_01 rather
        # than overwriting it.  On the second and every later run in
        # one directory -- each finite-difference displacement, each MD
        # step -- assert_scf_converged would then re-read the *first*
        # run's output and pass regardless of what this run did, which
        # is the same fail-open behaviour the guard was added to
        # remove.  Clearing the outputs keeps SPARC.out canonical.
        for stale_path in glob.glob(
                os.path.join(self.directory, "SPARC.out*"),
        ) + glob.glob(os.path.join(self.directory, "SPARC.static*")):
            os.remove(stale_path)
        if self.embedding:
            self._write_vext()
            for stale in (PHI_FILE,):
                stale_path = os.path.join(self.directory, stale)
                if os.path.isfile(stale_path):
                    os.remove(stale_path)
        qm_indices = sorted(self.system.select("subsystem I"))
        symbols = [str(self.system.elements[i]) for i in qm_indices]
        positions = np.asarray(self.system.positions)[qm_indices]
        atoms = Atoms(
            symbols=symbols,
            positions=positions,
            cell=self._cell_angstrom(),
            pbc=True,
        )
        atoms.calc = self.calculator
        if self.embedding:
            self.calculator.set(
                QMMM_FLAG=1,
                QMMM_VEXT_FILE=VEXT_FILE,
                QMMM_PHI_FILE=PHI_FILE,
            )
        # ASE decides whether to re-run from the Atoms object alone, and
        # that object holds only subsystem I.  Move an MM atom and the QM
        # coordinates are unchanged, so check_state() reports no change
        # and the cached energy comes back -- even though _write_vext has
        # just written a different V_ext, and even though system_cache
        # already established that this is a genuine miss.  SPARC would
        # never see the new external potential, making the SPARC term of
        # any MM-displacement gradient identically zero.  _run is only
        # entered on a real miss, so an unconditional reset is right.
        self.calculator.reset()
        energy = atoms.get_potential_energy() * KJMOL_PER_EV
        forces = atoms.get_forces() * KJMOL_PER_EV
        out_path = os.path.join(self.directory, "SPARC.out")
        sparc_utils.assert_scf_converged(out_path, self.directory)
        if self.embedding:
            sparc_utils.assert_embedding_capable(out_path, self.directory)
        self.frame[0] += 1
        return energy, forces


class SPARCPotential(SPARCInterface, AtomicPotential):
    """A potential wrapping SPARC functionality."""

    def compute_energy(self) -> float:
        r"""Compute the energy of the system using SPARC.

        Returns:
            The energy (:math:`\mathrm{kJ\;mol^{-1}}`) of the QM
            subsystem.
        """
        energy, _ = self._run()
        return energy

    def compute_forces(self) -> NDArray[np.float64]:
        r"""Compute the forces on the system using SPARC.

        Returns:
            The forces (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`)
            on every atom, with uncoupled atoms zeroed.
        """
        _, qm_forces = self._run()
        forces = np.zeros(self.system.positions.shape)
        qm_indices = sorted(self.system.select("subsystem I"))
        forces[qm_indices, :] = qm_forces
        if self.embedding:
            # Subsystem III is deliberately left at zero: under direct
            # QM/MM/PME its force from the QM region is the "Y = MM"
            # term, which OpenMM supplies from the static forcefield
            # charges.  See John et al., JCP 161, 034103 (2024), Table I.
            embed_indices = sorted(self.system.select("subsystem II"))
            forces[embed_indices, :] = self._read_mm_forces()
        return forces

    def compute_components(self) -> dict[str, float]:
        r"""Compute the components of energy using SPARC.

        Returns:
            An empty dict; SPARC does not expose sub-components.
        """
        return {}
