"""The PySCF interface and potential."""
from __future__ import annotations

__all__ = ["PySCFInterface", "PySCFPotential"]

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import TYPE_CHECKING

import numpy as np
from pyscf import gto

from pydft_qmmm.interfaces import ElectrostaticCouplingMode
from pydft_qmmm.interfaces import QMInterface
from pydft_qmmm.potentials import AtomicPotential
from pydft_qmmm.utils import BOHR_PER_ANGSTROM
from pydft_qmmm.utils import KJMOL_PER_EH
from pydft_qmmm.utils import system_cache

from .pyscf_backend import build_solver
from .pyscf_backend import to_like
from .pyscf_backend import to_numpy
from .pyscf_backend import load_backend
from .pyscf_backend import resolve_method
from .pyscf_embedding import add_finite_embedding
from .pyscf_embedding import build_quadrature_grid
from .pyscf_embedding import finite_embedding_forces
from .pyscf_embedding import pme_ao_operator
from .pyscf_embedding import pme_qm_forces_and_density
from .pyscf_utils import core_electrons
from .pyscf_utils import embedding_atom_indices
from .pyscf_utils import qm_atom_indices
from .pyscf_utils import spin_sum
from .pyscf_utils import validate_spin

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pydft_qmmm.potentials import ElectronicPotential
    from pydft_qmmm import System  # noqa: F401


@dataclass(frozen=True)
class _Quadrature:
    r"""A quadrature grid and its sampled operator.

    Attributes:
        coordinates: The quadrature coordinates
            (:math:`\mathrm{\mathring{A}}`).
        weights: The quadrature weights (:math:`\mathrm{a_0^3}`).
        potential: The potential energy of one electron
            (:math:`\mathrm{E_h}`) at each quadrature point, summed
            over every registered electronic potential.
        matrix: The one-electron operator (:math:`\mathrm{E_h}`) added
            to the core Hamiltonian.
    """
    coordinates: NDArray[np.float64]
    weights: NDArray[np.float64]
    potential: NDArray[np.float64]
    matrix: NDArray[np.float64]


@dataclass(frozen=True)
class _SCFState:
    r"""A converged solver and its atom mapping.

    Attributes:
        qm_indices: The original system indices of subsystem I, in the
            order they appear in the PySCF molecule.
        embed_indices: The original system indices of the embedded
            finite point charges, in the order they were embedded.
        mol: The molecular PySCF object for subsystem I.
        method: The converged RKS or UKS solver.
        dm: The converged density matrix.
        quadrature: The embedding quadrature, or None when no
            electronic potential is registered.
        finite_energy: The finite point-charge embedding energy
            (:math:`\mathrm{E_h}`), electronic plus nuclear.
    """
    qm_indices: tuple[int, ...]
    embed_indices: tuple[int, ...]
    mol: gto.Mole
    method: Any
    dm: NDArray[np.float64]
    quadrature: _Quadrature | None
    finite_energy: float


@dataclass(frozen=True)
class PySCFInterface(QMInterface):
    r"""Store and manipulate PySCF data types.

    Args:
        system: The system that will inform the interface to the
            external software.
        basis: The name of the basis set to use in QM calculations.
        ecp: The effective core potential, or None for an all-electron
            calculation. PySCF does not infer it from the basis.
        functional: The name of the functional to use in QM
            calculations.
        charge: The net charge (:math:`e`) of the QM subsystem.
        multiplicity: The spin multiplicity of the QM subsystem.
        output_file: The file to which PySCF output is written, or None
            to write to standard output.
        output_interval: The interval at which PySCF output should be
            written, e.g., the default value of 1 means that output
            will be written every calculation.
        conv_tol: The SCF energy convergence threshold
            (:math:`\mathrm{E_h}`).
        max_cycle: The maximum number of SCF iterations.
        grid_level: The PySCF quadrature level, used for both the
            exchange-correlation and the embedding integrals.
        verbose: The PySCF logging verbosity.
        options: Additional attributes to set on the PySCF solver.

    Attributes:
        potentials: A list of electronic potentials to incorporate into
            QM calculations.
        method: A single-element list holding the most recently
            converged PySCF solver.
        density_guess: The last compatible density matrix for SCF reuse.
        frame: The estimated current frame for output writing purposes.
        finite_embedding: Whether subsystem II enters the QM Hamiltonian.
    """
    basis: str
    ecp: Any
    functional: str | None
    method_name: str | None
    density_fit: bool
    auxbasis: str | None
    device: str
    charge: int
    multiplicity: int
    output_file: str | None
    output_interval: int
    conv_tol: float
    max_cycle: int
    grid_level: int
    verbose: int
    options: dict[str, Any] = field(default_factory=dict)
    potentials: list[ElectronicPotential] = field(
        default_factory=list,
        init=False,
    )
    method: list[Any] = field(
        default_factory=lambda: [None],
        init=False,
    )
    density_guess: list[Any] = field(
        default_factory=lambda: [None],
        init=False,
    )
    frame: list[int] = field(
        default_factory=lambda: [0],
        init=False,
    )
    backend: list[Any] = field(
        default_factory=lambda: [None],
        init=False,
    )
    stream: list[Any] = field(
        default_factory=lambda: [None],
        init=False,
    )
    finite_embedding: list[bool] = field(
        default_factory=lambda: [True],
        init=False,
    )

    def electrostatic_coupling_mode(self) -> ElectrostaticCouplingMode:
        """Use the shared molecular electrostatic coupling."""
        return ElectrostaticCouplingMode.MOLECULAR

    def configure_electrostatic_embedding(self, enabled: bool) -> None:
        """Match finite embedding to the QM/MM coupling.

        Disable it when OpenMM handles subsystem I-II electrostatics.

        Args:
            enabled: Whether any QM/MM electrostatic interaction is
                assigned to the QM level of theory.
        """
        self.finite_embedding[0] = enabled
        self._clear_cache()

    def add_electronic_potential(self, potential: ElectronicPotential) -> None:
        """Add an electronic potential to apply before calculations.

        Args:
            potential: The electronic potential to incorporate into
                QM calculations.
        """
        self.potentials.append(potential)
        self._clear_cache()

    def _clear_cache(self) -> None:
        """Invalidate cached calculations after configuration changes."""
        for name in list(vars(self)):
            if name.startswith("__wrapped_"):
                getattr(self, name).cache_clear()

    def _stdout(self) -> Any:
        """Get the shared PySCF output stream.

        Open the stream once to prevent PySCF from truncating each frame.

        Returns:
            The open output stream, or None to use standard output.
        """
        if self.output_file is None:
            return None
        if self.stream[0] is None:
            self.stream[0] = open(self.output_file, "a")
        return self.stream[0]

    @system_cache("positions", "elements", "subsystems")
    def _build_molecule(self) -> tuple[tuple[int, ...], gto.Mole]:
        """Build the molecular PySCF object for subsystem I.

        Returns:
            The original system indices of subsystem I and the PySCF
            molecule built from them, in the same order.

        Raises:
            ValueError: If subsystem I is empty, or if its electron
                count cannot realize the requested multiplicity.
        """
        qm_indices = qm_atom_indices(self.system)
        if not qm_indices:
            raise ValueError(
                "the PySCF interface requires a non-empty subsystem I",
            )
        elements = [str(self.system.elements[i]) for i in qm_indices]
        core = core_electrons(self.basis, elements, self.ecp)
        _, spin = validate_spin(
            elements, self.charge, self.multiplicity, core,
        )
        mol = gto.M(
            atom=[
                (element, tuple(self.system.positions[index]))
                for element, index in zip(elements, qm_indices)
            ],
            unit="Angstrom",
            basis=self.basis,
            ecp=self.ecp,
            charge=self.charge,
            spin=spin,
            verbose=self.verbose,
        )
        stream = self._stdout()
        if stream is not None:
            mol.stdout = stream
        return qm_indices, mol

    def nuclear_charges(self) -> NDArray[np.float64]:
        r"""Get the effective nuclear charges used by PySCF.

        ``Mole.atom_charge`` includes effective core potentials.

        Returns:
            The nuclear charges (:math:`e`) of the Subsystem I atoms,
            ordered by ascending system index.
        """
        _, mol = self._build_molecule()
        return np.array(
            [mol.atom_charge(i) for i in range(mol.natm)], dtype=float,
        )

    def _density_guess(self, mol: gto.Mole) -> NDArray[np.float64] | None:
        """Get a shape-compatible density matrix for SCF reuse.

        Args:
            mol: The molecule the guess would be handed to.

        Returns:
            The stored density matrix, or None if it is incompatible.
        """
        guess = self.density_guess[0]
        if guess is None:
            return None
        shape = np.shape(guess)
        expected_dim = 2 if mol.spin == 0 else 3
        if len(shape) != expected_dim:
            return None
        if shape[-2:] != (mol.nao, mol.nao):
            return None
        return guess

    def _build_method(self, mol: gto.Mole) -> Any:
        """Create the solver for a molecule.

        Args:
            mol: The molecule the solver will act on.

        Returns:
            The configured, unconverged PySCF solver.
        """
        backend = load_backend(self.device)
        name = resolve_method(self.method_name, self.functional, mol.spin)
        method = build_solver(backend, name, mol, self.functional)
        self.backend[0] = backend
        if self.density_fit:
            # Density fitting has to be applied before the environment
            # charges are, because decorating the solver replaces its
            # class and the fitted J/K build has to sit underneath that.
            method = method.density_fit(auxbasis=self.auxbasis)
        method.conv_tol = self.conv_tol
        method.max_cycle = self.max_cycle
        if hasattr(method, "grids"):
            method.grids.level = self.grid_level
        skipped = bool(self.frame[0] % self.output_interval)
        if self.output_file is not None and skipped:
            method.verbose = 0
        for key, value in self.options.items():
            setattr(method, key, value)
        return method

    @system_cache("positions", "charges", "elements", "subsystems", "box")
    def _run(self) -> _SCFState:
        """Converge the SCF for the current state of the system.

        Returns:
            The converged solver together with the atom mapping it was
            built from.

        Raises:
            RuntimeError: If the SCF does not converge.
        """
        qm_indices, mol = self._build_molecule()
        method = self._build_method(mol)
        bare_hcore = method.get_hcore()
        bare_nuc = mol.energy_nuc()
        embed_indices: tuple[int, ...] = ()
        if self.finite_embedding[0]:
            embed_indices = embedding_atom_indices(self.system)
        method = add_finite_embedding(
            self.backend[0], method, self.system, embed_indices,
        )
        finite_operator = method.get_hcore() - bare_hcore
        finite_nuc = method.energy_nuc() - bare_nuc
        quadrature = self._build_quadrature(method)
        # Keep the embedding operator fixed throughout the SCF.
        hcore = bare_hcore + finite_operator
        if quadrature is not None:
            # The embedding operator is built on the host; a GPU solver
            # needs it in device memory to add it to its own core
            # Hamiltonian.
            hcore = hcore + to_like(quadrature.matrix, hcore)
        method.get_hcore = lambda *args, **kwargs: hcore
        method.kernel(dm0=self._density_guess(mol))
        if not method.converged:
            raise RuntimeError(
                f"the PySCF SCF did not converge within {self.max_cycle}"
                " cycles",
            )
        dm = method.make_rdm1()
        finite_energy = float(
            np.einsum(
                "ij,ji->",
                to_numpy(spin_sum(dm)),
                to_numpy(finite_operator),
            ) + finite_nuc,
        )
        self.density_guess[0] = dm
        self.method[0] = method
        self.frame[0] += 1
        return _SCFState(
            qm_indices, embed_indices, mol, method, dm, quadrature,
            finite_energy,
        )

    def _build_quadrature(self, method: Any) -> _Quadrature | None:
        """Sample the registered electronic potentials for the SCF.

        Args:
            method: The solver whose grid and AOs the potential is
                contracted against.

        Returns:
            The quadrature and its one-electron operator, or None when
            no electronic potential is registered.
        """
        if not self.potentials:
            return None
        coordinates, weights = build_quadrature_grid(method, self.grid_level)
        potential = np.zeros(len(coordinates))
        for electronic in self.potentials:
            potential += np.asarray(
                electronic.compute_potential(coordinates),
            ).reshape(-1)
        matrix = pme_ao_operator(
            method.mol, potential, coordinates, weights,
        )
        return _Quadrature(coordinates, weights, potential, matrix)

    @system_cache("positions", "charges", "elements", "subsystems", "box")
    def _compute_forces(self) -> NDArray[np.float64]:
        r"""Assemble full-system forces from the converged solver.

        Returns:
            The forces
            (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`) acting
            on atoms in the system.
        """
        state = self._run()
        forces = finite_embedding_forces(
            state.method,
            state.dm,
            state.qm_indices,
            state.embed_indices,
            len(self.system.positions),
        )
        source_charges = None
        if state.quadrature is not None:
            qm_forces, density = pme_qm_forces_and_density(
                state.mol,
                state.dm,
                state.quadrature.potential,
                state.quadrature.coordinates,
                state.quadrature.weights,
            )
            forces[list(state.qm_indices)] += qm_forces
            # Represent the electron density as negative grid charges.
            source_charges = -density * state.quadrature.weights
        forces = forces * KJMOL_PER_EH * BOHR_PER_ANGSTROM
        if source_charges is not None:
            for electronic in self.potentials:
                forces += electronic.compute_source_forces(
                    state.quadrature.coordinates, source_charges,
                )
        return forces


class PySCFPotential(PySCFInterface, AtomicPotential):
    """A potential wrapping PySCF."""

    def compute_energy(self) -> float:
        r"""Compute the energy of the system using PySCF.

        Returns:
            The energy (:math:`\mathrm{kJ\;mol^{-1}}`) of the system.
        """
        return float(self._run().method.e_tot) * KJMOL_PER_EH

    def compute_forces(self) -> NDArray[np.float64]:
        r"""Compute the forces on the system using PySCF.

        Returns:
            The forces (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`)
            acting on atoms in the system.
        """
        return self._compute_forces()

    def compute_components(self) -> dict[str, float]:
        r"""Compute the components of energy using PySCF.

        These terms are already included in the total energy.

        Returns:
            The components of the energy (:math:`\mathrm{kJ\;mol^{-1}}`)
            of the system.
        """
        state = self._run()
        return {
            "Finite Embedding": state.finite_energy * KJMOL_PER_EH,
            "Reciprocal Embedding": self.pme_energy * KJMOL_PER_EH,
        }

    @property
    def pme_matrix(self) -> NDArray[np.float64] | None:
        r"""The reciprocal one-electron operator (:math:`\mathrm{E_h}`).
        """
        quadrature = self._run().quadrature
        return None if quadrature is None else quadrature.matrix

    @property
    def pme_energy(self) -> float:
        r"""The reciprocal electronic energy (:math:`\mathrm{E_h}`).
        """
        state = self._run()
        if state.quadrature is None:
            return 0.0
        return float(
            np.einsum(
                "ij,ji->",
                to_numpy(spin_sum(state.dm)),
                state.quadrature.matrix,
            ),
        )
