"""Periodic PySCF with MM embedding on the solver's FFT grid."""
from __future__ import annotations

__all__ = ["PySCFPBCInterface", "PySCFPBCPotential"]

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from pydft_qmmm.embedding.grid_potential import spectral_value_and_gradient
from pydft_qmmm.interfaces import ElectrostaticCouplingMode
from pydft_qmmm.interfaces import QMInterface
from pydft_qmmm.potentials import AtomicPotential
from pydft_qmmm.utils import BOHR_PER_ANGSTROM
from pydft_qmmm.utils import KJMOL_PER_EH
from pydft_qmmm.utils import system_cache

from ..pyscf.pyscf_backend import load_backend
from ..pyscf.pyscf_backend import load_submodule
from ..pyscf.pyscf_backend import to_like
from ..pyscf.pyscf_backend import to_numpy
from .pbc_cell import build_cell
from .pbc_cell import valence_charges
from .pbc_embedding import ao_operator
from .pbc_embedding import external_potential
from .pbc_embedding import grid_coordinates

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pydft_qmmm.potentials import ElectronicPotential


@dataclass(frozen=True)
class _SCFState:
    """Converged SCF state and its embedding data.

    Atom indices refer to the full system. Coordinates use Å, weights
    Bohr³, k-points inverse Bohr, and potentials/matrices/energies Hartree.
    Keep the forward potential instances for their reaction forces.
    """
    cell: Any
    qm_indices: tuple[int, ...]
    embed_indices: tuple[int, ...]
    kpts: NDArray[np.float64]
    method: Any
    dm: Any
    coords: NDArray[np.float64]
    weights: NDArray[np.float64]
    potential: NDArray[np.float64]
    matrix: NDArray[np.complex128]
    nuclear_energy: float
    potentials: tuple[Any, ...]


def _nuclear_coupling(
        cell: Any,
        potential: NDArray[np.float64],
        box: NDArray[np.float64],
) -> float:
    """Return −Σ Z V_ext(R) in Hartree, using nuclear valence charges.

    Interpolate the electron potential on the FFT grid; box vectors use Å.
    """
    mesh = tuple(int(n) for n in cell.mesh)
    positions = np.asarray(cell.atom_coords()) / BOHR_PER_ANGSTROM
    value, _ = spectral_value_and_gradient(
        potential.reshape(mesh),
        np.asarray(box, dtype=np.float64),
        positions,
    )
    return float(-np.sum(valence_charges(cell) * value))


@dataclass(frozen=True)
class PySCFPBCInterface(QMInterface):
    """Periodic SCF configuration; see pyscf_pbc_interface_factory for arguments.

    potentials holds added electronic fields; density_guess stores the last
    SCF density. frame tracks log writes and embedding enables QM/MM fields.
    """
    basis: str
    pseudo: str
    functional: str | None
    ke_cutoff: float | None
    mesh: tuple[int, int, int] | None
    embedding_sigma: float
    device: str
    charge: int
    multiplicity: int
    output_file: str | None
    output_interval: int
    conv_tol: float
    max_cycle: int
    verbose: int
    options: dict[str, Any] = field(default_factory=dict)
    potentials: list[ElectronicPotential] = field(
        default_factory=list,
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
    embedding: bool = field(default=False, init=False)

    def electrostatic_coupling_mode(self) -> ElectrostaticCouplingMode:
        """Own both electronic and nuclear QM/MM coupling."""
        return ElectrostaticCouplingMode.ENGINE

    def applies_nuclear_potential(self) -> bool:
        """Return whether this interface already includes nuclear embedding."""
        return self.embedding

    def configure_electrostatic_embedding(self, enabled: bool) -> None:
        """Enable QM/MM electrostatics.

        Raises ValueError if active embedding conflicts with MM-only coupling.
        """
        if enabled:
            # Set configuration on the frozen interface during calculator setup.
            object.__setattr__(self, "embedding", True)
        elif self.embedding:
            raise ValueError(
                "pyscf-pbc embedding conflicts with this QMMMHamiltonian:"
                " no QM/MM electrostatic interaction is assigned to the QM"
                " level. Disable embedding or select electrostatic coupling"
                " to avoid double-counting electrostatics.",
            )

    def add_electronic_potential(
            self, potential: ElectronicPotential,
    ) -> None:
        """Register an electronic field, adapting molecular PME to region III only."""
        # Other electronic fields do not require helPME.
        if hasattr(potential, "pme"):
            from pydft_qmmm.potentials.pme_potential import PMEElectronicPotential
            from .pbc_pme import PeriodicPMEElectronicPotential
            if isinstance(potential, PMEElectronicPotential) and not isinstance(
                    potential, PeriodicPMEElectronicPotential,
            ):
                potential = PeriodicPMEElectronicPotential(
                    potential.system, potential.pme_alpha,
                    potential.pme_gridnumber, potential.pme_spline_order,
                )
        self.potentials.append(potential)

    def nuclear_charges(self) -> NDArray[np.float64]:
        """Return QM valence charges (e), ordered by system index."""
        cell, _ = build_cell(
            self.system, self.basis, self.pseudo, self.ke_cutoff,
            self.mesh, self.charge, self.multiplicity, self.verbose,
        )
        return valence_charges(cell)

    @system_cache("positions", "charges", "elements", "subsystems", "box")
    def _scf_state(self) -> _SCFState:
        """Converge the periodic SCF; raise RuntimeError on nonconvergence.

        Keep PySCF's QM-QM Ewald energy and add nuclear embedding separately.
        """
        backend = load_backend(self.device)
        cell, qm_indices = build_cell(
            self.system, self.basis, self.pseudo, self.ke_cutoff,
            self.mesh, self.charge, self.multiplicity, self.verbose,
        )
        kpts = cell.make_kpts([1, 1, 1])
        dft = load_submodule(backend, "pbc.dft")
        solver = dft.KUKS if self.multiplicity > 1 else dft.KRKS
        method = solver(cell, kpts=kpts, xc=self.functional)
        method.conv_tol = self.conv_tol
        method.max_cycle = self.max_cycle
        if self.output_file is not None:
            self.frame[0] += 1
            if self.frame[0] % self.output_interval == 0:
                method.stdout = open(self.output_file, "a")
                method.verbose = max(self.verbose, 4)
        for key, value in self.options.items():
            setattr(method, key, value)
        embed_indices = tuple(sorted(self.system.select("subsystem II")))
        coords, weights = grid_coordinates(cell)
        nuclear_energy = 0.0
        potential = np.zeros(len(coords))
        nao = cell.nao_nr()
        matrix = np.zeros((len(kpts), nao, nao), dtype=np.complex128)
        if self.embedding:
            potential = external_potential(
                self.system, cell, self.potentials, embed_indices,
                self.embedding_sigma,
            )
            matrix = ao_operator(
                backend, cell, kpts, coords, weights, potential,
            )
            core = method.get_hcore()
            shifted = to_like(to_numpy(core) + matrix, core)
            method.get_hcore = lambda *args, **kwargs: shifted
            nuclear_energy = _nuclear_coupling(
                cell, potential, self.system.box,
            )
        method.kernel(dm0=self.density_guess[0])
        if not method.converged:
            raise RuntimeError(
                "the periodic SCF did not converge; raise max_cycle or"
                " loosen conv_tol",
            )
        dm = method.make_rdm1()
        self.density_guess[0] = dm
        return _SCFState(
            cell, qm_indices, embed_indices, kpts, method, dm,
            coords, weights, potential, matrix, nuclear_energy,
            tuple(self.potentials),
        )


class PySCFPBCPotential(PySCFPBCInterface, AtomicPotential):
    """A potential wrapping periodic PySCF."""

    def compute_energy(self) -> float:
        """Return the embedded QM energy (kJ/mol)."""
        state = self._scf_state()
        return (
            float(state.method.e_tot) + state.nuclear_energy
        ) * KJMOL_PER_EH

    def compute_forces(self) -> NDArray[np.float64]:
        """Return forces (kJ/mol/Å) in system atom order."""
        from .pbc_forces import mm_forces
        from .pbc_forces import qm_forces
        state = self._scf_state()
        backend = load_backend(self.device)
        natoms = len(self.system.positions)
        forces = qm_forces(backend, state, self.system.box, natoms)
        if self.embedding:
            forces += mm_forces(
                backend, state, self.system, self.embedding_sigma, natoms,
            )
        return forces * KJMOL_PER_EH * BOHR_PER_ANGSTROM

    def compute_components(self) -> dict[str, float]:
        """Return QM and nuclear embedding energies (kJ/mol)."""
        state = self._scf_state()
        return {
            "Periodic QM Energy": float(state.method.e_tot) * KJMOL_PER_EH,
            "Nuclear Embedding Energy": (
                state.nuclear_energy * KJMOL_PER_EH
            ),
        }
