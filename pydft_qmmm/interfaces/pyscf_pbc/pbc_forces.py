"""Periodic embedding forces: SCF, AO Pulay, nuclear, and MM reaction terms.

All embedding terms differentiate the same discrete grid potential.
"""
from __future__ import annotations

__all__ = [
    "pulay_forces",
    "nuclear_forces",
    "qm_forces",
    "qm_density_on_grid",
    "nuclear_charge_on_grid",
    "mm_forces",
]

from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from pydft_qmmm.embedding.grid_potential import contract_gaussian_gradient
from pydft_qmmm.embedding.grid_potential import poisson_fft
from pydft_qmmm.embedding.grid_potential import spectral_value_and_gradient
from pydft_qmmm.utils import BOHR_PER_ANGSTROM
from pydft_qmmm.utils import KJMOL_PER_EH
from pydft_qmmm.utils import KJMOL_PER_EV
from ..pyscf.pyscf_backend import load_submodule
from ..pyscf.pyscf_backend import to_numpy
from .pbc_cell import valence_charges
from .pbc_embedding import GRID_BLOCK_SIZE

if TYPE_CHECKING:
    from types import ModuleType
    from numpy.typing import NDArray
    from pydft_qmmm import System


def pulay_forces(
        backend: ModuleType,
        state: Any,
        natoms: int,
) -> NDArray[np.float64]:
    """Return the external-potential AO Pulay forces (Hartree/Bohr).

    PySCF's gradient omits this derivative of the added AO operator.
    The array covers natoms system atoms; only QM rows are nonzero.
    """
    numint = load_submodule(backend, "pbc.dft.numint")
    cell = state.cell
    ao_slices = cell.aoslice_by_atom()
    dm = np.asarray(to_numpy(state.dm))
    nkpts = len(state.kpts)
    bohr = state.coords * BOHR_PER_ANGSTROM
    gradient = np.zeros((cell.natm, 3))
    for start in range(0, len(state.coords), GRID_BLOCK_SIZE):
        stop = start + GRID_BLOCK_SIZE
        block = np.ascontiguousarray(bohr[start:stop])
        scale = state.weights[start:stop] * state.potential[start:stop]
        ao_kpts = numint.eval_ao_kpts(
            cell, block, kpts=state.kpts, deriv=1,
        )
        for index, ao in enumerate(ao_kpts):
            ao = np.asarray(to_numpy(ao))
            value, derivs = ao[0], ao[1:4]
            for atom in range(cell.natm):
                first, last = ao_slices[atom][2], ao_slices[atom][3]
                # Basis-center derivatives change sign; bra and ket give 2 Re.
                gradient[atom] += -2.0 * np.einsum(
                    "xgi,g,gj,ij->x",
                    derivs[:, :, first:last].conj(),
                    scale,
                    value,
                    dm[index][first:last, :],
                ).real / nkpts
    forces = np.zeros((natoms, 3))
    forces[list(state.qm_indices)] -= gradient
    return forces


def nuclear_forces(
        state: Any,
        box: NDArray[np.float64],
        natoms: int,
) -> NDArray[np.float64]:
    """Return QM nuclear embedding forces (Hartree/Bohr) in system atom order.

    Use the energy's Fourier interpolant and valence charges; box uses Å.
    """
    cell = state.cell
    mesh = tuple(int(n) for n in cell.mesh)
    positions = np.asarray(cell.atom_coords()) / BOHR_PER_ANGSTROM
    _, gradient = spectral_value_and_gradient(
        state.potential.reshape(mesh),
        np.asarray(box, dtype=np.float64),
        positions,
    )
    forces = np.zeros((natoms, 3))
    # E = −Σ ZV, so F = +Z∇V; convert Å⁻¹ to Bohr⁻¹.
    charges = valence_charges(cell)[:, None]
    forces[list(state.qm_indices)] += (
        charges * gradient / BOHR_PER_ANGSTROM
    )
    return forces


def qm_forces(
        backend: ModuleType,
        state: Any,
        box: NDArray[np.float64],
        natoms: int,
) -> NDArray[np.float64]:
    """Return SCF, Pulay, and nuclear forces (Hartree/Bohr) in system order."""
    forces = np.zeros((natoms, 3))
    gradient = state.method.nuc_grad_method()
    gradient.verbose = 0
    forces[list(state.qm_indices)] -= np.asarray(
        to_numpy(gradient.kernel()),
    )
    if state.potential.any():
        forces += pulay_forces(backend, state, natoms)
        forces += nuclear_forces(state, box, natoms)
    return forces


def qm_density_on_grid(
        backend: ModuleType,
        state: Any,
) -> NDArray[np.float64]:
    """Return signed electronic charges (e per FFT point), averaged over k-points.

    Nuclear charges are added separately in mm_forces.
    """
    numint = load_submodule(backend, "pbc.dft.numint")
    cell = state.cell
    dm = np.asarray(to_numpy(state.dm))
    nkpts = len(state.kpts)
    bohr = state.coords * BOHR_PER_ANGSTROM
    charge = np.zeros(len(state.coords))
    for start in range(0, len(state.coords), GRID_BLOCK_SIZE):
        stop = start + GRID_BLOCK_SIZE
        block = np.ascontiguousarray(bohr[start:stop])
        ao_kpts = numint.eval_ao_kpts(cell, block, kpts=state.kpts, deriv=0)
        density = np.zeros(len(block))
        for index, ao in enumerate(ao_kpts):
            ao = np.asarray(to_numpy(ao))
            density += np.einsum(
                "gi,ij,gj->g", ao.conj(), dm[index], ao,
            ).real / nkpts
        charge[start:stop] = -density * state.weights[start:stop]
    return charge


def nuclear_charge_on_grid(
        state: Any,
        box: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Transpose nuclear Fourier interpolation into charges (e per FFT point).

    Ensures Σ q_grid V_grid = Σ Z V(R), including even-mesh Nyquist modes.
    Box vectors use Å. MM sources already carry the Gaussian smoothing.
    """
    mesh = tuple(int(n) for n in state.cell.mesh)
    miller = [np.fft.fftfreq(n) * n for n in mesh]
    reciprocal = 2.0 * np.pi * np.linalg.inv(box).T
    coefficients = np.zeros(mesh, dtype=complex)
    positions = np.asarray(state.cell.atom_coords()) / BOHR_PER_ANGSTROM
    for position, charge in zip(positions, valence_charges(state.cell)):
        projection = reciprocal @ position
        phase = [np.exp(-1j * m * x) for m, x in zip(miller, projection)]
        coefficients += charge * np.einsum("i,j,k->ijk", *phase)
    return np.fft.ifftn(coefficients).real.reshape(-1)


def mm_forces(
        backend: ModuleType,
        state: Any,
        system: System,
        sigma: float,
        natoms: int,
) -> NDArray[np.float64]:
    """Return MM reaction forces (Hartree/Bohr) in system atom order.

    Use the Gaussian-spreading adjoint for II and the forward PME instance
    for III. Nuclear charges use the grid interpolation transpose.
    Sigma is the region II Gaussian width (Å).
    """
    cell = state.cell
    mesh = tuple(int(n) for n in cell.mesh)
    box = np.asarray(system.box, dtype=np.float64)
    forces = np.zeros((natoms, 3))
    charge = qm_density_on_grid(backend, state)
    charge += nuclear_charge_on_grid(state, box)

    indices = list(state.embed_indices)
    if indices:
        volume_element = abs(np.linalg.det(box)) / charge.size
        rho = charge.reshape(mesh) / volume_element
        # Poisson gives volts; Gaussian contraction gives eV/Å.
        phi = poisson_fft(rho, box)
        forces[indices] += contract_gaussian_gradient(
            phi,
            np.asarray(system.positions)[indices],
            np.asarray(system.charges)[indices],
            mesh, box, sigma,
        ) * KJMOL_PER_EV / KJMOL_PER_EH / BOHR_PER_ANGSTROM

    # Nuclear reactions must transpose the same interpolation as the energy.
    for potential in state.potentials:
        reciprocal = np.asarray(
            potential.compute_source_forces(state.coords, charge),
        ).copy()
        reciprocal[list(state.qm_indices)] = 0.0
        forces += reciprocal / KJMOL_PER_EH / BOHR_PER_ANGSTROM
    return forces
