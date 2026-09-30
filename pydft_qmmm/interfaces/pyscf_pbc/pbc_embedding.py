"""Periodic MM embedding on the pseudopotential solver's uniform FFT grid."""
from __future__ import annotations

__all__ = [
    "GRID_BLOCK_SIZE",
    "grid_coordinates",
    "near_potential",
    "external_potential",
    "ao_operator",
]

from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from pydft_qmmm.embedding.grid_potential import poisson_fft
from pydft_qmmm.embedding.grid_potential import spread_gaussian
from pydft_qmmm.utils import BOHR_PER_ANGSTROM
from pydft_qmmm.utils import KJMOL_PER_EH
from pydft_qmmm.utils import KJMOL_PER_EV
from ..pyscf.pyscf_backend import load_submodule
from ..pyscf.pyscf_backend import to_numpy

# Bound the (nkpts, block, nao) AO buffer.
GRID_BLOCK_SIZE = 16384

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType
    from numpy.typing import NDArray
    from pydft_qmmm import System
    from pydft_qmmm.potentials import ElectronicPotential


def grid_coordinates(
        cell: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return C-ordered FFT coordinates (Å) and quadrature weights (Bohr³)."""
    coords = np.asarray(cell.get_uniform_grids())
    weights = np.full(len(coords), cell.vol / len(coords))
    return coords / BOHR_PER_ANGSTROM, weights


def _smeared_potential_grid(
        positions: NDArray[np.float64],
        charges: NDArray[np.float64],
        mesh: tuple[int, int, int],
        box: NDArray[np.float64],
        sigma: float,
) -> NDArray[np.float64]:
    """Return the periodic Gaussian-charge potential on mesh, in volts.

    Positions, row lattice vectors in box, and Gaussian width sigma use Å;
    charges use e.
    """
    rho = spread_gaussian(positions, charges, mesh, box, sigma)
    return poisson_fft(rho, box)


def near_potential(
        system: System,
        embed_indices: Sequence[int],
        mesh: tuple[int, int, int],
        box: NDArray[np.float64],
        sigma: float,
) -> NDArray[np.float64]:
    """Return region II's periodic electron potential on mesh, in Hartree.

    embed_indices selects system atoms. Box vectors and Gaussian width
    sigma use Å.
    """
    indices = list(embed_indices)
    if not indices:
        return np.zeros(mesh)
    positions = np.asarray(system.positions)[indices]
    charges = np.asarray(system.charges)[indices]
    volts = _smeared_potential_grid(positions, charges, mesh, box, sigma)
    # Electron charge is −1; convert volts to Hartree per electron.
    return -volts * KJMOL_PER_EV / KJMOL_PER_EH


def external_potential(
        system: System,
        cell: Any,
        potentials: Sequence[ElectronicPotential],
        embed_indices: Sequence[int],
        sigma: float,
) -> NDArray[np.float64]:
    """Return V_ext (Hartree per electron) on the cell's C-ordered FFT grid.

    Combine region III-only PME with region II's periodic Gaussian field
    of width sigma (Å). Reaction forces must use the same source split.
    """
    coords, _ = grid_coordinates(cell)
    box = np.asarray(system.box, dtype=np.float64)
    mesh = tuple(int(n) for n in cell.mesh)
    total = np.zeros(len(coords))
    for potential in potentials:
        total += np.asarray(potential.compute_potential(coords)).reshape(-1)
    total += near_potential(
        system, embed_indices, mesh, box, sigma,
    ).reshape(-1)
    return total


def ao_operator(
        backend: ModuleType,
        cell: Any,
        kpts: NDArray[np.float64],
        coords: NDArray[np.float64],
        weights: NDArray[np.float64],
        potential: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Return the external AO operator (Hartree), shaped (nkpts, nao, nao).

    backend is pyscf or gpu4pyscf. Coordinates use Å, weights Bohr³,
    potential Hartree, and kpts inverse Bohr. The result is on the host.
    """
    numint = load_submodule(backend, "pbc.dft.numint")
    nao = cell.nao_nr()
    kpts = np.asarray(kpts).reshape(-1, 3)
    matrix = np.zeros((len(kpts), nao, nao), dtype=np.complex128)
    bohr = coords * BOHR_PER_ANGSTROM
    for start in range(0, len(coords), GRID_BLOCK_SIZE):
        stop = start + GRID_BLOCK_SIZE
        block = np.ascontiguousarray(bohr[start:stop])
        scale = weights[start:stop] * potential[start:stop]
        ao_kpts = numint.eval_ao_kpts(cell, block, kpts=kpts, deriv=0)
        for index, ao in enumerate(ao_kpts):
            ao = to_numpy(ao)
            matrix[index] += np.einsum(
                "gi,g,gj->ij", ao.conj(), scale, ao,
            )
    return matrix
