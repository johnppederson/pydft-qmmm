"""Build the periodic QM cell from the simulation box."""
from __future__ import annotations

__all__ = ["build_cell", "valence_charges"]

from typing import Any
from typing import TYPE_CHECKING

import numpy as np
from pyscf.pbc import gto

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pydft_qmmm import System


def build_cell(
        system: System,
        basis: str,
        pseudo: str,
        ke_cutoff: float | None,
        mesh: tuple[int, int, int] | None,
        charge: int,
        multiplicity: int,
        verbose: int,
) -> tuple[Any, tuple[int, ...]]:
    """Build subsystem I in the simulation box; return its cell and atom indices.

    Both backends use a PySCF Cell. Positions and lattice vectors are in Å;
    ke_cutoff is in Hartree. Supply either ke_cutoff or an explicit FFT mesh.
    """
    qm_indices = tuple(sorted(system.select("subsystem I")))
    positions = np.asarray(system.positions)
    elements = system.elements
    cell = gto.Cell()
    cell.atom = [
        (str(elements[index]), tuple(float(x) for x in positions[index]))
        for index in qm_indices
    ]
    cell.unit = "Angstrom"
    cell.a = np.asarray(system.box, dtype=np.float64)
    cell.basis = basis
    cell.pseudo = pseudo
    cell.charge = charge
    cell.spin = multiplicity - 1
    cell.verbose = verbose
    if ke_cutoff is not None:
        cell.ke_cutoff = ke_cutoff
    else:
        cell.mesh = list(mesh)          # type: ignore[arg-type]
    cell.build()
    return cell, qm_indices


def valence_charges(cell: Any) -> NDArray[np.float64]:
    """Return nuclear valence charges (e) in cell atom order."""
    return np.array(
        [cell.atom_charge(index) for index in range(cell.natm)],
        dtype=np.float64,
    )
