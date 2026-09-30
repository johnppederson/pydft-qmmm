"""PME external potential, evaluated inside VASP's interpreter.

This module needs helpme_py.

helPME can report the potential at ARBITRARY coordinates, so the natural
evaluation points are VASP's own FFT grid.  That grid is chosen by VASP from
ENCUT, PREC and the cell, and is only known once VASP is running.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

from .grid_potential import EPS0  # noqa: F401

if TYPE_CHECKING:
    from collections.abc import Sequence
    from numpy.typing import NDArray

# (kJ / mol) per eV, and helPME's Coulomb constant in kJ*A/mol/e**2.
KJMOL_PER_EV = 96.48533212331
COULOMB_CONSTANT = 1389.3545764438198

PME_FILE = "PME_DATA"


def _minimum_image():
    """Whether exclusions use the nearest-image separation."""
    return bool(os.environ.get("PYDFT_QMMM_PME_MINIMUM_IMAGE", "").strip())


def read_pme_data(path, expect_step=None):
    """Read the system state and Ewald parameters written by the driver.

    Returns:
        positions (Nx3, Angstrom), charges (N, e), excluded indices,
        alpha (1/Angstrom), gridnumber (3-tuple), spline_order, step.
    """
    with open(path) as fh:
        head = fh.readline().split()
        count, n_excluded, step = int(head[0]), int(head[1]), int(head[2])
        alpha = float(head[3])
        gridnumber = (int(head[4]), int(head[5]), int(head[6]))
        spline_order = int(head[7])
        data = np.array(
            [fh.readline().split() for _ in range(count)], dtype=np.float64,
        ).reshape(count, 4)
        excluded_line = fh.readline().split()
    if len(excluded_line) != n_excluded:
        raise ValueError(
            f"{path} declares {n_excluded} excluded indices but holds "
            f"{len(excluded_line)}.",
        )
    if expect_step is not None and step != expect_step:
        raise ValueError(
            f"{path} is at step {step}, expected {expect_step}.",
        )
    excluded = np.array([int(i) for i in excluded_line], dtype=np.int64)
    return (
        data[:, :3].copy(), data[:, 3].copy(), excluded,
        alpha, gridnumber, spline_order, step,
    )


def write_pme_data(
        path: str,
        positions: NDArray[np.float64],
        charges: NDArray[np.float64],
        excluded: Sequence[int],
        alpha: float,
        gridnumber: tuple[int, int, int],
        spline_order: int,
        step: int,
) -> None:
    r"""Write everything the plugin needs to rebuild the PME potential.

    Args:
        path: The destination file.
        positions: An Nx3 array of positions
            (:math:`\mathrm{\mathring{A}}`) for the whole system.
        charges: An N array of charges (:math:`e`).
        excluded: Indices whose real-space contribution is removed.
        alpha: The Ewald splitting parameter
            (:math:`\mathrm{\mathring{A}^{-1}}`).
        gridnumber: PME grid points along each lattice edge.
        spline_order: The B-spline interpolation order.
        step: A monotonically increasing stamp, as for MM_CHARGES.
    """
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    charges = np.asarray(charges, dtype=np.float64).reshape(-1)
    if len(positions) != len(charges):
        raise ValueError(
            f"{len(positions)} positions but {len(charges)} charges.",
        )
    excluded = np.asarray(sorted(excluded), dtype=np.int64)
    if len(excluded) and (excluded[-1] >= len(charges) or excluded[0] < 0):
        raise ValueError("excluded index out of range.")
    with open(path, "w") as fh:
        fh.write(
            f"{len(charges)} {len(excluded)} {step} {alpha:.12e} "
            f"{gridnumber[0]} {gridnumber[1]} {gridnumber[2]} "
            f"{spline_order}\n",
        )
        for (x, y, z), q in zip(positions, charges):
            fh.write(f"{x:.12e} {y:.12e} {z:.12e} {q:.12e}\n")
        fh.write(" ".join(str(int(i)) for i in excluded) + "\n")


def _lattice_constants(cell):
    """Return a, b, c, alpha, beta, gamma for a set of lattice vectors."""
    lengths = np.linalg.norm(cell, axis=1)
    a, b, c = lengths

    def angle(u, v):
        cosine = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    return (
        float(a), float(b), float(c),
        angle(cell[1], cell[2]),   # alpha
        angle(cell[0], cell[2]),   # beta
        angle(cell[0], cell[1]),   # gamma
    )


def grid_coordinates(shape, cell):
    """Cartesian coordinates of every FFT grid point, as an Nx3 array."""
    axes = [np.arange(n, dtype=np.float64) / n for n in shape]
    fractional = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return fractional.reshape(-1, 3) @ cell


def build_pme_potential(path, shape, cell, expect_step=None, chunk=1 << 20):
    """Build V_ext on VASP's grid from the PME reciprocal-space sum.

    Mirrors PMEElectronicPotential.compute_potential.

    Args:
        path: The PME_DATA file written by the interface.
        shape: VASP's FFT grid dimensions.
        cell: A 3x3 array whose rows are lattice vectors (Angstrom).
        expect_step: Optional step stamp to verify.
        chunk: Number of grid points evaluated per helPME call.

    Returns:
        V_ext on the grid (eV), with shape ``shape``.
    """
    import helpme_py

    positions, charges, excluded, alpha, gridnumber, spline_order, _ = (
        read_pme_data(path, expect_step)
    )
    pme = helpme_py.PMEInstanceD()
    pme.setup(1, alpha, spline_order, *gridnumber, COULOMB_CONSTANT, 0)
    pme.set_lattice_vectors(
        *_lattice_constants(np.asarray(cell, dtype=np.float64)),
        helpme_py.LatticeType.XAligned,
    )
    all_charges = helpme_py.MatrixD(charges.reshape(-1, 1))
    all_positions = helpme_py.MatrixD(positions)
    if excluded.size:
        excluded_charges = helpme_py.MatrixD(charges[excluded].reshape(-1, 1))
        excluded_positions = helpme_py.MatrixD(positions[excluded, :])

    coordinates = grid_coordinates(shape, cell)
    potential = np.zeros(len(coordinates))
    for start in range(0, len(coordinates), chunk):
        block = np.ascontiguousarray(coordinates[start:start + chunk])
        values = np.zeros((len(block), 1))
        matrix = helpme_py.MatrixD(values)
        pme.compute_P_rec(0, all_charges, all_positions,
                          helpme_py.MatrixD(block), 0, matrix)
        if excluded.size:
            pme.compute_P_adj(0, excluded_charges, excluded_positions,
                              helpme_py.MatrixD(block), matrix,
                              _minimum_image())
        potential[start:start + chunk] = values[:, 0]
    # helPME reports the electrostatic potential in kJ/mol/e; VASP wants
    # the electron potential ENERGY in eV, hence the negation.
    return (-potential / KJMOL_PER_EV).reshape(shape)
