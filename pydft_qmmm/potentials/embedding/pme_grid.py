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
from .grid_potential import _g_squared

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


def build_pme_potential(path, shape, cell, expect_step=None, chunk=1 << 20,
                        source_cell=None, origin=None):
    """Build V_ext on VASP's grid from the PME reciprocal-space sum.

    Mirrors PMEElectronicPotential.compute_potential.

    Args:
        path: The PME_DATA file written by the interface.
        shape: VASP's FFT grid dimensions.
        cell: A 3x3 array whose rows are lattice vectors (Angstrom).
        expect_step: Optional step stamp to verify.
        chunk: Number of grid points evaluated per helPME call.
        source_cell: Optional MM lattice (rows); defaults to cell.
        origin: Optional global Cartesian origin of the target grid.

    Returns:
        V_ext on the grid (eV), with shape ``shape``.
    """
    import helpme_py

    positions, charges, excluded, alpha, gridnumber, spline_order, _ = (
        read_pme_data(path, expect_step)
    )
    # Empty region III needs no reciprocal field or grid coordinates.
    if not np.any(charges):
        return np.zeros(shape, dtype=np.float64)
    pme = helpme_py.PMEInstanceD()
    pme.setup(1, alpha, spline_order, *gridnumber, COULOMB_CONSTANT, 0)
    pme.set_lattice_vectors(
        *_lattice_constants(np.asarray(cell if source_cell is None else source_cell, dtype=np.float64)),
        helpme_py.LatticeType.XAligned,
    )
    all_charges = helpme_py.MatrixD(charges.reshape(-1, 1))
    all_positions = helpme_py.MatrixD(positions)
    if excluded.size:
        excluded_charges = helpme_py.MatrixD(charges[excluded].reshape(-1, 1))
        excluded_positions = helpme_py.MatrixD(positions[excluded, :])

    coordinates = grid_coordinates(shape, cell)
    if origin is not None:
        coordinates += np.asarray(origin)
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


def charge_from_potential(phi, cell):
    """Grid charges whose periodic potential is ``phi``.

    Inverts ``poisson_fft``.  The G=0 term is lost, which the PME
    reciprocal sum ignores anyway.

    Args:
        phi: The electrostatic potential on the grid (volts).
        cell: A 3x3 array whose rows are lattice vectors (Angstrom).

    Returns:
        The charge (e) on each grid point, shaped like ``phi``.
    """
    phi = np.asarray(phi, dtype=np.float64)
    rho_g = np.fft.fftn(phi) * EPS0 * _g_squared(phi.shape, cell)
    rho_g[0, 0, 0] = 0.0
    rho = np.fft.ifftn(rho_g).real
    return rho * abs(np.linalg.det(cell)) / phi.size


def pme_source_forces(
        path, grid_charges, shape, cell,
        points=None, point_charges=None, expect_step=None,
        source_cell=None, origin=None,
):
    """Forces on the PME_DATA charges from the QM charge.

    The transpose of ``build_pme_potential``.  The QM energy in that
    field is sum_s q_s V(r_s) over grid charges and point charges s, and
    the reciprocal operator is symmetric in sources and targets, so
    helPME evaluates the field of s at the PME_DATA atoms instead.

    Args:
        path: The PME_DATA file the field was built from.
        grid_charges: The QM charge (e) on each point of the grid.
        shape: The grid dimensions.
        cell: A 3x3 array whose rows are lattice vectors (Angstrom).
        points: Optional Nx3 positions (Angstrom) of extra point charges,
            such as nuclei the engine couples to the field as points.
        point_charges: The charges (e) at ``points``. Points are global.
        source_cell: Optional MM lattice (rows); defaults to cell.
        origin: Optional global Cartesian origin of the QM grid.
        expect_step: Optional step stamp to verify.

    Returns:
        An Nx3 array of forces
        (:math:`\\mathrm{kJ\;mol^{-1}\;\\mathring{A}^{-1}}`), one row per
        PME_DATA atom; atoms with zero charge get zero.
    """
    import helpme_py

    positions, charges, excluded, alpha, gridnumber, spline_order, _ = (
        read_pme_data(path, expect_step)
    )
    # Validate the frame above even when there are no charged targets.
    if not np.any(charges):
        return np.zeros((len(charges), 3), dtype=np.float64)
    pme = helpme_py.PMEInstanceD()
    pme.setup(1, alpha, spline_order, *gridnumber, COULOMB_CONSTANT, 0)
    pme.set_lattice_vectors(
        *_lattice_constants(np.asarray(cell if source_cell is None else source_cell, dtype=np.float64)),
        helpme_py.LatticeType.XAligned,
    )
    sources = np.asarray(grid_charges, dtype=np.float64).reshape(-1)
    coordinates = grid_coordinates(shape, cell)
    if origin is not None:
        coordinates += np.asarray(origin)
    if points is not None and len(points):
        coordinates = np.vstack([coordinates, np.asarray(points).reshape(-1, 3)])
        sources = np.concatenate([sources, np.asarray(point_charges).reshape(-1)])
    sources = helpme_py.MatrixD(np.ascontiguousarray(sources.reshape(-1, 1)))
    coordinates = helpme_py.MatrixD(np.ascontiguousarray(coordinates))
    forces = np.zeros((len(charges), 3))
    targets = np.flatnonzero(charges)
    if targets.size:
        field = np.zeros((targets.size, 4))
        pme.compute_P_rec(
            0, sources, coordinates,
            helpme_py.MatrixD(np.ascontiguousarray(positions[targets])),
            1, helpme_py.MatrixD(field),
        )
        forces[targets] = -field[:, 1:] * charges[targets, None]
    if excluded.size:
        # The transpose of the erf subtraction in build_pme_potential.
        adjustment = np.zeros((excluded.size, 4))
        pme.compute_PDP_adj(
            0, sources, coordinates,
            helpme_py.MatrixD(np.ascontiguousarray(positions[excluded])),
            helpme_py.MatrixD(adjustment), _minimum_image(),
        )
        forces[excluded] -= adjustment[:, 1:] * charges[excluded, None]
    return forces
