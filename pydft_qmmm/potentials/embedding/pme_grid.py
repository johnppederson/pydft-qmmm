"""Native-grid PME potential and conservative force dispatch.

VASP chooses its grid at runtime. A zero grid in PME_DATA requests that
grid; explicit PME grids and unequal cells use arbitrary-point support.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

from .grid_potential import _g_squared
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
        gridnumber: PME grid points along each lattice edge. The internal
            marker (0, 0, 0) selects the runtime QM grid for shared cells.
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


def _native_grid_shape(shape):
    """Whether serial helPME keeps these FFT dimensions without padding."""
    for size in shape:
        remainder = int(size)
        if remainder != size or remainder < 1:
            return False
        for factor in (2, 3, 5, 7):
            while remainder > 1 and remainder % factor == 0:
                remainder //= factor
        if remainder not in (1, 11, 13):
            return False
    return True


def _resolve_grid(gridnumber, shape, cell, source_cell):
    """Resolve the runtime QM-grid marker, never across unequal cells."""
    if tuple(gridnumber) == (0, 0, 0):
        if source_cell is not None and not np.array_equal(source_cell, cell):
            raise ValueError("Automatic PME grid requires matching QM/MM cells")
        return tuple(shape)
    return gridnumber


def build_pme_potential(
    path, shape, cell, expect_step=None, chunk=1 << 20,
    source_cell=None, origin=None,
):
    """Build V_ext on VASP's grid from the PME reciprocal-space sum.

    Mirrors PMEElectronicPotential.compute_potential.
    With helpme-py 0.3.1 or newer, coincident PME and target grids use
    compute_P_rec_grid directly. Other grids use point evaluation.

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
    gridnumber = _resolve_grid(gridnumber, shape, cell, source_cell)
    # Empty region III needs no reciprocal field or grid coordinates.
    if not np.any(charges):
        return np.zeros(shape, dtype=np.float64)
    if (
            hasattr(helpme_py.PMEInstanceD, "compute_P_rec_grid")
            and np.lib.NumpyVersion(helpme_py.__version__) >= "0.3.1"
            and tuple(shape) == tuple(gridnumber)
            and _native_grid_shape(shape)
            and (source_cell is None or np.array_equal(source_cell, cell))
            and (origin is None or not np.any(origin))
            and not excluded.size
    ):
        pme = helpme_py.PMEInstanceD()
        pme.setup(1, alpha, spline_order, *gridnumber, COULOMB_CONSTANT, 0)
        pme.set_lattice_vectors(
            *_lattice_constants(np.asarray(
                cell if source_cell is None else source_cell, dtype=np.float64,
            )),
            helpme_py.LatticeType.XAligned,
        )
        all_charges = helpme_py.MatrixD(charges.reshape(-1, 1))
        all_positions = helpme_py.MatrixD(positions)
        values = np.zeros((int(np.prod(shape)), 1))
        pme.compute_P_rec_grid(
            0, all_charges, all_positions, helpme_py.MatrixD(values),
        )
        # helPME stores x fastest; our grid arrays store z fastest.
        potential = values[:, 0].reshape(tuple(shape)[::-1]).transpose(2, 1, 0)
        return np.ascontiguousarray(-potential / KJMOL_PER_EV)
    from .pme_points import point_potential
    return point_potential(
        positions, charges, excluded, alpha, gridnumber, spline_order,
        shape, cell, chunk, source_cell, origin,
    )


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

    The transpose of ``build_pme_potential``. Matching native grids in
    diagonal cells, with spline order >= 4, use a direct grid transpose
    without re-spreading every QM mesh point. Other cases use helPME's
    point evaluator.
    The QM energy in that
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
        (:math:`\\mathrm{kJ\\;mol^{-1}\\;\\mathring{A}^{-1}}`), one row per
        PME_DATA atom; atoms with zero charge get zero.
    """
    import helpme_py

    positions, charges, excluded, alpha, gridnumber, spline_order, _ = (
        read_pme_data(path, expect_step)
    )
    gridnumber = _resolve_grid(gridnumber, shape, cell, source_cell)
    # Validate the frame above even when there are no charged targets.
    if not np.any(charges):
        return np.zeros((len(charges), 3), dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    if (
            hasattr(helpme_py.PMEInstanceD, "compute_P_rec_grid")
            and np.lib.NumpyVersion(helpme_py.__version__) >= "0.3.1"
            and tuple(shape) == tuple(gridnumber)
            and _native_grid_shape(shape)
            and (source_cell is None or np.array_equal(source_cell, cell))
            and (origin is None or not np.any(origin))
            and not excluded.size
            and (points is None or not len(points))
            and cell.shape == (3, 3)
            and np.array_equal(cell, np.diag(np.diag(cell)))
            and np.all(np.diag(cell) > 0)
            and spline_order >= 4
    ):
        from .pme_grid_forces import native_grid_source_forces
        return native_grid_source_forces(
            positions, charges, grid_charges, tuple(shape), np.diag(cell),
            alpha, spline_order, COULOMB_CONSTANT,
        )
    from .pme_points import point_source_forces
    return point_source_forces(
        positions, charges, excluded, alpha, gridnumber, spline_order,
        grid_charges, shape, cell, points, point_charges, source_cell, origin,
    )
