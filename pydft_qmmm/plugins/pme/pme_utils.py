#! /usr/bin/env python3
"""A module to define the :class:`PBCCalculator` class.
"""
from __future__ import annotations

import itertools
import math
from typing import Any
from typing import TYPE_CHECKING

import numba
import numpy as np
import scipy.interpolate

from pydft_qmmm.common import BOHR_PER_ANGSTROM
from pydft_qmmm.common import KJMOL_PER_EH

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pydft_qmmm import System


BOHR_PER_NM = 18.89726  # a0 / nm


def pme_components(
        system: System,
        quadrature: NDArray[np.float64],
        pme_potential: NDArray[np.float64],
        pme_gridnumber: int,
        pme_alpha: float | int,
) -> tuple[Any, ...]:
    # Create necessary objects.
    qm_atoms = [
        x for y in system.topology.qm_atoms() for x in y
    ]
    nuclei = (
        system.state.positions()[qm_atoms, :]
        * BOHR_PER_ANGSTROM
    )
    indices = np.array(list(range(-1, pme_gridnumber+1)))
    grid = (indices,) * 3
    inverse_box = np.linalg.inv(
        system.state.box() * BOHR_PER_ANGSTROM,
    )
    # qm_drudes = self._topology.groups["qm_drude"]
    ae_atoms = [
        x for y in system.topology.ae_atoms() for x in y
    ]
    atoms = qm_atoms + ae_atoms  # + qm_drudes
    # Gather relevant State data.
    positions = system.state.positions()[atoms] * BOHR_PER_ANGSTROM
    charges = system.state.charges()[atoms]
    # Determine and apply exlcusions.
    pme_xyz, pme_exclusions = _compute_pme_exclusions(
        system.state.box(),
        inverse_box,
        quadrature,
        pme_gridnumber,
    )
    _apply_pme_exclusions(
        system.state.box(),
        atoms,
        positions,
        charges,
        pme_potential,
        pme_xyz,
        pme_exclusions,
        pme_gridnumber,
        pme_alpha,
    )
    # Prepare arrays for interpolation.
    pme_potential = np.reshape(
        pme_potential,
        (pme_gridnumber,) * 3,
    )
    pme_potential = np.pad(pme_potential, 1, mode="wrap")
    # Perform interpolations.
    pme_results = [
        _interp_pme_potential(
            inverse_box,
            grid,
            pme_potential,
            pme_gridnumber,
            quadrature,
        ),
        _interp_pme_potential(
            inverse_box,
            grid,
            pme_potential,
            pme_gridnumber,
            nuclei,
        ),
    ]
    pme_results.append(
        _interp_pme_gradient(
            inverse_box,
            grid,
            pme_potential,
            pme_gridnumber,
            nuclei,
        ),
    )
    # Calculate reciprocal-space correction energy.
    pme_results.insert(
        0,
        sum(
            (
                -v*q*KJMOL_PER_EH for v, q in
                zip(pme_results[1], system.state.charges()[qm_atoms])
            ),
        ),
    )
    return tuple(pme_results)


def _compute_pme_exclusions(
        box: NDArray[np.float64],
        inverse_box: NDArray[np.float64],
        quadrature: NDArray[np.float64],
        pme_gridnumber: int,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Create the PME points which will have exclusions applied.
    The points include the region containing the quadrature grid.

    :return: The x, y, and z, coordinates of PME gridpoints, and
        the indices to perform exclusions on.
    """
    # Create real-space coordinates of the PME grid in Bohr.
    norms = BOHR_PER_ANGSTROM*np.linalg.norm(box, axis=1)
    x = np.linspace(
        0, norms[0], pme_gridnumber, endpoint=False,
    )
    y = np.linspace(
        0, norms[1], pme_gridnumber, endpoint=False,
    )
    z = np.linspace(
        0, norms[2], pme_gridnumber, endpoint=False,
    )
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    x = X.flatten()[:, np.newaxis]
    y = Y.flatten()[:, np.newaxis]
    z = Z.flatten()[:, np.newaxis]
    pme_xyz = np.concatenate((x, y, z), axis=1)
    # Project quadrature grid to reciprocal space.
    points_project = project_to_grid(inverse_box, pme_gridnumber, quadrature)
    x_i = points_project
    indices = np.unique(np.floor(x_i).T, axis=1)
    edges = list(itertools.product(*[[i, i + 1] for i in indices]))
    edges = [np.stack(x, axis=-1) for x in edges]
    x_f = np.unique(np.concatenate(tuple(edges), axis=0), axis=0)
    x_f[x_f == pme_gridnumber] = 0
    pme_exclusions = np.unique(x_f, axis=0)
    return pme_xyz, pme_exclusions


def _apply_pme_exclusions(
        box: NDArray[np.float64],
        atoms: list[int],
        positions: NDArray[np.float64],
        charges: NDArray[np.float64],
        pme_potential: NDArray[np.float64],
        pme_xyz: NDArray[np.float64],
        pme_exclusions: NDArray[np.int32],
        pme_gridnumber: int,
        pme_alpha: float,
) -> None:
    """Apply exlcusions to relevant PME potential grid points.
    """
    indices = (
        pme_exclusions[:, 0]*pme_gridnumber**2
        + pme_exclusions[:, 1]*pme_gridnumber
        + pme_exclusions[:, 2]
    ).astype(np.int32)
    # Perform exclusion calculation
    exclusions = pme_xyz[indices, :]
    beta = pme_alpha / BOHR_PER_NM
    pme_potential = _compute_reciprocal_exclusions(
        pme_potential,
        indices,
        positions,
        charges,
        exclusions,
        beta,
        box * BOHR_PER_ANGSTROM,
    )


def _interp_pme_potential(
        inverse_box: NDArray[np.float64],
        grid: tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]],
        pme_potential: NDArray[np.float64],
        pme_gridnumber: int,
        points: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Calculates the PME potential interpolated at the points.

    :param points: The points to be interpolated on the PME
        potential grid.
    :return: The interpolated PME potential at the points.
    """
    points_project = project_to_grid(inverse_box, pme_gridnumber, points)
    interp_potential = scipy.interpolate.interpn(
        grid,
        pme_potential,
        points_project,
        method='linear',
    )
    return interp_potential


def _interp_pme_gradient(
        inverse_box: NDArray[np.float64],
        grid: tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]],
        pme_potential: NDArray[np.float64],
        pme_gridnumber: int,
        points: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Create the chain rule for the PME potential on the nuclei.

    :param points: The points to be interpolated on the PME
        potential gradient grid.
    :return: The interpolated PME gradient at the points.
    """
    points_project = project_to_grid(inverse_box, pme_gridnumber, points)
    # This code is largely based on
    # scipy.interpolate.RegularGridInterpolator._evaluate linear.
    interp_function = scipy.interpolate.RegularGridInterpolator(
        grid,
        pme_potential,
    )
    indices, norm_dist, _ = interp_function._find_indices(
        points_project.T,
    )
    edges = list(itertools.product(*[[i, i + 1] for i in indices]))
    grad_x = np.zeros((len(points),))
    grad_y = np.zeros((len(points),))
    grad_z = np.zeros((len(points),))
    for edge_indices in edges:
        weight_x = 1
        weight_y = 1
        weight_z = 1
        for j, (e_i, i, y_i) in enumerate(
                zip(edge_indices, indices, norm_dist),
        ):
            if j == 0:
                weight_x *= np.where(e_i == i, -1.0, 1.0)
                weight_z *= np.where(e_i == i, 1 - y_i, y_i)
                weight_y *= np.where(e_i == i, 1 - y_i, y_i)
            if j == 1:
                weight_y *= np.where(e_i == i, -1.0, 1.0)
                weight_x *= np.where(e_i == i, 1 - y_i, y_i)
                weight_z *= np.where(e_i == i, 1 - y_i, y_i)
            if j == 2:
                weight_z *= np.where(e_i == i, -1.0, 1.0)
                weight_y *= np.where(e_i == i, 1 - y_i, y_i)
                weight_x *= np.where(e_i == i, 1 - y_i, y_i)
        grad_x += np.array(
            interp_function.values[edge_indices],
        ) * weight_x
        grad_y += np.array(
            interp_function.values[edge_indices],
        ) * weight_y
        grad_z += np.array(
            interp_function.values[edge_indices],
        ) * weight_z
    grad_du = np.concatenate(
        (
            grad_x.reshape((-1, 1)),
            grad_y.reshape((-1, 1)),
            grad_z.reshape((-1, 1)),
        ),
        axis=1,
    )
    interp_gradient = (
        pme_gridnumber
        * (grad_du @ inverse_box)
    )
    return interp_gradient


def project_to_grid(
        inverse_box: NDArray[np.float64],
        pme_gridnumber: int,
        points: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Project points onto a PME grid in reciprocal space.  This
    algorithm is identical to that used in method
    'pme_update_grid_index_and_fraction' in OpenMM source code,
    ReferencePME.cpp.

    :param points: The real-space points, in Bohr, to project onto
        the reciprocal-space PME grid.
    :return: The projected points in reciprocal-space, in inverse
        Bohr.
    """
    fractional_points = np.matmul(points, inverse_box)
    floor_points = np.floor(fractional_points)
    decimal_points = (
        (fractional_points - floor_points) * pme_gridnumber
    )
    integer_points = decimal_points.astype(int)
    scaled_grid_points = np.mod(
        integer_points,
        pme_gridnumber,
    ) + (decimal_points - integer_points)
    return scaled_grid_points


@numba.jit(nopython=True, parallel=True, cache=True)
def _compute_reciprocal_exclusions(
        external_grid: NDArray[np.float64],
        indices: NDArray[np.float64],
        positions: NDArray[np.float64],
        charges: NDArray[np.float64],
        exclusions: NDArray[np.float64],
        beta: float,
        box: NDArray[np.float64],
) -> NDArray[np.float64]:
    n = len(exclusions)
    m = len(positions)
    for i in numba.prange(n):
        for j in range(m):
            ssc = 0
            for k in range(3):
                r = exclusions[i, k] - positions[j, k]
                d = box[k, k] * math.floor(r/box[k, k] + 0.5)
                ssc += (r - d)**2
            dr = ssc**0.5
            erf = math.erf(beta * dr)
            if erf <= 1*10**(-6):
                external_grid[indices[i]] -= beta * \
                    charges[j] * 2 * math.pi**(-0.5)
            else:
                external_grid[indices[i]] -= charges[j] * erf / dr
    return external_grid
