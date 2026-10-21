"""Neutral Yeh–Berkowitz slab terms (eV, Angstrom, elementary charge).

The volume is the full MM cell including vacuum. Coordinates must be
continuous across the slab, not wrapped independently across z faces.
These terms approximate open-z electrostatics; vacuum convergence remains
necessary. No QM self term is added when VASP truncates that interaction.
"""
from __future__ import annotations

import numpy as np

from .grid_potential import EPS0

# Allow small integration residuals in the VASP grid; never renormalize it.
QM_NEUTRALITY_TOLERANCE = 1e-4
# VASP's grid residual grows with system size (2.3e-4 e for 3980 electrons).
QM_NEUTRALITY_RELATIVE = 1e-7


def coefficient(cell):
    cell = np.asarray(cell, dtype=float)
    if (
        cell.shape != (3, 3) or not np.isfinite(cell).all()
        or not np.allclose(cell, np.diag(np.diag(cell)))
        or np.any(np.diag(cell) <= 0)
    ):
        raise ValueError(
            'Slab correction requires a positive orthorhombic cell',
        )
    return 1.0 / (2.0 * EPS0 * np.linalg.det(cell))


def neutral_dipole(positions, charges):
    positions, charges = np.asarray(positions), np.asarray(charges)
    if (
        positions.shape != (charges.size, 3) or charges.ndim != 1
        or not np.isfinite(positions).all() or not np.isfinite(charges).all()
    ):
        raise ValueError('Invalid slab coordinates or charges')
    if abs(charges.sum()) > 1e-7:
        raise ValueError(
            'Slab correction currently requires neutral MM sources',
        )
    return float(charges @ positions[:, 2])


def mm_correction(positions, charges, cell):
    c = coefficient(cell)
    moment = neutral_dipole(positions, charges)
    forces = np.zeros(np.shape(positions), dtype=float)
    forces[:, 2] = -2 * c * moment * np.asarray(charges)
    return c * moment**2, forces


def grid_z(shape, cell, origin):
    """Unwrapped global z at the QM grid nodes, broadcast over x and y."""
    z = origin[2] + np.arange(shape[2]) * cell[2, 2] / shape[2]
    return z[None, None, :]


def qm_dipole(density, positions, valence, cell, origin):
    """Use VASP's final positive electron density and analytic ions."""
    density = np.asarray(density)
    electrons = density / density.size
    tolerance = max(
        QM_NEUTRALITY_TOLERANCE, QM_NEUTRALITY_RELATIVE * np.sum(valence),
    )
    residual = abs(electrons.sum() - np.sum(valence))
    if not np.isfinite(density).all() or residual > tolerance:
        raise ValueError(
            f'Slab correction requires a neutral QM density: '
            f'electrons={electrons.sum():.12g}, ions={np.sum(valence):.12g}',
        )
    return float(
        np.dot(valence, positions[:, 2] + origin[2])
        - np.sum(electrons * grid_z(density.shape, cell, origin)),
    )
