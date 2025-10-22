"""Helper functions accessed by multiple classes.
"""
from __future__ import annotations

__all__ = [
    "check_array",
    "generate_velocities",
    "wrap_positions",
    "center_positions",
    "residue_partition",
    "numerical_gradient",
]

from typing import TYPE_CHECKING

import numpy as np

from .constants import KB

if TYPE_CHECKING:
    from types import MappingProxyType
    from collections.abc import Callable
    from numpy.typing import NDArray
    from pydft_qmmm.calculators import Calculator


def check_array(value: NDArray[np.float64]) -> bool:
    """Check if an array has any NaN or Inf vlaues.

    Args:
        value: The array to evaluate.

    Returns:
        Whether the array has any NaN or Inf values.
    """
    return bool(np.isnan(value).any() or np.isinf(value).any())


def generate_velocities(
        masses: NDArray[np.float64],
        temperature: float | int,
        seed: int | None = None,
) -> NDArray[np.float64]:
    r"""Generate velocities with the Maxwell-Boltzmann distribution.

    Args:
        masses: The masses (:math:`\mathrm{AMU}`) of particles.
        temperature: The temperature (:math:`\mathrm{K}`) of the system.
        seed: A seed for the random number generator.

    Returns:
        A set of velocities (:math:`\mathrm{\mathring{A}\;fs^{-1}}`)
        equal in number to the set of masses provided.
    """
    avg_ke = temperature * KB
    masses = masses.reshape((-1, 1)) * (10**-3)
    if seed:
        np.random.seed(seed)
    z = np.random.standard_normal((len(masses), 3))
    momenta = z * np.sqrt(avg_ke * masses)
    velocities = (momenta / masses) * (10**-5)
    zero_mass = np.where(masses == 0)
    velocities[zero_mass, :] = np.array([0., 0., 0.])
    return velocities


def wrap_positions(
        positions: NDArray[np.float64],
        box: NDArray[np.float64],
        residue_map: MappingProxyType[int, frozenset[int]],
) -> NDArray[np.float64]:
    r"""Wrap atom positions in accord with PBC.

    Atoms are wrapped to stay inside of the periodic box.  This
    function ensures molecules are not broken up by a periodic
    boundary, since OpenMM electrostatics will be incorrect if atoms
    in a molecule are not on the same side of the periodic box.

    Args:
        positions: The positions (:math:`\mathrm{\mathring{A}}`) to
            wrap.
        box: The lattice vectors (:math:`\mathrm{\mathring{A}}`) of
            the box containing the system.
        residue_map: Sets of atom indices indexed by residue index.

    Returns:
        The new wrapped positions of the system.
    """
    inverse_box = np.linalg.inv(box)
    new_positions = np.zeros(positions.shape)
    for residue in residue_map.values():
        atoms = sorted(residue)
        residue_positions = positions[atoms, :]
        residue_centroid = np.average(
            residue_positions,
            axis=0,
        ).reshape((3, 1))
        inverse_centroid = inverse_box @ residue_centroid
        mask = np.floor(inverse_centroid)
        diff = (-box @ mask).reshape((-1, 3))
        temp = residue_positions + diff[:, np.newaxis, :]
        new_positions[atoms] = temp.reshape((len(atoms), 3))
    return new_positions


def center_positions(
        positions: NDArray[np.float64],
        box: NDArray[np.float64],
        atoms: frozenset[int],
) -> NDArray[np.float64]:
    r"""Center positions about the centroid of a set of atoms.

    Args:
        positions: The positions (:math:`\mathrm{\mathring{A}}`) to
            center.
        box: The lattice vectors (:math:`\mathrm{\mathring{A}}`) of
            the box containing the system.
        atoms: The set of atom indices whose centroid will become the
            center of the box.

    Returns:
        The new centered positions of the system.
    """
    center = 0.5*box.sum(axis=0)
    centroid = np.average(positions[list(atoms), :], axis=0)
    differential = center - centroid
    new_positions = positions + differential
    return new_positions


def residue_partition(
        atoms: frozenset[int],
        positions: NDArray[np.float64],
        residue_map: dict[int, frozenset[int]],
        atoms_metric: Callable[[NDArray], float],
        other_metric: Callable[[NDArray], float],
        cutoff: float | int,
) -> list[int]:
    r"""Perform the residue-wise system partitioning.

    Args:
        atoms: The set of atom indices whose metric will be used to
            determine the partition.
        positions: The system positions (:math:`\mathrm{\mathring{A}}`).
        residue_map: Sets of atom indices indexed by residue index.
        atoms_metric: A function to apply to a residue's positions to
            generate a single relevant coordinate.  This metric is
            applied to the provided set of atoms.
        other_metric: A function to apply to a residue's positions to
            generate a single relevant coordinate.  This metric is
            applied to all residues in the system.
        cutoff: The cutoff distance (:math:`\mathrm{\mathring{A}}`) to
            apply in the partition.
    """
    atoms_reference = atoms_metric(
        positions[sorted(atoms), :],
    )
    region_ii: list[int] = []
    for residue in residue_map.values():
        others = sorted(residue)
        not_atoms = atoms.isdisjoint(residue)
        if not_atoms and others:
            other_reference = other_metric(
                positions[others, :],
            )
            metric_vector = atoms_reference - other_reference
            distance = np.sum(metric_vector**2)**0.5
            if distance < cutoff:
                region_ii.extend(others)
    return region_ii


def numerical_gradient(
        calculator: Calculator,
        atoms: frozenset[int],
        dist: int | float = 0.0025,
        components: list[str] | None = None,
) -> NDArray[np.float64]:
    r"""Calculate the numerical energy gradients for a set of atoms.

    Args:
        calculator: A calculator object used for the energy evaluations.
        atoms: The atoms for which to perform a numerical gradient
            calculation.
        dist: The displacement for central differencing
            (:math:`\mathrm{\mathring{A}}`).
        component: The component of the energy dictionary to use in
            central differencing calculations.

    Returns:
        The numerical gradients of the energy
        (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`).
    """
    grad = np.zeros((len(atoms), 3))
    for i, atom in enumerate(sorted(atoms)):
        for j in range(3):
            # Perform first finite difference displacement.
            calculator.system.positions[atom, j] += dist
            if components is not None:
                ref_1 = 0
                comps = calculator.calculate().components
                for key in components:
                    ref_1 += comps[key]
            else:
                ref_1 = calculator.calculate().energy
            # Perform second finite difference displacement.
            calculator.system.positions[atom, j] -= 2*dist
            ref_0 = calculator.calculate().energy
            if components is not None:
                ref_0 = 0
                comps = calculator.calculate().components
                for key in components:
                    ref_0 += comps[key]
            else:
                ref_0 = calculator.calculate().energy
            grad[i, j] = (ref_1 - ref_0) / (2*dist)
            calculator.system.positions[atom, j] += dist
    return grad
