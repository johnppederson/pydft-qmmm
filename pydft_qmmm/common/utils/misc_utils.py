"""A module containing helper functions accessed by multiple classes.

Attributes:
    Components: The type corresponding to the energy components
        determined by calculators.
    SELECTORS: Pairs of VMD selection keywords and the corresponding
        attribute and type to check in a system.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from dataclasses import field
from importlib.abc import Loader
from importlib.machinery import ModuleSpec
from importlib.util import find_spec
from importlib.util import LazyLoader
from importlib.util import module_from_spec
from typing import Any
from typing import Callable
from typing import TYPE_CHECKING

import numpy as np

from ..constants import KB

if TYPE_CHECKING:
    from types import ModuleType
    from numpy.typing import NDArray
    from pydft_qmmm.calculators import Calculator

Components = dict[str, Any]


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
        A set of velocities (:math:`\mathrm{\mathring{A}\;fs^{-1}}`) equal in
        number to the set of masses provided.
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


def empty_array() -> NDArray[np.float64]:
    """Factory method for empty arrays.

    Returns:
        An empty array.
    """
    return np.empty(0)


def zero_vector() -> NDArray[np.float64]:
    """Create a zero vector with three dimensions.

    Returns:
        An array with three dimensions of zero magnitude.
    """
    return np.array([0., 0., 0.])


@dataclass
class Results:
    r"""Store the results of a calculation.

    Args:
        energy: The energy (:math:`\mathrm{kJ\;mol^{-1}}`) of a system
            determined by a calculator.
        forces: The forces (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`) on
            a system determined by a calculator.
        componentes: The energy components
            (:math:`\mathrm{kJ\;mol^{-1}}`) of a system determined by a
            calculator.
    """
    energy: float = 0
    forces: NDArray[np.float64] = field(
        default_factory=empty_array,
    )
    components: Components = field(
        default_factory=dict,
    )


def lazy_load(name: str) -> ModuleType:
    """Load a module lazily, not performing execution until necessary.

    Args:
        name: The name of the module to load.

    Returns:
        The module that has been lazily loaded.
    """
    spec = find_spec(name)
    if not isinstance(spec, ModuleSpec):
        raise TypeError()
    if not isinstance(spec.loader, Loader):
        raise TypeError()
    loader = LazyLoader(spec.loader)
    spec.loader = loader
    module = module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


def align_dict(
        dictionary: dict[str, Any],
) -> dict[str, float]:
    """Create a 'flat' version of an energy components dictionary.

    Args:
        dictionary: The components dictionary to flatten.

    Returns:
        A flattened version of the components dictionary.
    """
    flat = {}
    for key, val in dictionary.items():
        flat.update(
            align_dict(val) if isinstance(val, dict) else {key: val},
        )
    return flat


def wrap_positions(
        positions: NDArray[np.float64],
        box: NDArray[np.float64],
        residue_map: dict[int, frozenset[int]],
) -> NDArray[np.float64]:
    r"""Wrap atom positions in accord with PBC.

    Atoms are wrapped to stay inside of the periodic box.  This
    function ensures molecules are not broken up by a periodic
    boundary, since OpenMM electrostatics will be incorrect if atoms
    in a molecule are not on the same side of the periodic box.
    This method currently assumes an isotropic box.

    Args:
        positions: The positions (:math:`\mathrm{\mathring{A}}`) which will be
            wrapped.
        box: The lattice vectors (:math:`\mathrm{\mathring{A}}`) of the box
            containing the system.
        residue_map: Sets of atom indices mapped by residue index.

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


def center_positions(
        positions: NDArray[np.ndarray],
        box: NDArray[np.float64],
        atoms: frozenset[int],
) -> NDArray[np.float64]:
    r"""Center positions about the centroid of a set of atoms.

    Args:
        positions: The positions (:math:`\mathrm{\mathring{A}}`) which will be
            centered.
        box: The lattice vectors (:math:`\mathrm{\mathring{A}}`) of the box
            containing the system.
        atoms: The set of atom indices whose centroid will become the center of
            the box.

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
        positions: The positions (:math:`\mathrm{\mathring{A}}`) which will be
            centered.
        box: The lattice vectors (:math:`\mathrm{\mathring{A}}`) of the box
            containing the system.
        atoms: The set of atom indices whose centroid will become the center of
            the box.
    """
    atoms_reference = atoms_metric(
        positions[sorted(atoms), :],
    )
    region_ii: list[int] = []
    for residue in residue_map.values():
        others = sorted(residue)
        not_atoms = set(atoms).isdisjoint(set(others))
        if not_atoms and others:
            other_reference = other_metric(
                positions[others, :],
            )
            metric_vector = atoms_reference - other_reference
            distance = np.sum(metric_vector**2)**0.5
            if distance < cutoff:
                region_ii.extend(atoms)
    return region_ii


def numerical_gradient(
        calculator: Calculator,
        atoms: frozenset[int],
        dist: int | float = 0.0025,
        component: str | None = None,
) -> NDArray[np.float64]:
    r"""Calculate the numerical energy gradients for a set of atoms.

    Args:
        simulation: A simulation object used for the energy evaluations.
        atoms: The atoms to perform numerical gradients on.
        dist: The displacement for central differencing
            (:math:`\mathrm{\mathring{A}}`).
        component: The component of the energy dictionary to use in
            central differencing calculations.

    Returns:
        The numerical gradients of the energy
        (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`)
    """
    grad = np.zeros((len(atoms), 3))
    for i, atom in enumerate(sorted(atoms)):
        for j in range(3):
            # Perform first finite difference displacement.
            calculator.system.positions[atom, j] += dist
            if component:
                ref_1 = calculator.calculate().components[component]
            else:
                ref_1 = calculator.calculate().energy
            # Perform second finite difference displacement.
            calculator.system.positions[atom, j] -= 2*dist
            ref_0 = calculator.calculate().energy
            if component:
                ref_0 = calculator.calculate().components[component]
            else:
                ref_0 = calculator.calculate().energy
            grad[i, j] = (ref_1 - ref_0) / (2*dist)
            calculator.system.positions[atom, j] += dist
    return grad
