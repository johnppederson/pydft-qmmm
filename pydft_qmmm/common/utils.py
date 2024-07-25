"""A module containing helper functions accessed by multiple classes.

Attributes:
    Components: The type corresponding to the energy components
        determined by calculators.
    SELECTORS: Pairs of VMD selection keywords and the corresponding
        attribute and type to check in a system.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from importlib.abc import Loader
from importlib.machinery import ModuleSpec
from importlib.util import find_spec
from importlib.util import LazyLoader
from importlib.util import module_from_spec
from typing import Any
from typing import Callable
from typing import TYPE_CHECKING

import numpy as np

from .units import KB

if TYPE_CHECKING:
    from types import ModuleType
    from numpy.typing import NDArray
    from pydft_qmmm import System
    from pydft_qmmm.calculators import Calculator

Components = dict[str, Any]


class TheoryLevel(Enum):
    """Enumeration of the different levels of theory.
    """
    NO = "No level of theory (a default)."
    QM = "The quantum mechanical (DFT) level of theory."
    MM = "The molecular mechanical (forcefield) level of theory."


class Subsystem(Enum):
    """Enumeration of the regions of the system.
    """
    NULL = "NULL"
    I = "I"
    II = "II"
    III = "III"


SELECTORS = {
    "element": ("elements", str),
    "atom": ("atoms", int),
    "name": ("names", str),
    "residue": ("residues", int),
    "resid": ("residues", int),
    "resname": ("residue_names", str),
    "subsystem": ("subsystems", Subsystem),
}


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
    return velocities


def empty_array() -> NDArray[np.float64]:
    """Factory method for empty arrays.

    Returns:
        An empty array.
    """
    return np.empty(0)


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


def decompose(text: str) -> list[str]:
    """Decompose an atom selection query into meaningful components.

    Args:
        text: The atom selection query.

    Returns:
        The atom selection query broken into meaningful parts,
        demarcated by keywords.
    """
    line = [a.strip() for a in re.split(r"(not|or|and|\(|\))", text)]
    while "" in line:
        line.remove("")
    return line


def evaluate(text: str, system: System) -> frozenset[int]:
    """Evaluate a part of an atom selection query.

    Args:
        text: A single contained statement from an atom selection query.
        system: The system whose atoms will be selected by evaluating
            a single query statement.

    Returns:
        The set of atom indices selected by the query statement.
    """
    line = text.split(" ")
    category = SELECTORS[line[0].lower()]
    if " ".join(line).lower().startswith("atom name"):
        category = SELECTORS["name"]
        del line[1]
    elif " ".join(line).lower().startswith("residue name"):
        category = SELECTORS["resname"]
        del line[1]
    population = getattr(system, category[0])
    ret: frozenset[int] = frozenset({})
    for string in line[1:]:
        value = category[1](string)
        indices = {i for i, x in enumerate(population) if x == value}
        ret = ret | frozenset(indices)
    return ret


def parens_slice(line: list[str], start: int) -> slice:
    """Find the slice of a query within parentheses.

    Args:
        line: The atom selection query, broken into meaningful
            components.
        start: The index of the line where the statement within
            parentheses begins.

    Returns:
        The slice whose start and stop corresponds to the phrase
        contained by parentheses.
    """
    flag = True
    count = 1
    index = start
    while flag:
        if line[index] == "(":
            count += 1
        if line[index] == ")":
            count -= 1
        if count == 0:
            stop = index
            flag = False
        index += 1
    return slice(start, stop)


def not_slice(line: list[str], start: int) -> slice:
    """Find the slice of a query modified by the 'not' keyword.

    Args:
        line: The atom selection query, broken into meaningful
            components.
        start: The index of the line where the statement modified by the
            'not' keyword begins.

    Returns:
        The slice whose start and stop corresponds to the phrase
        modified by the 'not' keyword.
    """
    flag = True
    count = 0
    index = start
    while flag:
        if line[index] == "(":
            count += 1
        if line[index] == ")":
            count -= 1
        if count == 0:
            stop = index + 1
            flag = False
        index += 1
    return slice(start, stop)


def and_slice(line: list[str], start: int) -> slice:
    """Find the slice of a query modified by the 'and' keyword.

    Args:
        line: The atom selection query, broken into meaningful
            components.
        start: The index of the line where the statement modified by the
            'and' keyword begins.

    Returns:
        The slice whose start and stop corresponds to the phrase
        modified by the 'and' keyword.
    """
    flag = True
    count = 0
    index = start
    while flag:
        if line[index] == "(":
            count += 1
        if line[index] == ")":
            count -= 1
        if count == 0 and line[index] != "not":
            stop = index + 1
            flag = False
        index += 1
    return slice(start, stop)


def or_slice(line: list[str], start: int) -> slice:
    """Find the slice of a query modified by the 'or' keyword.

    Args:
        line: The atom selection query, broken into meaningful
            components.
        start: The index of the line where the statement modified by the
            'or' keyword begins.

    Returns:
        The slice whose start and stop corresponds to the phrase
        modified by the 'or' keyword.
    """
    flag = True
    count = 0
    index = start
    while flag:
        if line[index] == "(":
            count += 1
        if line[index] == ")":
            count -= 1
        if index < len(line) - 1:
            if line[index+1] == "and":
                count += 1
        if index >= 1:
            if line[index-1] == "and":
                count -= 1
        if count == 0 and line[index] != "not":
            stop = index + 1
            flag = False
        index += 1
    return slice(start, stop)


def interpret(line: list[str], system: System) -> frozenset[int]:
    """Interpret a line of atom selection query language.

    Args:
        line: The atom selection query, broken into meaningful
            components.
        system: The system whose atoms will be selected by interpreting
            the selection query.

    Returns:
        The set of atom indices selected by the query.

    .. note:: Based on the VMD atom selection rules.
    """
    # Precedence: () > not > and > or
    full = frozenset(range(len(system)))
    selection: frozenset[int] = frozenset({})
    count = 0
    while count < len(line):
        entry = line[count]
        if entry == "all":
            selection = selection | full
        elif entry == "none":
            selection = selection | frozenset({})
        elif entry.split(" ")[0].lower() in SELECTORS:
            selection = selection | evaluate(entry, system)
        elif entry == "(":
            indices = parens_slice(line, count + 1)
            selection = selection | interpret(line[indices], system)
            count = indices.stop
        elif entry == "not":
            indices = not_slice(line, count + 1)
            selection = selection | (full - interpret(line[indices], system))
            count = indices.stop
        elif entry == "and":
            indices = and_slice(line, count + 1)
            selection = selection & interpret(line[indices], system)
            count = indices.stop
        elif entry == "or":
            indices = or_slice(line, count + 1)
            selection = selection | interpret(line[indices], system)
            count = indices.stop
        else:
            print(f"{entry = }")
            raise ValueError
        count += 1
    return selection


def compute_least_mirror(
        i_vector: NDArray[np.float64],
        j_vector: NDArray[np.float64],
        box: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Calculate the least mirror vector.

    Args:
        i_vector: The position vector (:math:`\mathrm{\mathring{A}}`).
        j_vector: The reference vector (:math:`\mathrm{\mathring{A}}`).
        box: The lattice vectors (:math:`\mathrm{\mathring{A}}`) of an arbitrary
            triclinic box.

    Returns:
        Returns the least mirror coordinates of i_vector with respect to
        j_vector given a set of lattice vectors from a periodic
        triclinic system.
    """
    r_vector = i_vector - j_vector
    r_vector -= box[2] * np.floor(r_vector[2]/box[2][2] + 0.5)
    r_vector -= box[1] * np.floor(r_vector[1]/box[1][1] + 0.5)
    r_vector -= box[0] * np.floor(r_vector[0]/box[0][0] + 0.5)
    return r_vector


def compute_lattice_constants(
        box: NDArray[np.float64],
) -> tuple[float, ...]:
    r"""Calculate the length and angle constants from lattice vectors.

    Returns the lattice constants a, b, c, alpha, beta, and gamma using
    a set of box vectors for a periodic triclinic system.

    Args:
        box: The lattice vectors (:math:`\mathrm{\mathring{A}}`) of an arbitrary
            triclinic box.

    Returns:
        The characteristic lengths (:math:`\mathrm{\mathring{A}}`) and angles
        (:math:`\mathrm{\degree}`) of an arbitrary triclinic box.
    """
    vec_a = box[:, 0]
    vec_b = box[:, 1]
    vec_c = box[:, 2]
    len_a = np.linalg.norm(vec_a)
    len_b = np.linalg.norm(vec_b)
    len_c = np.linalg.norm(vec_c)
    alpha = 180*np.arccos(np.dot(vec_b, vec_c)/len_b/len_c)/np.pi
    beta = 180*np.arccos(np.dot(vec_a, vec_c)/len_a/len_c)/np.pi
    gamma = 180*np.arccos(np.dot(vec_a, vec_b)/len_a/len_b)/np.pi
    return tuple(float(x) for x in (len_a, len_b, len_c, alpha, beta, gamma))


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
        residue_centroid = np.average(residue_positions, axis=0)
        inverse_centroid = residue_centroid @ inverse_box
        mask = np.floor(inverse_centroid)
        diff = (-mask @ box).reshape((-1, 3))
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
