"""Geometric Lennard-Jones mixing with XML NBFix pair overrides.

Class-specific Lorentz mixing and zero-LJ sites are optional.  XML sigma
and epsilon values use nm and kJ/mol, as in OpenMM force fields.
"""
from __future__ import annotations

__all__ = ["apply_geometric_lj"]

import math
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import openmm
from openmm import unit

if TYPE_CHECKING:
    from openmm.app import Topology


_CUSTOM_METHODS = {
    openmm.NonbondedForce.NoCutoff: openmm.CustomNonbondedForce.NoCutoff,
    openmm.NonbondedForce.CutoffNonPeriodic: (
        openmm.CustomNonbondedForce.CutoffNonPeriodic
    ),
}


def _atom_classes(
        topology: Topology,
        xml_paths: Iterable[str],
) -> tuple[list[str], dict[tuple[str, str], tuple[float, float]]]:
    """Read atom classes and NBFix pair overrides from XML force fields.

    Args:
        topology: The OpenMM topology whose atoms are classified.
        xml_paths: The XML force field files.

    Returns:
        The atom class of each topology atom, and the (sigma, epsilon)
        override of each sorted pair of classes.

    Raises:
        ValueError: If two files give different parameters for the same
            pair of classes.
    """
    types: dict[str, str] = {}
    residues: dict[str, dict[str, str]] = {}
    overrides: dict[tuple[str, str], tuple[float, float]] = {}
    for path in xml_paths:
        root = ET.parse(path).getroot()
        for atom_type in root.findall("./AtomTypes/Type"):
            types[atom_type.get("name")] = atom_type.get("class")
        for residue in root.findall("./Residues/Residue"):
            residues[residue.get("name")] = {
                atom.get("name"): atom.get("type")
                for atom in residue.findall("Atom")
            }
        for pair in root.findall("./NonbondedForce/NBFixPair"):
            key = tuple(sorted((pair.get("class1"), pair.get("class2"))))
            value = (float(pair.get("sigma")), float(pair.get("epsilon")))
            if key in overrides and overrides[key] != value:
                raise ValueError(f"Conflicting pair parameters: {key}")
            overrides[key] = value
    classes = [
        types[residues[atom.residue.name][atom.name]]
        for atom in topology.atoms()
    ]
    return classes, overrides


def apply_geometric_lj(
        system: openmm.System,
        topology: Topology,
        xml_paths: Iterable[str],
        *,
        lorentz_classes: tuple[str, ...] = (),
        zero_classes: tuple[str, ...] = (),
) -> openmm.System:
    """Replace particle LJ terms, retaining Coulomb and 1-4 exceptions.

    Sigma is geometric except for pairs involving ``lorentz_classes``,
    where its arithmetic mean is used.  XML NBFixPair entries override
    both sigma and epsilon; ``zero_classes`` then suppresses selected LJ
    sites.  Long-range LJ corrections are disabled, matching a truncated
    LJ model.

    Args:
        system: The OpenMM system, modified in place.
        topology: The OpenMM topology of the system.
        xml_paths: The XML force field files that built the system.
        lorentz_classes: Atom classes whose pairs use arithmetic sigma.
        zero_classes: Atom classes whose pairs have zero epsilon.

    Returns:
        The modified OpenMM system.

    Raises:
        ValueError: If the topology and system particle counts differ.
    """
    classes, overrides = _atom_classes(topology, xml_paths)
    nonbonded, = [
        force for force in system.getForces()
        if isinstance(force, openmm.NonbondedForce)
    ]
    if len(classes) != system.getNumParticles():
        raise ValueError("Topology/particle count mismatch")
    # Group particles into kinds sharing a class, sigma, and epsilon.
    kinds: list[tuple[str, float, float]] = []
    kind_index: dict[tuple[str, float, float], int] = {}
    particle_kinds = []
    lj = []
    for i, atom_class in enumerate(classes):
        _, sigma, epsilon = nonbonded.getParticleParameters(i)
        sigma = sigma.value_in_unit(unit.nanometer)
        epsilon = epsilon.value_in_unit(unit.kilojoule_per_mole)
        key = (atom_class, sigma, epsilon)
        if key not in kind_index:
            kind_index[key] = len(kinds)
            kinds.append(key)
        particle_kinds.append(kind_index[key])
        lj.append((sigma, epsilon))
    size = len(kinds)
    sigma_table = np.zeros((size, size))
    epsilon_table = np.zeros((size, size))
    for i, (class_i, sigma_i, epsilon_i) in enumerate(kinds):
        for j, (class_j, sigma_j, epsilon_j) in enumerate(kinds):
            if class_i in lorentz_classes or class_j in lorentz_classes:
                sigma = (sigma_i + sigma_j) / 2
            else:
                sigma = math.sqrt(sigma_i * sigma_j)
            epsilon = math.sqrt(epsilon_i * epsilon_j)
            sigma, epsilon = overrides.get(
                tuple(sorted((class_i, class_j))), (sigma, epsilon),
            )
            if class_i in zero_classes or class_j in zero_classes:
                epsilon = 0.0
            sigma_table[i, j] = sigma
            epsilon_table[i, j] = epsilon
    force = openmm.CustomNonbondedForce(
        "4*epsilon*((sigma/r)^12-(sigma/r)^6);"
        "sigma=sigma_pair(kind1,kind2);"
        "epsilon=epsilon_pair(kind1,kind2)",
    )
    force.setName("Geometric LJ with class pair overrides")
    force.addPerParticleParameter("kind")
    force.addTabulatedFunction(
        "sigma_pair",
        openmm.Discrete2DFunction(
            size, size, sigma_table.flatten(order="F"),
        ),
    )
    force.addTabulatedFunction(
        "epsilon_pair",
        openmm.Discrete2DFunction(
            size, size, epsilon_table.flatten(order="F"),
        ),
    )
    force.setNonbondedMethod(
        _CUSTOM_METHODS.get(
            nonbonded.getNonbondedMethod(),
            openmm.CustomNonbondedForce.CutoffPeriodic,
        ),
    )
    force.setCutoffDistance(nonbonded.getCutoffDistance())
    force.setUseSwitchingFunction(nonbonded.getUseSwitchingFunction())
    if nonbonded.getUseSwitchingFunction():
        force.setSwitchingDistance(nonbonded.getSwitchingDistance())
    # Same truncated LJ treatment as the source runs.
    force.setUseLongRangeCorrection(False)
    for i, kind in enumerate(particle_kinds):
        force.addParticle([kind])
        charge, sigma, epsilon = nonbonded.getParticleParameters(i)
        nonbonded.setParticleParameters(i, charge, sigma, 0 * epsilon)
    for i in range(nonbonded.getNumExceptions()):
        p, q, charge, sigma, epsilon = nonbonded.getExceptionParameters(i)
        force.addExclusion(p, q)
        if epsilon.value_in_unit(unit.kilojoule_per_mole) != 0:
            # Preserve the original OPLS 1-4 energy scaling, with a
            # geometric 1-4 sigma.
            nonbonded.setExceptionParameters(
                i, p, q, charge, math.sqrt(lj[p][0] * lj[q][0]), epsilon,
            )
    system.addForce(force)
    return system
