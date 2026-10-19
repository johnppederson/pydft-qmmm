"""Geometric Lennard-Jones mixing and XML NBFix pair overrides.

Class-specific Lorentz mixing and zero-LJ sites are optional. XML sigma and
epsilon values use nm and kJ/mol, as in OpenMM force fields.
"""
import math
import xml.etree.ElementTree as ET
import numpy as np
import openmm as mm
from openmm import unit


def atom_classes(topology, xml_paths):
    types, residues, overrides = {}, {}, {}
    for path in xml_paths:
        root = ET.parse(path).getroot()
        for typ in root.findall('./AtomTypes/Type'):
            types[typ.get('name')] = typ.get('class')
        for res in root.findall('./Residues/Residue'):
            residues[res.get('name')] = {a.get('name'): a.get('type') for a in res.findall('Atom')}
        for pair in root.findall('./NonbondedForce/NBFixPair'):
            key = tuple(sorted((pair.get('class1'), pair.get('class2'))))
            value = (float(pair.get('sigma')), float(pair.get('epsilon')))
            if key in overrides and overrides[key] != value:
                raise ValueError(f'Conflicting pair parameters: {key}')
            overrides[key] = value
    classes = [types[residues[a.residue.name][a.name]] for a in topology.atoms()]
    return classes, overrides


def apply_geometric_lj(system, topology, xml_paths, *,
                       lorentz_classes=(), zero_classes=()):
    """Replace particle LJ terms, retaining Coulomb and 1-4 exceptions.

    Sigma is geometric except for pairs involving ``lorentz_classes``,
    where its arithmetic mean is used. XML NBFixPair entries override both
    sigma and epsilon; ``zero_classes`` then suppresses selected LJ sites.
    Long-range LJ corrections are disabled, matching a truncated LJ model.
    """
    classes, overrides = atom_classes(topology, xml_paths)
    nb, = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)]
    if len(classes) != system.getNumParticles():
        raise ValueError('Topology/particle count mismatch')
    kinds, kind_index, particle_types, lj = [], {}, [], []
    for i, cls in enumerate(classes):
        q, s, e = nb.getParticleParameters(i)
        s = s.value_in_unit(unit.nanometer)
        e = e.value_in_unit(unit.kilojoule_per_mole)
        key = (cls, s, e)
        if key not in kind_index:
            kind_index[key] = len(kinds)
            kinds.append(key)
        particle_types.append(kind_index[key])
        lj.append((s, e))
    size = len(kinds)
    sigma, epsilon = np.zeros((size, size)), np.zeros((size, size))
    for i, (ci, si, ei) in enumerate(kinds):
        for j, (cj, sj, ej) in enumerate(kinds):
            lorentz_pair = ci in lorentz_classes or cj in lorentz_classes
            s = (si + sj) / 2 if lorentz_pair else math.sqrt(si * sj)
            e = math.sqrt(ei * ej)
            s, e = overrides.get(tuple(sorted((ci, cj))), (s, e))
            if ci in zero_classes or cj in zero_classes:
                e = 0.0
            sigma[i, j], epsilon[i, j] = s, e
    force = mm.CustomNonbondedForce(
        '4*epsilon*((sigma/r)^12-(sigma/r)^6);'
        'sigma=sigma_pair(kind1,kind2);epsilon=epsilon_pair(kind1,kind2)')
    force.setName('Geometric LJ with class pair overrides')
    force.addPerParticleParameter('kind')
    force.addTabulatedFunction('sigma_pair', mm.Discrete2DFunction(size, size, sigma.flatten(order='F')))
    force.addTabulatedFunction('epsilon_pair', mm.Discrete2DFunction(size, size, epsilon.flatten(order='F')))
    method = nb.getNonbondedMethod()
    force.setNonbondedMethod(mm.CustomNonbondedForce.NoCutoff if method == nb.NoCutoff else
                             mm.CustomNonbondedForce.CutoffNonPeriodic if method == nb.CutoffNonPeriodic else
                             mm.CustomNonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(nb.getCutoffDistance())
    force.setUseSwitchingFunction(nb.getUseSwitchingFunction())
    if nb.getUseSwitchingFunction():
        force.setSwitchingDistance(nb.getSwitchingDistance())
    force.setUseLongRangeCorrection(False)  # Same truncated LJ treatment as source runs.
    for i, kind in enumerate(particle_types):
        force.addParticle([kind])
        q, s, e = nb.getParticleParameters(i)
        nb.setParticleParameters(i, q, s, 0 * e)
    for i in range(nb.getNumExceptions()):
        p, q, charge, s, e = nb.getExceptionParameters(i)
        force.addExclusion(p, q)
        if e.value_in_unit(unit.kilojoule_per_mole) != 0:
            # Preserve original OPLS 1-4 energy scaling and geometric 1-4 sigma.
            nb.setExceptionParameters(i, p, q, charge, math.sqrt(lj[p][0] * lj[q][0]), e)
    system.addForce(force)
    return system
