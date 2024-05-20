#! /usr/bin/env python3
"""A module to define the :class:`OpenMMInterface` class.
"""
from __future__ import annotations

import openmm
from openmm.app import Element
from openmm.app import ForceField
from openmm.app import Modeller
from openmm.app import Topology
from simtk.unit import angstrom
from simtk.unit import femtosecond
from simtk.unit import nanometer

from ..interface import MMSettings
from .openmm_interface import OpenMMInterface


SUPPORTED_FORCES = [
    openmm.CMMotionRemover,
    openmm.CustomNonbondedForce,
    openmm.HarmonicAngleForce,
    openmm.HarmonicBondForce,
    openmm.NonbondedForce,
    openmm.PeriodicTorsionForce,
    openmm.RBTorsionForce,
]


def openmm_interface_factory(settings: MMSettings) -> OpenMMInterface:
    """A function which constructs the :class:`OpenMMInterface` for a
    standalone MM system.

    :param settings: The :class:`MMSettings` object to build the
        standalone MM system interface from.
    :return: The :class:`OpenMMInterface` for the standalone MM system.
    """
    topology = _build_topology(settings)
    modeller = _build_modeller(settings, topology)
    forcefield = _build_forcefield(settings, modeller)
    system = _build_system(forcefield, modeller)
    _adjust_forces(settings, system)
    base_context = _build_context(settings, system, modeller)
    ixn_context = _build_context(settings, _empty_system(settings), modeller)
    wrapper = OpenMMInterface(settings, base_context, ixn_context)
    # Register observer functions.
    settings.system.charges.register_notifier(wrapper.update_charges)
    settings.system.positions.register_notifier(wrapper.update_positions)
    settings.system.box.register_notifier(wrapper.update_box)
    settings.system.subsystems.register_notifier(wrapper.update_subsystems)
    return wrapper


def _build_topology(settings: MMSettings) -> Topology:
    """Build the OpenMM PDBFile object.

    :param settings: The :class:`MMSettings` object to build from.
    :return: The OpenMM PDBFile object built from the given settings.
    """
    if (x := settings.topology_file):
        if isinstance(x, list):
            for fh in x:
                Topology.loadBondDefinitions(fh)
        elif isinstance(x, str):
            Topology.loadBondDefinitions(x)
        else:
            raise TypeError("...")
    topology = Topology()
    chain = topology.addChain()
    molecule_map = settings.system.molecule_map
    for i in molecule_map.keys():
        atoms = list(molecule_map[i])
        atoms.sort()
        residue = topology.addResidue(
            settings.system.molecule_names[atoms[0]],
            chain,
        )
        for j in atoms:
            _ = topology.addAtom(
                settings.system.names[j],
                Element.getBySymbol(settings.system.elements[j]),
                residue,
            )
    topology.createStandardBonds()
    return topology


def _build_modeller(settings: MMSettings, topology: Topology) -> Modeller:
    """Build the OpenMM Modeller object.

    :param pdb: The OpenMM PDBFile object to build from.
    :return: The OpenMM Modeller object built from the given pdb.
    """
    temp = []
    for position in settings.system.positions:
        temp.append(
            openmm.Vec3(
                position[0],
                position[1],
                position[2],
            ) * angstrom,
        )
    modeller = Modeller(topology, temp)
    return modeller


def _build_forcefield(settings: MMSettings, modeller: Modeller) -> ForceField:
    """Build the OpenMM ForceField object.

    :param settings: The :class:`MMSettings` object to build from.
    :param modeller: The OpenMM Modeller object to build from.
    :return: The OpenMM ForceField object built from the given settings
        and modeller.
    """
    if isinstance(settings.forcefield_file, str):
        forcefield = ForceField(settings.forcefield_file)
    elif isinstance(settings.forcefield_file, list):
        forcefield = ForceField(*settings.forcefield_file)
    else:
        raise TypeError("...")
    # modeller.addExtraParticles(forcefield)
    return forcefield


def _build_system(
        forcefield: ForceField, modeller: Modeller,
) -> openmm.System:
    """Build the OpenMM System object.

    :param forcefield: The OpenMM ForceField object to build from.
    :param modeller: The OpenMM Modeller object to build from.
    :return: The OpenMM System object built from the given settings,
        forcefield, and modeller.
    """
    system = forcefield.createSystem(
        modeller.topology,
        rigidWater=False,
    )
    return system


def _empty_system(settings: MMSettings) -> openmm.System:
    system = openmm.System()
    for i in range(len(settings.system)):
        system.addParticle(0.)
    return system


def _adjust_forces(settings: MMSettings, system: openmm.System) -> None:
    """Adjust the OpenMM Nonbonded forces.

    :param settings: The :class:`MMSettings` object to adjust with.
    :param system: The OpenMM System object to adjust.
    """
    temp = []
    for box_vec in settings.system.box:
        temp.append(
            openmm.Vec3(
                box_vec[0] / 10.,
                box_vec[1] / 10.,
                box_vec[2] / 10.,
            ) * nanometer,
        )
    system.setDefaultPeriodicBoxVectors(*temp)
    for i, force in enumerate(system.getForces()):
        force.setForceGroup(i)
        if type(force) not in SUPPORTED_FORCES:
            print(type(force))
            raise Exception
        if isinstance(force, openmm.NonbondedForce):
            force.setNonbondedMethod(openmm.NonbondedForce.PME)
            force.setCutoffDistance(settings.nonbonded_cutoff / 10.)
        if isinstance(force, openmm.CustomNonbondedForce):
            force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
            force.setCutoffDistance(settings.nonbonded_cutoff / 10.)


def _build_context(
        settings: MMSettings, system: openmm.System, modeller: Modeller,
) -> openmm.Context:
    """Build the OpenMM Context object.

    :param settings: The :class:`MMSettings` object to build from.
    :param system: The OpenMM System object to build from.
    :param modeller: The OpenMM Modeller object to build from.
    :return: The OpenMM System object built from the given settings,
        system, and modeller.
    """
    integrator = openmm.VerletIntegrator(1. * femtosecond)
    platform = openmm.Platform.getPlatformByName("CPU")
    context = openmm.Context(system, integrator, platform)
    context.setPositions(modeller.positions)
    return context
