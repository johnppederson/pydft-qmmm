"""Functionality for building the OpenMM interface.

Attributes:
    SUPPORTED_FORCES: OpenMM force classes which can be processed by
        PyDFT-QMMM currently.
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
    """Build the interface to OpenMM given the settings.

    Args:
        settings: The settings used to build the OpenMM interface.

    Returns:
        The OpenMM interface.
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
    """Build the OpenMM Topology object.

    Args:
        settings: The settings used to build the OpenMM interface.

    Returns:
        The internal representation of system topology for OpenMM.
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
    residue_map = settings.system.residue_map
    for i in residue_map.keys():
        atoms = sorted(residue_map[i])
        residue = topology.addResidue(
            settings.system.residue_names[atoms[0]],
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

    Args:
        settings: The settings used to build the OpenMM interface.
        topology: The OpenMM representation of system topology.

    Returns:
        The internal representation of the system OpenMM, integrating
        the topology and atomic positions.
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

    Args:
        settings: The settings used to build the OpenMM interface.
        modeller: The OpenMM representation of the system.

    Returns:
        The internal representation of the force field for OpenMM.
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

    Args:
        forcefield: The OpenMM representation of the forcefield.
        modeller: The OpenMM representation of the system.

    Returns:
        The internal representation of forces, constraints, and
        particles for OpenMM.
    """
    system = forcefield.createSystem(
        modeller.topology,
        rigidWater=False,
    )
    return system


def _empty_system(settings: MMSettings) -> openmm.System:
    """Build an empty OpenMM System object.

    Args:
        settings: The settings used to build the OpenMM interface.

    Returns:
        An internal representation of forces, constraints, and
        particles in OpenMM for a system with no forces or constraints.
    """
    system = openmm.System()
    for i in range(len(settings.system)):
        system.addParticle(0.)
    return system


def _adjust_forces(settings: MMSettings, system: openmm.System) -> None:
    """Adjust the OpenMM Nonbonded forces with appropriate settings.

    Args:
        settings: The settings used to build the OpenMM interface.
        system: The OpenMM representation of forces, constraints, and
            particles.
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
            if (
                settings.nonbonded_method == "PME"
                and settings.pme_gridnumber
                and settings.pme_alpha
            ):
                force.setPMEParameters(
                    settings.pme_alpha,
                    settings.pme_gridnumber,
                    settings.pme_gridnumber,
                    settings.pme_gridnumber,
                )
        if isinstance(force, openmm.CustomNonbondedForce):
            force.setNonbondedMethod(
                openmm.CustomNonbondedForce.CutoffPeriodic,
            )
            force.setCutoffDistance(settings.nonbonded_cutoff / 10.)


def _build_context(
        settings: MMSettings, system: openmm.System, modeller: Modeller,
) -> openmm.Context:
    """Build the OpenMM Context object.

    Args:
        settings: The settings used to build the OpenMM interface.
        system: The OpenMM representation of forces, constraints, and
            particles.
        modeller: The OpenMM representation of the system.

    Returns:
        The OpenMM machinery required to perform energy and force
        calculations, containing the System object and the specific
        platform to use, which is currently just the CPU platform.
    """
    integrator = openmm.VerletIntegrator(1. * femtosecond)
    platform = openmm.Platform.getPlatformByName("CPU")
    context = openmm.Context(system, integrator, platform)
    context.setPositions(modeller.positions)
    return context
