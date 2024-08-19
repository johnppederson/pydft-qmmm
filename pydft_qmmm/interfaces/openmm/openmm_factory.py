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
from simtk.unit import daltons
from simtk.unit import elementary_charge
from simtk.unit import femtosecond
from simtk.unit import nanometer

from ..interface import MMSettings
from .openmm_interface import OpenMMInterface
from pydft_qmmm.common import lazy_load


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
    box_vectors = []
    for box_vec in settings.system.box.T:
        box_vectors.append(
            openmm.Vec3(
                box_vec[0] / 10.,
                box_vec[1] / 10.,
                box_vec[2] / 10.,
            ) * nanometer,
        )
    if all(x := [fh.endswith(".xml") for fh in settings.forcefield]):
        topology = _build_topology(settings)
        modeller = _build_modeller(settings, topology)
        forcefield = _build_forcefield(settings, modeller)
        system = _build_system(forcefield, modeller)
    elif not any(x):
        parmed = lazy_load("parmed")
        forcefield = settings.forcefield.copy()
        if len(x) > 1:
            # Assuming a set of CHARMM parameter files and a psf file.
            mask = [fh.endswith(".psf") for fh in forcefield]
            psf = forcefield.pop(mask.index(1))
            struct = parmed.load_file(psf)
            params = parmed.charmm.CharmmParameterSet(*forcefield)
            struct.box_vectors = box_vectors
            system = struct.createSystem(params)
            topology = struct.topology
        else:
            # Assuming the file is a GROMACS top and AMBER prmtop files
            struct = parmed.load_file(*forcefield)
            struct.box_vectors = box_vectors
            system = struct.createSystem()
            topology = struct.topology
        modeller = _build_modeller(settings, topology)
    else:
        raise OSError(
            (
                "Both FF XML and non-XML files have been provided as the "
                "forcefield data to the MM interface factory.  Mixing of "
                "forcefield file formats is not currently supported."
            ),
        )
    system.setDefaultPeriodicBoxVectors(*box_vectors)
    _adjust_forces(settings, system)
    _adjust_system(settings, system)
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
    for fh in settings.forcefield:
        Topology.loadBondDefinitions(fh)
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
    forcefield = ForceField(*settings.forcefield)
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
    for i, force in enumerate(system.getForces()):
        force.setForceGroup(i)
        if type(force) not in SUPPORTED_FORCES:
            raise ValueError(f"{type(force)}")
        if isinstance(force, openmm.NonbondedForce):
            force.setNonbondedMethod(openmm.NonbondedForce.PME)
            force.setCutoffDistance(settings.nonbonded_cutoff / 10.)
            if (
                settings.nonbonded_method == "PME"
                and (settings.pme_gridnumber is not None)
                and settings.pme_alpha
            ):
                force.setPMEParameters(
                    settings.pme_alpha,
                    settings.pme_gridnumber[0],
                    settings.pme_gridnumber[1],
                    settings.pme_gridnumber[2],
                )
        if isinstance(force, openmm.CustomNonbondedForce):
            force.setNonbondedMethod(
                openmm.CustomNonbondedForce.CutoffPeriodic,
            )
            force.setCutoffDistance(settings.nonbonded_cutoff / 10.)


def _adjust_system(settings: MMSettings, system: openmm.System) -> None:
    """Replace system masses and charges with those from the forcefield.

    Args:
        settings: The settings used to build the OpenMM interface.
        system: The OpenMM representation of forces, constraints, and
            particles.
    """
    masses = []
    charges = []
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            for atom in range(system.getNumParticles()):
                masses.append(system.getParticleMass(atom) / daltons)
                q, _, _ = force.getParticleParameters(
                    atom,
                )
                charges.append(q / elementary_charge)
    settings.system.masses[:] = masses
    settings.system.charges[:] = charges


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
