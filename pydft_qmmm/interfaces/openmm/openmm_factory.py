"""Functionality for building the OpenMM interface.

Attributes:
    NEEDS_CUTOFF: OpenMM nonbonded methods that require a cutoff.
    PERIODIC: OpenMM nonbonded methods that are periodic.
    SUPPORTED_FORCES: OpenMM force classes which can be processed by
        PyDFT-QMMM currently.
"""
from __future__ import annotations

__all__ = ["openmm_interface_factory"]

from typing import TYPE_CHECKING

import numpy as np
import openmm
import openmm.app
import openmm.unit

from pydft_qmmm.utils import DependencyImportError

from . import openmm_interface

if TYPE_CHECKING:
    from pydft_qmmm import System

NEEDS_CUTOFF = ("PME", "EWALD", "CUTOFFPERIODIC", "CUTOFFNONPERIODIC")
PERIODIC = ("PME", "EWALD", "CUTOFFPERIODIC")
SUPPORTED_FORCES = (
    openmm.CMMotionRemover,
    openmm.CustomNonbondedForce,
    openmm.HarmonicAngleForce,
    openmm.HarmonicBondForce,
    openmm.NonbondedForce,
    openmm.PeriodicTorsionForce,
    openmm.RBTorsionForce,
)


def openmm_interface_factory(
        system: System,
        /,
        forcefield: list[str] | str,
        nonbonded_method: str = "PME",
        nonbonded_cutoff: float | int = 14.,
        pme_gridnumber: int | tuple[int, int, int] | None = None,
        pme_alpha: float | int | None = None,
) -> openmm_interface.OpenMMPotential:
    r"""Build the interface to OpenMM.

    Args:
        system: The system which will be tied to the OpenMM interface.
        forcefield: The files containing forcefield  and topology
            data for the system.
        nonbonded_method: The method for treating non-bonded
            interactions in OpenMM.
        nonbonded_cutoff: The distance
            (:math:`\mathrm{\mathring(A)^{-1}}`) at which to truncate
            close-range non-bonded interactions.
        pme_gridnumber: The number of grid points to include along each
            lattice edge in PME summation.
        pme_alpha: The Gaussian width parameter in Ewald summation
            (:math:`\mathrm{nm^{-1}}`).

    Returns:
        The OpenMM interface.
    """
    if isinstance(forcefield, str):
        forcefield = [forcefield]
    if isinstance(pme_gridnumber, int):
        pme_gridnumber = (pme_gridnumber,) * 3
    omm_box = [openmm.Vec3(*x)*openmm.unit.angstrom for x in system.box.T]
    if all(x := [fh.endswith(".xml") for fh in forcefield]):
        omm_topology = _build_omm_topology(system, forcefield)
        omm_topology.setPeriodicBoxVectors(omm_box)
        omm_modeller = _build_omm_modeller(system, omm_topology)
        omm_forcefield = _build_omm_forcefield(forcefield, omm_modeller)
        base_system = _build_omm_system(omm_forcefield, omm_modeller)
    elif not any(x):
        try:
            import parmed
        except ImportError:
            raise DependencyImportError(
                "ParmEd",
                "parsing AMBER, CHARMM, or GROMACS input files",
                "https://github.com/ParmEd/ParmEd",
            )
        if len(x) > 1:
            # Assuming a set of CHARMM parameter files and a psf file.
            mask = [fh.endswith(".psf") for fh in forcefield]
            psf = forcefield.pop(mask.index(True))
            struct = parmed.load_file(psf)
            params = parmed.charmm.CharmmParameterSet(*forcefield)
            struct.box_vectors = omm_box
            base_system = struct.createSystem(params)
            omm_topology = struct.topology
        else:
            # Assuming the file is a GROMACS top and AMBER prmtop files
            struct = parmed.load_file(*forcefield)
            struct.box_vectors = omm_box
            base_system = struct.createSystem()
            omm_topology = struct.topology
        omm_modeller = _build_omm_modeller(system, omm_topology)
    else:
        raise OSError(
            (
                "Both FF XML and non-XML files have been provided as the "
                "forcefield data to the MM interface factory.  Mixing of "
                "forcefield file formats is not currently supported."
            ),
        )
    if nonbonded_method.upper() in PERIODIC:
        base_system.setDefaultPeriodicBoxVectors(*omm_box)
    _adjust_omm_forces(
        nonbonded_method,
        nonbonded_cutoff,
        pme_gridnumber,
        pme_alpha,
        base_system,
    )
    _adjust_system(system, base_system)
    aux_system = _empty_omm_system(system)
    base_context = _build_omm_context(base_system, omm_modeller)
    aux_context = _build_omm_context(aux_system, omm_modeller)
    wrapper = openmm_interface.OpenMMPotential(
        system,
        base_context=base_context,
        aux_context=aux_context,
        base_force_mask=np.ones(system.positions.shape),
        aux_energy_group_force_mask=np.ones(system.positions.shape),
        aux_forces_group_force_mask=np.ones(system.positions.shape),
    )
    return wrapper


def _build_omm_topology(
        system: System,
        forcefield: list[str],
) -> openmm.app.Topology:
    """Build the OpenMM Topology object.

    Args:
        system: The system which will be tied to the OpenMM interface.
        forcefield: The files containing forcefield  and topology
            data for the system.

    Returns:
        The internal representation of system topology for OpenMM.
    """
    for fh in forcefield:
        openmm.app.Topology.loadBondDefinitions(fh)
    omm_topology = openmm.app.Topology()
    chains = {x: omm_topology.addChain(x) for x in np.unique(system.chains)}
    residue_map = system.residue_map
    for i in residue_map.keys():
        atoms = sorted(residue_map[i])
        residue = omm_topology.addResidue(
            system.residue_names[atoms[0]],
            chains[system.chains[atoms[0]]],
        )
        for j in atoms:
            _ = omm_topology.addAtom(
                system.names[j],
                openmm.app.Element.getBySymbol(system.elements[j]),
                residue,
            )
    omm_topology.createStandardBonds()
    return omm_topology


def _build_omm_modeller(
        system: System,
        omm_topology: openmm.app.Topology,
) -> openmm.app.Modeller:
    """Build the OpenMM Modeller object.

    Args:
        system: The system which will be tied to the OpenMM interface.
        omm_topology: The OpenMM representation of system topology.

    Returns:
        The internal representation of the system OpenMM, integrating
        the topology and atomic positions.
    """
    omm_pos = [openmm.Vec3(*x)*openmm.unit.angstrom for x in system.positions]
    omm_modeller = openmm.app.Modeller(omm_topology, omm_pos)
    return omm_modeller


def _build_omm_forcefield(
        forcefield: list[str],
        omm_modeller: openmm.app.Modeller,
) -> openmm.app.ForceField:
    """Build the OpenMM ForceField object.

    Args:
        forcefield: The files containing forcefield  and topology
            data for the system.
        omm_modeller: The OpenMM representation of the system.

    Returns:
        The internal representation of the force field for OpenMM.
    """
    omm_forcefield = openmm.app.ForceField(*forcefield)
    # modeller.addExtraParticles(forcefield)
    return omm_forcefield


def _build_omm_system(
        omm_forcefield: openmm.app.ForceField,
        omm_modeller: openmm.app.Modeller,
) -> openmm.System:
    """Build the OpenMM System object.

    Args:
        omm_forcefield: The OpenMM representation of the forcefield.
        omm_modeller: The OpenMM representation of the system.

    Returns:
        The internal representation of forces, constraints, and
        particles for OpenMM.
    """
    omm_system = omm_forcefield.createSystem(
        omm_modeller.topology,
        rigidWater=False,
    )
    return omm_system


def _empty_omm_system(system: System) -> openmm.System:
    """Build an empty OpenMM System object.

    Args:
        system: The system which will be tied to the OpenMM System.

    Returns:
        An internal representation of forces, constraints, and
        particles in OpenMM for a system with no forces or constraints.
    """
    omm_system = openmm.System()
    for i in range(len(system)):
        omm_system.addParticle(0.)
    return omm_system


def _adjust_omm_forces(
        nonbonded_method: str,
        nonbonded_cutoff: int | float,
        pme_gridnumber: tuple[int, int, int] | None,
        pme_alpha: int | float | None,
        omm_system: openmm.System,
) -> None:
    r"""Adjust the OpenMM Nonbonded forces with appropriate settings.

    Args:
        nonbonded_method: The method for treating non-bonded
            interactions in OpenMM.
        nonbonded_cutoff: The distance
            (:math:`\mathrm{\mathring(A)^{-1}}`) at which to truncate
            close-range non-bonded interactions.
        pme_gridnumber: The number of grid points to include along each
            lattice edge in PME summation.
        pme_alpha: The Gaussian width parameter in Ewald summation
            (:math:`\mathrm{nm^{-1}}`).
        omm_system: The OpenMM representation of forces, constraints,
            and particles.
    """
    for i, force in enumerate(omm_system.getForces()):
        force.setForceGroup(i)
        if type(force) not in SUPPORTED_FORCES:
            raise ValueError(f"{type(force)}")
        if isinstance(force, openmm.NonbondedForce):
            if nonbonded_method.upper() in NEEDS_CUTOFF:
                force.setCutoffDistance(nonbonded_cutoff*openmm.unit.angstrom)
            if nonbonded_method.upper() == "PME":
                force.setNonbondedMethod(openmm.NonbondedForce.PME)
                if (pme_alpha is not None and pme_gridnumber is not None):
                    force.setPMEParameters(pme_alpha, *pme_gridnumber)
        if isinstance(force, openmm.CustomNonbondedForce):
            if nonbonded_method.upper() in NEEDS_CUTOFF:
                force.setCutoffDistance(nonbonded_cutoff*openmm.unit.angstrom)
            if nonbonded_method.upper() in PERIODIC:
                force.setNonbondedMethod(
                    openmm.CustomNonbondedForce.CutoffPeriodic,
                )


def _adjust_system(
        system: System,
        omm_system: openmm.System,
) -> None:
    """Replace system masses and charges with those from the forcefield.

    Args:
        system: The system whose masses and charges will be updated.
        omm_system: The OpenMM representation of forces, constraints,
            and particles.
    """
    masses = []
    charges = []
    for force in omm_system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            for atom in range(omm_system.getNumParticles()):
                masses.append(
                    omm_system.getParticleMass(atom) / openmm.unit.daltons,
                )
                q, _, _ = force.getParticleParameters(
                    atom,
                )
                charges.append(q / openmm.unit.elementary_charge)
    system.masses[:] = masses
    system.charges[:] = charges


def _build_omm_context(
        omm_system: openmm.System,
        omm_modeller: openmm.app.Modeller,
) -> openmm.Context:
    """Build the OpenMM Context object.

    Args:
        omm_system: The OpenMM representation of forces, constraints,
            and particles.
        omm_modeller: The OpenMM representation of the system.

    Returns:
        The OpenMM machinery required to perform energy and force
        calculations, containing the System object and the specific
        platform to use, which is currently just the CPU platform.
    """
    omm_integrator = openmm.VerletIntegrator(1. * openmm.unit.femtosecond)
    # We currently only support the CPU platform.
    omm_platform = openmm.Platform.getPlatformByName("CPU")
    omm_context = openmm.Context(omm_system, omm_integrator, omm_platform)
    omm_context.setPositions(omm_modeller.positions)
    return omm_context
