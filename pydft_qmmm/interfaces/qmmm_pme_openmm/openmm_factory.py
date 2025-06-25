"""Functionality for building the QM/MM/PME-hacked OpenMM interface.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import openmm
from simtk.unit import femtosecond
from simtk.unit import nanometer

from .openmm_interface import PMEOpenMMInterface
from pydft_qmmm.interfaces.openmm.openmm_factory import _adjust_forces
from pydft_qmmm.interfaces.openmm.openmm_factory import _build_forcefield
from pydft_qmmm.interfaces.openmm.openmm_factory import _build_modeller
from pydft_qmmm.interfaces.openmm.openmm_factory import _build_system
from pydft_qmmm.interfaces.openmm.openmm_factory import _build_topology
from pydft_qmmm.interfaces.openmm.openmm_factory import _empty_system

if TYPE_CHECKING:
    from openmm.app import Modeller
    from ..interface import MMSettings


def pme_openmm_interface_factory(settings: MMSettings) -> PMEOpenMMInterface:
    """Build the interface to OpenMM given the settings.

    Args:
        settings: The settings used to build the OpenMM interface.

    Returns:
        The QM/MM/PME OpenMM interface.
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
    if not (x := settings.pme_gridnumber) is None:
        for num in x:
            if num != x[0]:
                raise ValueError(
                    (
                        "Non-uniform number of grid points along each axis "
                        "is not currently supported for QM/MM/PME."
                    ),
                )
    topology = _build_topology(settings)
    modeller = _build_modeller(settings, topology)
    forcefield = _build_forcefield(settings, modeller)
    system = _build_system(forcefield, modeller)
    system.setDefaultPeriodicBoxVectors(*box_vectors)
    _adjust_forces(settings, system)
    base_context = _build_context(
        settings, system, modeller, {
            "ReferenceVextGrid": "true",
        },
    )
    ixn_context = _build_context(settings, _empty_system(settings), modeller)
    wrapper = PMEOpenMMInterface(settings, base_context, ixn_context)
    # Register observer functions.
    settings.system.charges.register_notifier(wrapper.update_charges)
    settings.system.positions.register_notifier(wrapper.update_positions)
    settings.system.box.register_notifier(wrapper.update_box)
    settings.system.subsystems.register_notifier(wrapper.update_subsystems)
    return wrapper


def _build_context(
        settings: MMSettings,
        system: openmm.System,
        modeller: Modeller,
        properties: None | dict[str, str] = None,
) -> openmm.Context:
    """Build the QM/MM/PME OpenMM Context object.

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
    if properties:
        context = openmm.Context(system, integrator, platform, properties)
    else:
        context = openmm.Context(system, integrator, platform)
    context.setPositions(modeller.positions)
    return context
