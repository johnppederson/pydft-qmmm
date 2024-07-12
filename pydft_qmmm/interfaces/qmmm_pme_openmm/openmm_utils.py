"""Functionality for performing exclusions and generating State objects.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import openmm


def _generate_state(
        context: openmm.Context,
        groups: set[int] | int | None = -1,
) -> openmm.State:
    """Generate an OpenMM State in order to collect energies and forces.

    Args:
        context: An OpenMM Context object containing a representation
            of the system and appropriate forces.
        groups: The force groups of the context to include in the
            State evaluation.

    Return:
        An OpenMM State object containing the energies and forces of
        the current state of the system represented within the Context
        object for the specified groups of forces.
    """
    if groups is None:
        groups = -1
    return context.getState(
        getEnergy=True,
        getForces=True,
        getVext_grids=True,
        groups=groups,
    )
