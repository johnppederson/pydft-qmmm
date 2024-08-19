"""Functionality for building the QM/MM/PME-hacked Psi4 interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import psi4.core

from .psi4_interface import PMEPsi4Interface
from pydft_qmmm.interfaces.psi4.psi4_utils import _build_context
from pydft_qmmm.interfaces.psi4.psi4_utils import Psi4Options


if TYPE_CHECKING:
    from pydft_qmmm.interfaces import QMSettings


@dataclass(frozen=True)
class PMEPsi4Options(Psi4Options):
    """An immutable wrapper class for storing Psi4 global options.

    Args:
        pme: Whether or not to perform a Psi4 calculation with the
            interpolated PME potential.
    """
    pme: str = "true"


def pme_psi4_interface_factory(settings: QMSettings) -> PMEPsi4Interface:
    """Build the interface to Psi4 given the settings.

    Args:
        settings: The settings used to build the Psi4 interface.

    Returns:
        The QM/MM/PME-hacked Psi4 interface.
    """
    basis = settings.basis_set
    if "assign" not in settings.basis_set:
        basis = "assign " + settings.basis_set.strip()
    psi4.basis_helper(basis, name="default")
    options = _build_options(settings)
    functional = settings.functional
    context = _build_context(settings)
    wrapper = PMEPsi4Interface(
        settings, options, functional, context,
    )
    # Register observer functions.
    settings.system.charges.register_notifier(wrapper.update_charges)
    settings.system.positions.register_notifier(wrapper.update_positions)
    settings.system.subsystems.register_notifier(wrapper.update_subsystems)
    return wrapper


def _build_options(settings: QMSettings) -> PMEPsi4Options:
    """Build the PMEPsi4Options object.

    Args:
        settings: The settings used to build the Psi4 interface.

    Returns:
        The global options used by the QM/MM/PME-hacked Psi4 in each
        calculation.
    """
    options = PMEPsi4Options(
        "default",
        settings.quadrature_spherical,
        settings.quadrature_radial,
        settings.scf_type,
        "uks" if settings.spin > 1 else "rks",
        "read" if settings.read_guess else "auto",
        "true",
    )
    return options
