"""Functionality for building the Psi4 interface.
"""
from __future__ import annotations

__all__ = ["psi4_interface_factory"]

from typing import TYPE_CHECKING

import psi4

from . import psi4_interface
from . import psi4_utils

if TYPE_CHECKING:
    from pydft_qmmm import System


def psi4_interface_factory(
        system: System,
        /,
        basis: str,
        functional: str,
        charge: int,
        multiplicity: int,
        output_file: str | None = None,
        output_interval: int = 1,
        **options: psi4_utils.Psi4Options,
) -> psi4_interface.Psi4Potential:
    """Build the interface to Psi4.

    Args:
        system: The system which will be tied to the OpenMM interface.
        basis: The name of the basis set or a custom bases set string
            to use in QM calculations.
        functional: The name of the functional to use in QM
            calculations.
        charge: The net charge (:math:`e`) of the QM subsystem.
        multiplicity: The spin multiplicity of the QM subsystem.
        output_file: The file to which Psi4 output is written.
        output_interval: The interval at which Psi4 output should be
            written, e.g., the default value of 1 means that output
            will be written every calculation.
        options: Additional options to provide to Psi4.  See
            `Psi4 options`_ for additional Psi4 options.

    Returns:
        The Psi4 interface.
    """
    if "assign" not in basis:
        basis = "assign " + basis.strip()
    else:
        basis = basis
    psi4.basis_helper(basis, name="default")
    options["basis"] = "default"
    psi4_utils.set_options(**options)
    output_file = "stdout" if output_file is None else output_file
    wrapper = psi4_interface.Psi4Potential(
        system,
        functional,
        charge,
        multiplicity,
        output_file,
        output_interval,
    )
    return wrapper
