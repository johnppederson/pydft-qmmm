"""Functionality for building the SPARC interface.
"""
from __future__ import annotations

__all__ = ["sparc_interface_factory"]

from typing import Any
from typing import TYPE_CHECKING

import numpy as np
from sparc.calculator import SPARC

from . import sparc_utils
from .sparc_interface import SPARCPotential

if TYPE_CHECKING:
    from pydft_qmmm import System


def sparc_interface_factory(
        system: System,
        /,
        charge: int = 0,
        directory: str = "./sparc_workdir",
        fd_grid: tuple[int, int, int] | None = None,
        h: float | None = None,
        embedding: bool = False,
        embedding_sigma: float = 0.3,
        **options: Any,
) -> SPARCPotential:
    r"""Build the interface to SPARC.

    Args:
        system: The system which will be tied to the SPARC interface.
        charge: The net charge (:math:`e`) of the QM subsystem.
        directory: The working directory for SPARC calculations.
        fd_grid: The finite-difference grid.  Pinned rather than derived
            from ``h`` so that the driver knows the grid before SPARC
            runs and can build the external potential on it.
        h: The mesh spacing (:math:`\mathrm{\mathring{A}}`), converted
            to ``fd_grid`` once, here.  Mutually exclusive with
            ``fd_grid``.
        embedding: Whether to electrostatically embed subsystem II point
            charges.  Requires a SPARC binary built from the
            ``qmmm-embedding`` branch.
        embedding_sigma: The Gaussian width
            (:math:`\mathrm{\mathring{A}}`) used to represent MM point
            charges on SPARC's grid.  Should comfortably exceed the grid
            spacing: too small aliases, too large over-softens the near
            field.
        options: Additional keyword arguments forwarded to
            ``sparc.calculator.SPARC``.

    Returns:
        The SPARC interface.
    """
    if fd_grid is not None and h is not None:
        raise ValueError(
            "Give either fd_grid or h, not both.  The SPARC interface "
            "pins FD_GRID so that the external potential can be built "
            "before SPARC runs.",
        )
    if fd_grid is None:
        if h is None:
            raise ValueError(
                "One of fd_grid or h is required, so that FD_GRID can "
                "be pinned.",
            )
        # Rows are lattice vectors; PyDFT-QMMM stores them as columns.
        cell = np.asarray(system.box).T
        lengths = np.linalg.norm(cell, axis=1)
        fd_grid = tuple(int(np.ceil(length / h)) for length in lengths)
    if embedding_sigma <= 0.0:
        raise ValueError(
            f"embedding_sigma must be positive, got {embedding_sigma}.",
        )
    if charge != int(charge):
        raise ValueError(
            f"charge must be an integer, got {charge}; SPARC's "
            "NET_CHARGE is an int.",
        )
    if charge:
        options["NET_CHARGE"] = int(charge)
    calculator = SPARC(
        directory=directory,
        sparc_json_file=sparc_utils.extended_parameters_path(),
        check_version=False,
        FD_GRID=tuple(fd_grid),
        **options,
    )
    return SPARCPotential(
        system,
        calculator,
        int(charge),
        directory,
        tuple(fd_grid),
        embedding,
        embedding_sigma,
    )
