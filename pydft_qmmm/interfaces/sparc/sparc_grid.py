"""Unit and sign conversions between PyDFT-QMMM and SPARC.

Every convention mismatch between the two codes lives in this module,
so that a sign error has exactly one place to hide.

SPARC works in Hartree atomic units on a grid whose memory order is
``index = i + j*Nx + k*Nx*Ny`` (x fastest).  That ordering is handled
where the bytes are written and read, in ``sparc_utils``, by using
Fortran ravel order; this module handles only units and sign.

The two directions are deliberately asymmetric:

* ``vext_to_sparc`` does NOT change sign.  The grid physics in
  ``pydft_qmmm.embedding.grid_potential`` already returns the electron
  potential ENERGY, and SPARC's ``Veff = Vxc + elecstPotential`` uses
  the same convention.  Negating here would invert the whole QM/MM
  coupling while still converging.
* ``phi_from_sparc`` DOES change sign, because
  ``contract_gaussian_gradient`` takes the physical potential in volts.
"""
from __future__ import annotations

__all__ = [
    "EV_PER_HARTREE",
    "BOHR_PER_ANGSTROM",
    "vext_to_sparc",
    "phi_from_sparc",
    "cell_to_bohr",
]

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

EV_PER_HARTREE = 27.211386245988
BOHR_PER_ANGSTROM = 1.8897261246257702


def vext_to_sparc(v_ext_ev: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Convert an external potential for SPARC to consume.

    Args:
        v_ext_ev: The external potential as electron potential energy
            (:math:`\mathrm{eV}`), as returned by
            ``build_external_potential``, ``erfc_potential`` and
            ``build_pme_potential``.

    Returns:
        The same field in :math:`\mathrm{Hartree}`, sign unchanged.
    """
    return np.asarray(v_ext_ev, dtype=np.float64) / EV_PER_HARTREE


def phi_from_sparc(phi_hartree: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Convert SPARC's electrostatic potential to volts.

    Args:
        phi_hartree: ``pSPARC->elecstPotential``, the electron potential
            energy (:math:`\mathrm{Hartree}`).

    Returns:
        The physical electrostatic potential
        (:math:`\mathrm{V}`), which is the negative of the input, in the
        convention ``contract_gaussian_gradient`` expects.
    """
    return -np.asarray(phi_hartree, dtype=np.float64) * EV_PER_HARTREE


def cell_to_bohr(cell_angstrom: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Convert lattice vectors to SPARC's units.

    Args:
        cell_angstrom: A 3x3 array whose rows are lattice vectors
            (:math:`\mathrm{\mathring{A}}`).

    Returns:
        The same vectors in :math:`\mathrm{Bohr}`.
    """
    return np.asarray(cell_angstrom, dtype=np.float64) * BOHR_PER_ANGSTROM
