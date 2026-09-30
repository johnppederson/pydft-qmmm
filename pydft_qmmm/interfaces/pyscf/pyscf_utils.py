"""Utilities for the PySCF interface."""
from __future__ import annotations

__all__ = [
    "core_electrons",
    "validate_spin",
    "qm_atom_indices",
    "embedding_atom_indices",
    "spin_sum",
]

from typing import Any
from typing import TYPE_CHECKING

import numpy as np
from pyscf import gto
from pyscf.gto import basis as gto_basis

if TYPE_CHECKING:
    from collections.abc import Sequence
    from numpy.typing import NDArray
    from pydft_qmmm import System


def _ecp_name(ecp: Any, element: str) -> Any:
    """Get the ECP assigned to one element.

    Args:
        ecp: The ECP specification, as PySCF accepts it: a name, or a
            mapping of element symbol to name with an optional
            ``default`` entry.
        element: The element symbol.

    Returns:
        The ECP name for the element, or None if it has none.
    """
    if isinstance(ecp, dict):
        return ecp.get(element, ecp.get("default"))
    return ecp


def _load_core(name: Any, element: str) -> int:
    """Get the electron count an ECP replaces for one element.

    Args:
        name: The basis or ECP name to look the element up in.
        element: The element symbol.

    Returns:
        The number of core electrons the ECP replaces, or zero when the
        name defines no ECP for the element or cannot be looked up.
    """
    if not isinstance(name, str):
        return 0
    try:
        data = gto_basis.load_ecp(name, element)
    except Exception:
        return 0
    return int(data[0]) if data else 0


def core_electrons(
        basis: str,
        elements: Sequence[str],
        ecp: Any,
) -> int:
    """Count the electrons replaced by an ECP.

    Reject a valence-only basis without its ECP because PySCF does not
    infer the ECP from the basis.

    Args:
        basis: The orbital basis name.
        elements: The element symbols of the QM subsystem.
        ecp: The requested ECP specification, or None.

    Returns:
        The total number of core electrons replaced by the ECP.

    Raises:
        ValueError: If the basis defines an ECP for an element of the
            QM subsystem but no ECP was requested.
    """
    if ecp is None:
        implied = sorted(
            {
                element for element in elements
                if _load_core(basis, element)
            },
        )
        if implied:
            raise ValueError(
                f"the {basis!r} basis defines an effective core potential"
                f" for {', '.join(implied)}, but no ecp was requested."
                " PySCF would place every electron in a valence-only"
                f" basis; pass ecp={basis!r} to apply it, as Psi4 does"
                " automatically",
            )
        return 0
    return sum(
        _load_core(_ecp_name(ecp, element), element) for element in elements
    )


def validate_spin(
        elements: Sequence[str],
        charge: int,
        multiplicity: int,
        core: int = 0,
) -> tuple[int, int]:
    """Check that the electron count and multiplicity are consistent.

    Args:
        elements: The element symbols of the QM subsystem.
        charge: The net charge (:math:`e`) of the QM subsystem.
        multiplicity: The spin multiplicity of the QM subsystem.
        core: The number of electrons replaced by an ECP.

    Returns:
        The number of electrons the solver treats explicitly and the
        number of unpaired electrons.

    Raises:
        ValueError: If the multiplicity cannot be realized by the
            electron count implied by the elements and net charge.
    """
    electrons = sum(
        gto.charge(str(element)) for element in elements
    ) - charge - core
    spin = multiplicity - 1
    if multiplicity < 1 or spin > electrons or (electrons - spin) % 2:
        raise ValueError(
            f"electron count {electrons} is incompatible with multiplicity"
            f" {multiplicity}",
        )
    return electrons, spin


def qm_atom_indices(system: System) -> tuple[int, ...]:
    """Get the sorted indices of the QM subsystem.

    Args:
        system: The system whose subsystem I will be selected.

    Returns:
        The sorted original system indices of subsystem I.
    """
    return tuple(sorted(system.select("subsystem I")))


def embedding_atom_indices(system: System) -> tuple[int, ...]:
    """Get the sorted indices of the finite point-charge environment.

    Args:
        system: The system whose subsystem II will be selected.

    Returns:
        The sorted original system indices of subsystem II.
    """
    return tuple(sorted(system.select("subsystem II")))


def spin_sum(dm: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sum a density matrix over its spin channels.

    A restricted method produces a single spin-summed matrix and an
    unrestricted or restricted-open one produces a matrix per spin
    channel.  The array type is preserved, so a device array stays on
    the device for the integrals that want it there.

    Args:
        dm: A restricted or unrestricted density matrix.

    Returns:
        The spin-summed density matrix.
    """
    if getattr(dm, "ndim", None) == 3:
        return dm.sum(axis=0)
    return dm
