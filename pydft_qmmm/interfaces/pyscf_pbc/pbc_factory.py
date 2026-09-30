"""Build the periodic PySCF interface."""
from __future__ import annotations

__all__ = ["pyscf_pbc_interface_factory"]

from typing import Any
from typing import TYPE_CHECKING

from ..pyscf import pyscf_backend
from . import pbc_interface

if TYPE_CHECKING:
    from pydft_qmmm import System


def pyscf_pbc_interface_factory(
        system: System,
        /,
        basis: str,
        pseudo: str | None,
        charge: int,
        multiplicity: int,
        functional: str | None = None,
        ke_cutoff: float | None = None,
        mesh: tuple[int, int, int] | None = None,
        embedding_sigma: float = 0.3,
        device: str = "cpu",
        ecp: Any = None,
        output_file: str | None = None,
        output_interval: int = 1,
        conv_tol: float = 1e-9,
        max_cycle: int = 100,
        verbose: int = 0,
        **options: Any,
) -> pbc_interface.PySCFPBCPotential:
    """Build a periodic PySCF interface.

    Args:
        system: Simulation system.
        basis: GTH basis name.
        pseudo: Required GTH pseudopotential name.
        charge: QM charge (e).
        multiplicity: QM spin multiplicity.
        functional: XC functional, or None for Hartree-Fock.
        ke_cutoff: FFT cutoff (Hartree); mutually exclusive with mesh.
        mesh: FFT dimensions; required if ke_cutoff is absent.
        embedding_sigma: Region II Gaussian width (Å).
        device: "cpu" for PySCF or "gpu" for GPU4PySCF.
        ecp: Unsupported; use pseudo.
        output_file: Log path, or None for standard output.
        output_interval: SCF calls between log writes.
        conv_tol: SCF energy tolerance (Hartree).
        max_cycle: Maximum SCF iterations.
        verbose: PySCF logging level.
        options: Extra solver attributes.

    Raises:
        ValueError: Unsupported configuration.
    """
    if pseudo is None:
        raise ValueError(
            "pseudo is required: the periodic gradient code rejects"
            " all-electron cells, so a GTH pseudopotential such as"
            " 'gth-pbe' must be given",
        )
    if ecp is not None:
        raise ValueError(
            "ecp is not supported under periodic boundary conditions;"
            " use a GTH pseudopotential through the pseudo argument",
        )
    if ke_cutoff is not None and mesh is not None:
        raise ValueError(
            "ke_cutoff and mesh both set the FFT grid; pass one, not both",
        )
    if ke_cutoff is None and mesh is None:
        raise ValueError(
            "the FFT grid is undetermined; pass ke_cutoff or mesh",
        )
    if ke_cutoff is not None and ke_cutoff <= 0:
        raise ValueError(f"ke_cutoff must be positive, got {ke_cutoff}")
    if embedding_sigma <= 0.0:
        raise ValueError(
            f"embedding_sigma must be positive, got {embedding_sigma}",
        )
    if output_interval < 1:
        raise ValueError(
            f"output_interval must be a positive integer, got"
            f" {output_interval}",
        )
    if conv_tol <= 0:
        raise ValueError(f"conv_tol must be positive, got {conv_tol}")
    if max_cycle < 0:
        raise ValueError(f"max_cycle must not be negative, got {max_cycle}")
    pyscf_backend.resolve_method(None, functional, multiplicity - 1)
    pyscf_backend.load_backend(device)
    return pbc_interface.PySCFPBCPotential(
        system,
        basis,
        pseudo,
        functional,
        ke_cutoff,
        mesh,
        embedding_sigma,
        device,
        charge,
        multiplicity,
        output_file,
        output_interval,
        conv_tol,
        max_cycle,
        verbose,
        dict(options),
    )
