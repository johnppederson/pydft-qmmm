"""Build the PySCF interface."""
from __future__ import annotations

__all__ = ["pyscf_interface_factory"]

from typing import Any
from typing import TYPE_CHECKING

from . import pyscf_backend
from . import pyscf_interface
from . import pyscf_utils

if TYPE_CHECKING:
    from pydft_qmmm import System


def pyscf_interface_factory(
        system: System,
        /,
        basis: str,
        charge: int,
        multiplicity: int,
        functional: str | None = None,
        method: str | None = None,
        density_fit: bool = False,
        auxbasis: str | None = None,
        device: str = "cpu",
        ecp: Any = None,
        output_file: str | None = None,
        output_interval: int = 1,
        conv_tol: float = 1e-9,
        max_cycle: int = 100,
        grid_level: int = 3,
        verbose: int = 0,
        **options: Any,
) -> pyscf_interface.PySCFPotential:
    r"""Build the interface to PySCF.

    Args:
        system: The system which will be tied to the PySCF interface.
        basis: The name of the basis set to use in QM calculations.
        charge: The net charge (:math:`e`) of the QM subsystem.
        multiplicity: The spin multiplicity of the QM subsystem.
        functional: The name of the exchange-correlation functional to
            use, or None for Hartree-Fock.
        method: The solver to use, one of ``rhf``, ``uhf``, ``rohf``,
            ``rks``, ``uks``, or ``roks``.  The default follows the
            wavefunction: Kohn-Sham when a functional is given and
            Hartree-Fock otherwise, restricted for a closed shell and
            unrestricted above it.
        density_fit: Whether to build the Coulomb and exchange matrices
            from a fitted auxiliary basis rather than from the exact
            four-center integrals.
        auxbasis: The auxiliary basis for density fitting, or None to
            let PySCF choose one to match the orbital basis.
        device: Either ``cpu`` to run on PySCF or ``gpu`` to run on
            GPU4PySCF.
        ecp: The effective core potential, or None for an all-electron
            calculation. PySCF does not infer it from the basis.
        output_file: The file to which PySCF output is written, or None
            to write to standard output.
        output_interval: The interval at which PySCF output should be
            written, e.g., the default value of 1 means that output
            will be written every calculation.
        conv_tol: The SCF energy convergence threshold
            (:math:`\mathrm{E_h}`).
        max_cycle: The maximum number of SCF iterations.
        grid_level: The PySCF quadrature level, used for both the
            exchange-correlation and the embedding integrals.
        verbose: The PySCF logging verbosity.
        options: Additional attributes to set on the PySCF solver.

    Returns:
        The PySCF interface.
    """
    if output_interval < 1:
        raise ValueError(
            f"output_interval must be a positive integer, got"
            f" {output_interval}",
        )
    if grid_level not in range(0, 10):
        raise ValueError(
            f"grid_level must be an integer between 0 and 9, got"
            f" {grid_level}",
        )
    if conv_tol <= 0:
        raise ValueError(f"conv_tol must be positive, got {conv_tol}")
    if max_cycle < 0:
        raise ValueError(f"max_cycle must not be negative, got {max_cycle}")
    if auxbasis is not None and not density_fit:
        raise ValueError(
            "auxbasis only applies to a density-fitted solver; pass"
            " density_fit=True or drop it",
        )
    # Fail on an unusable combination here rather than at the first
    # calculation, which may be many minutes into a run.
    pyscf_backend.resolve_method(method, functional, multiplicity - 1)
    pyscf_backend.load_backend(device)
    return pyscf_interface.PySCFPotential(
        system,
        basis,
        ecp,
        functional,
        method,
        density_fit,
        auxbasis,
        device,
        charge,
        multiplicity,
        output_file,
        output_interval,
        conv_tol,
        max_cycle,
        grid_level,
        verbose,
        dict(options),
    )
