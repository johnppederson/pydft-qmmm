"""Functionality for building the VASP interface.

Attributes:
    DEFAULT_INCAR: INCAR tags applied to every calculation unless the
        user overrides them. Symmetry is switched off because the MM
        environment breaks the symmetry of the QM subsystem. And
        the wavefunction and charge density are written so that the next
        step can restart from them.
"""
from __future__ import annotations

__all__ = ["vasp_interface_factory", "DEFAULT_INCAR"]

import os
from typing import TYPE_CHECKING

from .vasp_interface import VaspPotential

if TYPE_CHECKING:
    from typing import Any
    from pydft_qmmm import System


DEFAULT_INCAR: dict[str, Any] = {
    "PREC": "Accurate",
    "ALGO": "Fast",
    "ENCUT": 400,
    "EDIFF": 1e-6,
    "NELM": 100,
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "ISYM": 0,
    "LREAL": False,
    "LWAVE": True,
    "LCHARG": True,
}


def vasp_interface_factory(
        system: System,
        /,
        charge: int = 0,
        directory: str = "./vasp_workdir",
        command: str | None = None,
        kpts: tuple[int, int, int] = (1, 1, 1),
        pp_path: str | None = None,
        potcar_map: dict[str, str] | None = None,
        incar: dict[str, Any] | None = None,
        embedding: bool = False,
        embedding_sigma: float = 0.3,
        **options: Any,
) -> VaspPotential:
    r"""Build the interface to VASP.

    Args:
        system: The system which will be tied to the VASP interface.
        charge: The net charge (:math:`e`) of the QM subsystem.  Only
            ``charge=0`` is currently supported.
        directory: The working directory for VASP calculations.  It is
            created if it does not exist and is reused across steps so
            that WAVECAR and CHGCAR can seed the next calculation.
        command: The shell command that launches VASP.  Defaults to the
            ``PYDFT_QMMM_VASP_COMMAND`` or ``ASE_VASP_COMMAND``
            environment variable, or ``"vasp_std"``.
        kpts: The number of k-points along each reciprocal lattice
            vector.  Defaults to the Gamma point alone.
        pp_path: The directory holding per-species POTCAR
            subdirectories, e.g. ``".../potpaw_PBE.54"``.  Defaults to
            the ``VASP_PP_PATH`` environment variable.
        potcar_map: A mapping from element symbol to POTCAR
            subdirectory name, for non-default potentials, e.g.
            ``{"Li": "Li_sv"}``.
        incar: INCAR tags to merge over :data:`DEFAULT_INCAR`.  Use this
            for tags whose names are not Python identifiers, such as
            ``{"PLUGINS/LOCAL_POTENTIAL": True}``.
        embedding: Whether to electrostatically embed subsystem II point
            charges via the VASP Python plugin.  Requires a VASP binary
            built with ``-DPLUGINS``, and ``PYTHONHOME``/``PATH``
            pointing at an environment whose Python has numpy.  When
            ``False``, VASP sees only the QM subsystem and QM/MM
            coupling is left to the MM force field.
        embedding_sigma: The Gaussian width
            (:math:`\mathrm{\mathring{A}}`) used to represent MM point
            charges on VASP's FFT grid.  A point charge cannot be
            represented exactly on a finite grid, so it is smeared.
            Should comfortably exceed the grid spacing: too small
            aliases, too large over-softens the near field.  The
            default is ~2x the grid spacing of a typical run; below one
            spacing the deposited charge stops being conserved and
            starts depending on where the atom sits between nodes.
        options: Additional INCAR tags given as keyword arguments; the
            names are upper-cased, so ``encut=520`` sets ``ENCUT``.
            These take precedence over ``incar``.

    Returns:
        The VASP interface.
    """
    if charge:
        raise NotImplementedError(
            "VASP has no molecular-charge keyword; a charged QM region "
            "requires setting NELECT together with a compensating "
            "background.  Only charge=0 is supported.",
        )
    if command is None:
        command = os.environ.get(
            "PYDFT_QMMM_VASP_COMMAND",
            os.environ.get("ASE_VASP_COMMAND", "vasp_std"),
        )
    if pp_path is None:
        pp_path = os.environ.get("VASP_PP_PATH")
    if pp_path is None:
        raise ValueError(
            "No POTCAR library was given.  Set the pp_path keyword or "
            "the VASP_PP_PATH environment variable to the directory "
            "holding per-species POTCAR subdirectories.",
        )
    if embedding_sigma <= 0.0:
        raise ValueError(
            f"embedding_sigma must be positive, got {embedding_sigma}.",
        )
    tags = dict(DEFAULT_INCAR)
    if incar is not None:
        tags.update(incar)
    tags.update({key.upper(): value for key, value in options.items()})
    return VaspPotential(
        system,
        charge,
        directory,
        command,
        tags,
        tuple(kpts),
        pp_path,
        {} if potcar_map is None else dict(potcar_map),
        embedding,
        embedding_sigma,
    )
