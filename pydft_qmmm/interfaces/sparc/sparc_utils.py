"""File I/O and output checking for the SPARC interface.

The grid file format is shared with the SPARC fork's ``src/qmmm.c``:
two ASCII header lines followed by a raw little-endian float64 payload
in SPARC's memory order, ``index = i + j*Nx + k*Nx*Ny``.  In numpy that
is Fortran order for an ``(Nx, Ny, Nz)`` array, so ``order="F"`` on the
ravel and the reshape is the entire layout conversion.
"""
from __future__ import annotations

__all__ = [
    "SparcExecutionError",
    "write_grid_file",
    "read_grid_file",
    "assert_embedding_capable",
    "assert_scf_converged",
    "extended_parameters_path",
]

import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

MAGIC = "PYDFT_QMMM_GRID"
VERSION = 1


class SparcExecutionError(RuntimeError):
    """An exception raised when a SPARC calculation fails.

    Args:
        directory: The working directory of the failed calculation.
        message: A description of the failure.
    """

    def __init__(self, directory: str, message: str) -> None:
        super().__init__(
            f"SPARC calculation in '{directory}' failed: {message}\n"
            "Inspect SPARC.out, stdout, and stderr in that directory "
            "for details.",
        )


def extended_parameters_path() -> str:
    """
    Returns:
        The path to the SPARC-X-API schema extended with the QM/MM tags.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "parameters.json")


def write_grid_file(
        path: str,
        field: NDArray[np.float64],
        cell_bohr: NDArray[np.float64],
        step: int,
) -> None:
    r"""Write a scalar field on the grid for SPARC to read.

    Args:
        path: The destination file.
        field: An (Nx, Ny, Nz) array in SPARC's units.
        cell_bohr: A 3x3 array whose rows are lattice vectors
            (:math:`\mathrm{Bohr}`).
        step: A monotonically increasing stamp, so a stale file left by
            a previous evaluation is detectable.
    """
    field = np.ascontiguousarray(field, dtype=np.float64)
    if field.ndim != 3:
        raise ValueError(f"field must be 3D, got shape {field.shape}.")
    nx, ny, nz = field.shape
    cell = np.asarray(cell_bohr, dtype=np.float64).reshape(3, 3)
    with open(path, "wb") as fh:
        fh.write(f"{MAGIC} {VERSION} {nx} {ny} {nz} {step}\n".encode())
        fh.write(
            (" ".join(f"{v:.16e}" for v in cell.ravel()) + "\n").encode(),
        )
        # Fortran order IS the layout conversion; see the module docstring.
        fh.write(field.astype("<f8").ravel(order="F").tobytes())


def read_grid_file(
        path: str,
        expect_shape: tuple[int, int, int] | None = None,
        expect_step: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    r"""Read a scalar field written by SPARC.

    Args:
        path: The file to read.
        expect_shape: The grid dimensions the caller requires, if any.
        expect_step: The step stamp the caller requires, if any.

    Returns:
        The field as an (Nx, Ny, Nz) array, the lattice vectors
        (:math:`\mathrm{Bohr}`), and the step stamp.

    Raises:
        ValueError: If the magic, version, shape, or step disagree.
    """
    with open(path, "rb") as fh:
        head = fh.readline().decode().split()
        if len(head) != 6 or head[0] != MAGIC:
            raise ValueError(f"{path} is not a {MAGIC} file.")
        if int(head[1]) != VERSION:
            raise ValueError(
                f"{path} is format version {head[1]}, expected {VERSION}.",
            )
        shape = (int(head[2]), int(head[3]), int(head[4]))
        step = int(head[5])
        cell = np.array(
            [float(v) for v in fh.readline().decode().split()],
        ).reshape(3, 3)
        payload = np.frombuffer(fh.read(), dtype="<f8")
    if expect_shape is not None and shape != tuple(expect_shape):
        raise ValueError(
            f"{path} holds grid {shape} but {tuple(expect_shape)} was "
            "expected.",
        )
    if expect_step is not None and step != expect_step:
        raise ValueError(
            f"{path} carries step {step} but {expect_step} was expected; "
            "this is a stale file from a previous evaluation.",
        )
    expected = int(np.prod(shape))
    if payload.size != expected:
        raise ValueError(
            f"{path} holds {payload.size} values but {expected} were "
            "expected.",
        )
    return payload.reshape(shape, order="F").copy(), cell, step


def assert_embedding_capable(out_path: str, directory: str) -> None:
    """Verify the binary that ran actually supports embedding.

    A stock SPARC ignores the QM/MM tags entirely and produces a
    perfectly ordinary, converged, unembedded result.  That is the most
    dangerous failure available here, so it is checked explicitly.

    Args:
        out_path: The path to SPARC.out.
        directory: The working directory, for the error message.

    Raises:
        SparcExecutionError: If the output has no QMMM_FLAG echo.
    """
    if not os.path.isfile(out_path):
        raise SparcExecutionError(
            directory, f"{out_path} is absent; SPARC produced no output.",
        )
    with open(out_path) as fh:
        if "QMMM_FLAG" not in fh.read():
            raise SparcExecutionError(
                directory,
                "electrostatic embedding was requested but SPARC.out "
                "contains no QMMM_FLAG echo, so the binary is a stock "
                "SPARC that ignored the external potential.  The energy "
                "reported would be the unembedded one.  Build the fork "
                "from the qmmm-embedding branch and point the command "
                "keyword at it.",
            )


def assert_scf_converged(out_path: str, directory: str) -> None:
    """Verify the SCF reached the requested tolerance and the run finished.

    A run that crashed mid-SCF, or was killed before writing any output
    at all, leaves behind an empty or truncated ``SPARC.out`` with
    neither a failure message nor a completion marker.  That file would
    otherwise read as "no failure detected", so a positive completion
    marker (the ``Total walltime`` line SPARC prints at the very end of
    a normal run) is required in addition to the absence of the
    "did not converge" message.

    Args:
        out_path: The path to SPARC.out.
        directory: The working directory, for the error message.

    Raises:
        SparcExecutionError: If SPARC.out is absent, reports
            non-convergence, or has no completion marker.
    """
    if not os.path.isfile(out_path):
        raise SparcExecutionError(
            directory, f"{out_path} is absent; SPARC produced no output.",
        )
    with open(out_path) as fh:
        text = fh.read()
    if "did not converge" in text:
        raise SparcExecutionError(
            directory,
            "the SCF did not converge to the requested accuracy; the "
            "energy and forces are unconverged.  Loosen TOL_SCF, raise "
            "MAXIT_SCF, or improve the initial guess.",
        )
    if "Total walltime" not in text:
        raise SparcExecutionError(
            directory,
            f"{out_path} has no 'Total walltime' line, so the run did "
            "not finish; the file is empty or truncated and the energy "
            "and forces it reports, if any, are incomplete.",
        )
