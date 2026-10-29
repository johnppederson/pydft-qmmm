"""Solver construction for the PySCF interface.

PySCF exposes Hartree-Fock and Kohn-Sham solvers as separate classes,
one per spin treatment, and GPU4PySCF mirrors that layout with its own
classes that do not inherit from the CPU ones.  This module keeps the
mapping from a requested method name to the class that implements it,
so the interface itself never has to branch on either axis.
"""
from __future__ import annotations

__all__ = [
    "METHODS",
    "resolve_method",
    "build_solver",
    "load_backend",
    "load_submodule",
    "to_numpy",
    "to_like",
]

import importlib
from typing import Any
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from types import ModuleType

#: The solver each method name selects, as a (sub-module, class) pair,
#: together with whether it is Kohn-Sham and whether it is restricted to
#: a closed shell.
METHODS: dict[str, tuple[str, str, bool, bool]] = {
    "rhf": ("scf", "RHF", False, True),
    "uhf": ("scf", "UHF", False, False),
    "rohf": ("scf", "ROHF", False, False),
    "rks": ("dft", "RKS", True, True),
    "uks": ("dft", "UKS", True, False),
    "roks": ("dft", "ROKS", True, False),
}


def resolve_method(
        method: str | None,
        functional: str | None,
        spin: int,
) -> str:
    """Choose the solver for a requested method, functional, and spin.

    With no method named, the choice follows the wavefunction: Kohn-Sham
    when a functional is given and Hartree-Fock otherwise, restricted
    for a closed shell and unrestricted above it.

    Args:
        method: The requested method name, or None to choose one.
        functional: The requested exchange-correlation functional, or
            None for Hartree-Fock.
        spin: The number of unpaired electrons.

    Returns:
        The normalized method name.

    Raises:
        ValueError: If the method is unknown, if a Kohn-Sham method is
            requested without a functional or a Hartree-Fock method
            with one, or if a closed-shell method is requested for an
            open-shell system.
    """
    if method is None:
        if functional is None:
            return "rhf" if spin == 0 else "uhf"
        return "rks" if spin == 0 else "uks"
    name = str(method).lower()
    if name not in METHODS:
        raise ValueError(
            f"unknown method {method!r}; choose one of"
            f" {', '.join(sorted(METHODS))}",
        )
    _, _, kohn_sham, restricted = METHODS[name]
    if kohn_sham and functional is None:
        raise ValueError(
            f"the {name} method needs a functional; pass one, or choose a"
            " Hartree-Fock method",
        )
    if not kohn_sham and functional is not None:
        raise ValueError(
            f"the {name} method takes no functional, but {functional!r} was"
            " given; drop it, or choose a Kohn-Sham method",
        )
    if restricted and spin != 0:
        raise ValueError(
            f"the {name} method is restricted to a closed shell, but the QM"
            f" subsystem has {spin} unpaired electrons; choose the"
            f" unrestricted or restricted-open form instead",
        )
    return name


def load_backend(device: str) -> ModuleType:
    """Import the package implementing the solvers for a device.

    Args:
        device: Either ``"cpu"`` for PySCF or ``"gpu"`` for GPU4PySCF.

    Returns:
        The imported package.

    Raises:
        ValueError: If the device is not recognized.
        ImportError: If GPU4PySCF is requested but not installed.
    """
    if device == "cpu":
        import pyscf
        return pyscf
    if device != "gpu":
        raise ValueError(
            f"unknown device {device!r}; choose 'cpu' or 'gpu'",
        )
    try:
        import gpu4pyscf
    except ImportError as error:
        raise ImportError(
            "running the PySCF interface on a GPU needs GPU4PySCF, which"
            " is not installed in this environment; install it or set"
            " device='cpu'",
        ) from error
    return gpu4pyscf


def build_solver(
        backend: ModuleType,
        method: str,
        mol: Any,
        functional: str | None,
) -> Any:
    """Instantiate the solver a method name selects.

    Args:
        backend: The package providing the solvers.
        method: The normalized method name.
        mol: The molecule the solver will act on.
        functional: The exchange-correlation functional for a Kohn-Sham
            method, or None for Hartree-Fock.

    Returns:
        The unconverged solver.
    """
    module_name, class_name, kohn_sham, _ = METHODS[method]
    module = load_submodule(backend, module_name)
    solver = getattr(module, class_name)
    return solver(mol, xc=functional) if kohn_sham else solver(mol)


def load_submodule(backend: ModuleType, name: str) -> ModuleType:
    """Import a sub-module of a backend by name.

    Both packages populate some sub-modules lazily, so importing by
    name is more dependable than reaching for an attribute.

    Args:
        backend: The package providing the solvers.
        name: The sub-module name, such as ``scf`` or ``qmmm``.

    Returns:
        The imported sub-module.
    """
    return importlib.import_module(f"{backend.__name__}.{name}")


def _on_device(array: Any) -> bool:
    """
    GPU4PySCF hands back tagged subclasses of ``cupy.ndarray`` defined
    in its own modules, so the base classes have to be walked rather
    than the type's own module inspected.

    Args:
        array: Any object.

    Returns:
        Whether the object is an array living in GPU memory.
    """
    return any(
        base.__module__.split(".")[0] == "cupy" and base.__name__ == "ndarray"
        for base in type(array).__mro__
    )


def to_numpy(array: Any) -> Any:
    """Bring an array back from GPU memory if it is there.

    GPU4PySCF returns device arrays from most of its solvers, and the
    embedding integrals, the PME operators and every energy this
    interface reports are computed on the host.

    Args:
        array: A host or device array.

    Returns:
        The equivalent host array.
    """
    return array.get() if _on_device(array) else array


def to_like(array: Any, reference: Any) -> Any:
    """Put an array in the same memory as another one.

    Args:
        array: The host array to place.
        reference: The array whose memory it should share.

    Returns:
        The array, moved to the device if the reference is on one.
    """
    if _on_device(reference):
        import cupy
        return cupy.asarray(array)
    return np.asarray(array)
