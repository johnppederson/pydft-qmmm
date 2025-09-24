"""The PyDFT-QMMM package, a simulation engine employing the QM/MM/PME
method described in `The Journal of Chemical Physics`_.

Todo:
    * Evaluate performance using cProfile
    * Add unit and integration tests
    * Add way to track plugin operations on :class:`Calculator` and
      :class:`Integrator` objects.
    * Add support for arbitrary triclinic boxes

.. _The Journal of Chemical Physics: https://doi.org/10.1063/5.0087386
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

from .hamiltonians import MMHamiltonian
from .hamiltonians import QMHamiltonian
from .hamiltonians import QMMMHamiltonian
from .integrators import LangevinIntegrator
from .integrators import VerletIntegrator
from .system import Atom
from .system import System
from .utils import generate_velocities
from .wrappers import Optimization
from .wrappers import Simulation

__author__ = "Jesse McDaniel, John Pederson"
try:
    __version__ = version("pydft-qmmm")
except PackageNotFoundError:
    pass

del version, PackageNotFoundError
