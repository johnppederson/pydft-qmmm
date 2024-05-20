#! /usr/bin/env python3
"""A module to define the :class:`SoftwareInterface` base class and the
various :class:`SoftwareSettings` classes.
"""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import TypeVar

from pydft_qmmm.common import TheoryLevel

if TYPE_CHECKING:
    from pydft_qmmm import System
    from numpy.typing import NDArray
    import numpy as np

T = TypeVar("T")


class SoftwareSettings(ABC):
    """An abstract :class:`SoftwareSettings` base class.

    .. note:: This currently doesn't do anything.
    """


@dataclass(frozen=True)
class MMSettings(SoftwareSettings):
    """An immutable wrapper class which holds settings for an MM
    software interface.

    :param system: |system| to perform MM calculations on.
    """
    system: System
    forcefield_file: str | list[str]
    topology_file: list[str] | None = None
    nonbonded_method: str = "PME"
    nonbonded_cutoff: float | int = 14.
    pme_gridnumber: int | None = None
    pme_alpha: float | int | None = None


@dataclass(frozen=True)
class QMSettings(SoftwareSettings):
    """An immutable wrapper class which holds settings for a QM software
    interface.

    :param system: |system| to perform QM calculations on.
    :param basis_set: |basis_set|
    :param functional: |functional|
    :param charge: |charge|
    :param spin: |spin|
    :param quadrature_spherical: |quadrature_spherical|
    :param quadrature_radial: |quadrature_radial|
    :param scf_type: |scf_type|
    :param read_guess: |read_guess|
    """
    system: System
    basis_set: str
    functional: str
    charge: int
    spin: int
    quadrature_spherical: int = 302
    quadrature_radial: int = 75
    scf_type: str = "df"
    read_guess: bool = True


class SoftwareInterface(ABC):
    """The abstract :class:`SoftwareInterface` base class.
    """
    theory_level: TheoryLevel

    @abstractmethod
    def compute_energy(self) -> float:
        """
        """

    @abstractmethod
    def compute_forces(self) -> NDArray[np.float64]:
        """
        """

    @abstractmethod
    def compute_components(self) -> dict[str, float]:
        """
        """


class MMInterface(SoftwareInterface):
    """The abstract :class:`SoftwareInterface` base class.
    """
    theory_level = TheoryLevel.MM

    @abstractmethod
    def zero_intramolecular(self, atoms: frozenset[int]) -> None:
        """
        """

    @abstractmethod
    def zero_charges(self, atoms: frozenset[int]) -> None:
        """
        """

    @abstractmethod
    def zero_intermolecular(self, atoms: frozenset[int]) -> None:
        """
        """

    @abstractmethod
    def zero_forces(self, atoms: frozenset[int]) -> None:
        """
        """

    @abstractmethod
    def add_real_elst(
            self,
            atoms: frozenset[int],
            const: float | int = 1,
            inclusion: NDArray[np.float64] | None = None,
    ) -> None:
        """
        """

    @abstractmethod
    def add_non_elst(
            self,
            atoms: frozenset[int],
            inclusion: NDArray[np.float64] | None = None,
    ) -> None:
        """
        """


class QMInterface(SoftwareInterface):
    """The abstract :class:`SoftwareInterface` base class.
    """
    theory_level = TheoryLevel.QM

    @abstractmethod
    def disable_embedding(self) -> None:
        """
        """
