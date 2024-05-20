#! /usr/bin/env python3
"""A module defining the :class:`QMHamiltonian` class.
"""
from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .hamiltonian import CalculatorHamiltonian
from pydft_qmmm.calculators import InterfaceCalculator
from pydft_qmmm.common import Subsystem
from pydft_qmmm.common import TheoryLevel
from pydft_qmmm.interfaces import qm_factory
from pydft_qmmm.interfaces import QMSettings

if TYPE_CHECKING:
    from pydft_qmmm import System


@dataclass
class QMHamiltonian(CalculatorHamiltonian):
    """A wrapper class to store settings for QM calculations.

    :param basis_set: |basis_set|
    :param functional: |functional|
    :param charge: |charge|
    :param spin: |spin|
    :param quadrature_spherical: |quadrature_spherical|
    :param quadrature_radial: |quadrature_radial|
    :param scf_type: |scf_type|
    :param read_guess: |read_guess|
    """
    basis_set: str
    functional: str
    charge: int
    spin: int
    quadrature_spherical: int = 302
    quadrature_radial: int = 75
    scf_type: str = "df"
    read_guess: bool = True

    def __post_init__(self) -> None:
        self.theory_level = TheoryLevel.QM

    def build_calculator(self, system: System) -> InterfaceCalculator:
        qm_atoms = self._parse_atoms(system)
        system.subsystems[qm_atoms] = Subsystem.I
        settings = QMSettings(system=system, **asdict(self))
        interface = qm_factory(settings)
        calculator = InterfaceCalculator(system=system, interface=interface)
        return calculator

    def __str__(self) -> str:
        return "H^{QM}" + super().__str__()
