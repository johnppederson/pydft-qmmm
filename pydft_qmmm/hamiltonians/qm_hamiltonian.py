"""A Hamiltonian representing the QM level of theory.
"""
from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Callable
from typing import TYPE_CHECKING

from .hamiltonian import CalculatorHamiltonian
from pydft_qmmm.calculators import InterfaceCalculator
from pydft_qmmm.common import lazy_load
from pydft_qmmm.common import Subsystem
from pydft_qmmm.common import TheoryLevel
from pydft_qmmm.interfaces import QMSettings

if TYPE_CHECKING:
    from pydft_qmmm import System
    from pydft_qmmm.interfaces.interface import QMInterface
    Factory = Callable[[QMSettings], QMInterface]


@dataclass
class QMHamiltonian(CalculatorHamiltonian):
    """A Hamiltonian representing the QM level of theory.

    Args:
        basis_set: The name of the basis set to be used in QM
            calculations.
        functional: The name of the functional set to be used in QM
            calculations.
        charge: The net charge (:math:`e`) of the system represented by
            the QM Hamiltonian.
        spin: The net spin of the system represented by the QM
            Hamiltonian
        quadrature_spherical: The number of spherical Lebedev points
            to use in the DFT quadrature.
        quadrature_radial: The number of radial points to use in the
            DFT quadrature.
        scf_type: The name of the type of SCF to perform, relating to
            the JK build algorithms as in Psi4.
        read_guess: Whether or not to reuse previous wavefunctions as
            initial guesses in subsequent QM calculations.
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
        """Set level of theory.
        """
        self.theory_level = TheoryLevel.QM

    def build_calculator(self, system: System) -> InterfaceCalculator:
        """Build the calculator corresponding to the Hamiltonian.

        Args:
            system: The system that will be used to calculate the
                calculator.

        Returns:
            The calculator which is defined by the system and the
            Hamiltonian.
        """
        qm_atoms = self._parse_atoms(system)
        system.subsystems[qm_atoms] = Subsystem.I
        settings = QMSettings(system=system, **asdict(self))
        interface = lazy_load("pydft_qmmm.interfaces").qm_factory(settings)
        calculator = InterfaceCalculator(system=system, interface=interface)
        return calculator

    def __str__(self) -> str:
        """Create a LATEX string representation of the Hamiltonian.

        Returns:
            The string representation of the Hamiltonian.
        """
        return "H^{QM}" + super().__str__()
