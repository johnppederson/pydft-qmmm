"""A Hamiltonian representing the QM level of theory.
"""
from __future__ import annotations

__all__ = ["QMHamiltonian"]

from typing import TYPE_CHECKING

from .hamiltonian import PotentialHamiltonian
from pydft_qmmm.calculators import PotentialCalculator
from pydft_qmmm.utils import Subsystem
from pydft_qmmm.utils import TheoryLevel
from pydft_qmmm.utils import TheoryLevelError
from pydft_qmmm.utils import InterfaceImportError
from pydft_qmmm.interfaces import interfaces

if TYPE_CHECKING:
    from typing import Any
    from pydft_qmmm import System


class QMHamiltonian(PotentialHamiltonian):
    """A Hamiltonian representing the QM level of theory.

    Args:
        interface: The name of the software that implements the desired
            level of theory described by the Hamiltonian.
        options: Keyword arguments to provide to the interface factory.
    """
    theory_level: TheoryLevel = TheoryLevel.QM

    def __init__(
            self,
            interface: str = "psi4",
            **options: dict[str, Any],
    ) -> None:
        self.interface = interface
        self.options = options

    def build_calculator(self, system: System) -> PotentialCalculator:
        """Build the calculator corresponding to the Hamiltonian.

        Args:
            system: The system that will be used in calculations.

        Returns:
            The calculator which is defined by the system and the
            Hamiltonian.
        """
        qm_atoms = self._parse_atoms(system)
        system.subsystems[qm_atoms] = Subsystem.I
        try:
            interface_info = interfaces[self.interface]
        except KeyError:
            raise InterfaceImportError(self.interface)
        if interface_info[0] != self.theory_level:
            raise TheoryLevelError(
                self.theory_level,
                interface_info[0],
                "interface specified for the QMHamiltonian object",
                "Please specify an interface at an appropriate level "
                "of theory for the QMHamiltonian.",
            )
        try:
            interface = interface_info[1](system, **self.options)
        except TypeError as e:
            raise e  # Todo: Make this informative.
        calculator = PotentialCalculator(system, interface)
        return calculator

    def __str__(self) -> str:
        """Create a LATEX string representation of the Hamiltonian.

        Returns:
            The string representation of the Hamiltonian.
        """
        return "H^{QM}" + super().__str__()
