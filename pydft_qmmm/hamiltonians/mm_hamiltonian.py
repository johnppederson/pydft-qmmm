"""A Hamiltonian representing the MM level of theory.
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
from pydft_qmmm.interfaces import MMSettings

if TYPE_CHECKING:
    from pydft_qmmm import System
    from pydft_qmmm.interfaces.interface import MMInterface
    Factory = Callable[[MMSettings], MMInterface]


@dataclass
class MMHamiltonian(CalculatorHamiltonian):
    r"""A Hamiltonian representing the MM level of theory.

    Args:
        forcefield: The files containing forcefield and topology
            data for the system.
        nonbonded_method: The method for treating non-bonded
            interactions, as in OpenMM.
        nonbonded_cutoff: The distance at which to truncate close-range
            non-bonded interactions.
        pme_gridnumber: The number of grid points to include along each
            lattice edge in PME summation.
        pme_alpha: The Gaussian width parameter in Ewald summation
            (:math:`\mathrm{nm^{-1}}`).
    """
    forcefield: str | list[str]
    nonbonded_method: str = "PME"
    nonbonded_cutoff: float | int = 14.
    pme_gridnumber: int | tuple[int, int, int] | None = None
    pme_alpha: float | int | None = None

    def __post_init__(self) -> None:
        """Set level of theory.
        """
        self.theory_level = TheoryLevel.MM

    def build_calculator(self, system: System) -> InterfaceCalculator:
        """Build the calculator corresponding to the Hamiltonian.

        Args:
            system: The system that will be used to calculate the
                calculator.

        Returns:
            The calculator which is defined by the system and the
            Hamiltonian.
        """
        mm_atoms = self._parse_atoms(system)
        system.subsystems[mm_atoms] = Subsystem.III
        if isinstance(self.pme_gridnumber, int):
            self.pme_gridnumber = (self.pme_gridnumber,) * 3
        if isinstance(self.forcefield, str):
            self.forcefield = [self.forcefield]
        settings = MMSettings(system=system, **asdict(self))
        interface = lazy_load("pydft_qmmm.interfaces").mm_factory(settings)
        calculator = InterfaceCalculator(system=system, interface=interface)
        return calculator

    def __str__(self) -> str:
        """Create a LATEX string representation of the Hamiltonian.

        Returns:
            The string representation of the Hamiltonian.
        """
        return "H^{MM}" + super().__str__()
