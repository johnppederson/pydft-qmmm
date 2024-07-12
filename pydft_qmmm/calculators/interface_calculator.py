"""A calculator utilizing external software.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .calculator import Calculator
from pydft_qmmm.common import Results

if TYPE_CHECKING:
    from pydft_qmmm.interfaces import SoftwareInterface
    from pydft_qmmm import System


@dataclass
class InterfaceCalculator(Calculator):
    """A calculator utilizing external software.

    Args:
        system: The system whose atom positions, atom identities, and
            geometry will be used to calculate energies and forces.
        interface: The interface to an external software that will
            be used to calculate energies and forces.
    """
    system: System
    interface: SoftwareInterface

    def __post_init__(self) -> None:
        """Set level of theory and calculator name.
        """
        self.theory_level = self.interface.theory_level
        self.name = str(self.interface.theory_level).split(".")[1]

    def calculate(
            self,
            return_forces: bool | None = True,
            return_components: bool | None = True,
    ) -> Results:
        r"""Calculate energies and forces.

        Args:
            return_forces: Whether or not to return forces.
            return_components: Whether or not to return the components of
                the energy.

        Returns:
            The energy (:math:`\mathrm{kJ\;mol^{-1}}`), forces
            (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`), and energy
            components (:math:`\mathrm{kJ\;mol^{-1}}`) of the
            calculation.
        """
        energy = self.interface.compute_energy()
        results = Results(energy)
        if return_forces:
            forces = self.interface.compute_forces()
            results.forces = forces
        if return_components:
            components = self.interface.compute_components()
            results.components = components
        return results
