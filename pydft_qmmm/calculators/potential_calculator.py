"""A calculator class using a potential object.
"""
from __future__ import annotations

__all__ = ["PotentialCalculator"]

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .calculator import Calculator
from .calculator import Results
from pydft_qmmm.utils import pluggable_method

if TYPE_CHECKING:
    from pydft_qmmm.potentials import AtomicPotential
    from pydft_qmmm import System  # noqa: F401


@dataclass(frozen=True)
class PotentialCalculator(Calculator):
    """A calculator utilizing a potential object.

    Args:
        system: The system whose atom positions, atom identities, and
            geometry will be used to calculate energies and forces.
        potential: The potential object that will be used to
            calculate energies and forces.
    """
    potential: AtomicPotential

    @pluggable_method
    def calculate(
            self,
            return_forces: bool = True,
            return_components: bool = True,
    ) -> Results:
        r"""Calculate energies and forces.

        Args:
            return_forces: Whether or not to return forces.
            return_components: Whether or not to return the components
                of the energy.

        Returns:
            A wrapper containing the energy
            (:math:`\mathrm{kJ\;mol^{-1}}`), forces
            (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`), and
            energy components (:math:`\mathrm{kJ\;mol^{-1}}`) of the
            calculation.
        """
        energy = self.potential.compute_energy()
        results = Results(energy)
        if return_forces:
            forces = self.potential.compute_forces()
            results.forces = forces
        if return_components:
            components = self.potential.compute_components()
            results.components = components
        return results

    @property
    def name(self) -> str:
        """The name of the calculator, for logging purposes.
        """
        return type(self.potential).__name__.replace("Potential", "").strip()
