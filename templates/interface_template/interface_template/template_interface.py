"""A module containing the TemplateInterface class.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

from pydft_qmmm.interfaces import SoftwareInterface
from pydft_qmmm.potentials import AtomicPotential
from pydft_qmmm.utils import TheoryLevel

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass(frozen=True)
class TemplateInterface(SoftwareInterface):
    """A mix-in for storing and manipulating Template data types.

    Attributes:
        theory_level: The level of theory that the software applies in
            energy and force calculations.
    """
    theory_level: TheoryLevel = field(default=TheoryLevel.NO, init=False)
    # Provide other methods for manipulation of the data types
    # belonging to the external software (called 'Template' here).


class TemplatePotential(TemplateInterface, AtomicPotential):
    """A potential wrapping Template functionality.
    """

    def compute_energy(self) -> float:
        r"""Compute the energy of the system using Template.

        Returns:
            The energy (:math:`\mathrm{kJ\;mol^{-1}}`) of the system.
        """
        # All potentials must calculate energies.
        ...

    def compute_forces(self) -> NDArray[np.float64]:
        r"""Compute the forces on the system using Template.

        Returns:
            The forces
            (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`) acting
            on atoms in the system.
        """
        # All potentials must calculate forces.  Although some
        # potentials may calculate forces for only a portion of the
        # entire system, the "compute_forces" method must return an
        # array whose size comprises the entire system, so the
        # irrelevant parts of the system should be given forces of
        # 0 in these cases.
        ...

    def compute_components(self) -> dict[str, float]:
        r"""Compute the components of energy using Template.

        Returns:
            The components of the energy (:math:`\mathrm{kJ\;mol^{-1}}`)
            of the system.
        """
        # All potentials must calculate meaningful energy components.
        # This may not be applicable in many cases, and so an empty
        # dict may also be returned.
        ...
