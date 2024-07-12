#! /usr/bin/env python3
"""A module to define the TemplateInterface class.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from pydft_qmmm.common import Subsystem
from pydft_qmmm.common import TheoryLevel
from pydft_qmmm.interfaces import SoftwareInterface

if TYPE_CHECKING:
    from typing import Any
    import numpy as np
    from numpy.typing import NDArray


class TemplateInterface(SoftwareInterface):
    """The TemplateInterface class.
    """
    theory_level: TheoryLevel.NO

    # Perform initialization as needed.
    def __init__(
            self,
    ) -> None:
        ...

    def compute_energy(self) -> float:
        """A function which calculates energy.

        :return: The potential energy [kJ / mol].
        """
        # All interfaces must calculate energies.
        ...

    def compute_forces(self) -> NDArray[np.float64]:
        """A function which calculates forces.

        :return: The forces acting on the system [kJ / mol / Å].
        """
        # All interfaces must calculate forces.  Although some interfaces
        # may calculate forces for only a portion of the entire system,
        # the "compute_forces" method must return an array whose size
        # comprises the entire system, so the irrelevant parts of the
        # system should be given forces of 0 in these cases.
        ...

    def compute_components(self) -> dict[str, float]:
        """A function which calculates components of the energy.

        :return: Components of the potential energy [kJ / mol].
        """
        # All interfaces must calculate meaningful energy components.
        # This may not be applicable in many cases, and so an empty
        # dict may also be passed.
        ...

    def update_threads(self, threads: int) -> None:
        """A function which updates the number of threads.

        :param threads: The number of threads to provide to the
            external software.
        """
        # All interfaces must provide a way to update the number of
        # threads used by its associated software.  This may not be
        # application in many cases, and so a "pass" statement may
        # be written instead.
        ...

    def update_memory(self, memory: str) -> None:
        """A function which updates the available memory.

        :param memory: The amount of memory to provide to the
            external software.
        """
        # All interfaces must provide a way to update the amount of
        # memory used by its associated software.  This may not be
        # application in many cases, and so a "pass" statement may
        # be written instead.
        ...

    def update_positions(self, positions: NDArray[np.float64]) -> None:
        """A function which updates the atom positions.

        :param positions: The new positions of atoms to provide to the
            external software [Å].
        """
        # This function is passed to the system in the factory method,
        # adding it to the list of functions called when positions are
        # altered.  This is how the external software is updated to
        # reflect the state of the system.
        ...

    def update_charges(self, charges: NDArray[np.float64]) -> None:
        """A function which updates the atom charges.

        :param charges: The new charges of atoms to provide to the
            external software [a.u.].
        """
        # This function is passed to the system in the factory method,
        # adding it to the list of functions called when charges are
        # altered.  This is how the external software is updated to
        # reflect the state of the system.
        ...

    def update_subsystems(
            self,
            subsystems: np.ndarray[Any, np.dtype[np.object_]],
    ) -> None:
        """A function which updates the subsystem membership.

        :param subsystems: The new subsystems of atoms to provide to the
            external software.
        """
        # This function is passed to the system in the factory method,
        # adding it to the list of functions called when subsystem
        # membership is altered.  This is how the external software is
        # updated to reflect the state of the system.
        ...
