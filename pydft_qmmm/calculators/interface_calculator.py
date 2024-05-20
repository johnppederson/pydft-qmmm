#! /usr/bin/env python3
"""A module defining the :class:`Calculator` base class and derived
non-multiscale classes.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import TYPE_CHECKING

from .calculator import Calculator
from pydft_qmmm.common import Results

if TYPE_CHECKING:
    from pydft_qmmm.interfaces import SoftwareInterface
    from pydft_qmmm import System


@dataclass
class InterfaceCalculator(Calculator):
    """A :class:`Calculator` class, defining the procedure for
    standalone QM or MM calculations.

    :param system: |system| to perform calculations on.
    :param interface: |interface| to perform calculations with.
    :param options: Options to provide to the
        :class:`SoftwareInterface`.
    """
    system: System
    interface: SoftwareInterface
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Send notifier functions from the interface to the respective
        state or topology variable for monitoring, immediately after
        initialization.
        """
        self.theory_level = self.interface.theory_level
        self.name = str(self.interface.theory_level).split(".")[1]

    def calculate(
            self,
            return_forces: bool | None = True,
            return_components: bool | None = True,
    ) -> Results:
        energy = self.interface.compute_energy(**self.options)
        results = Results(energy)
        if return_forces:
            forces = self.interface.compute_forces(**self.options)
            results.forces = forces
        if return_components:
            components = self.interface.compute_components(**self.options)
            results.components = components
        return results
