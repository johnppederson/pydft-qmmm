"""Abstract base classes for plugins.
"""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydft_qmmm.calculators import Calculator
    from pydft_qmmm.calculators import CompositeCalculator
    from pydft_qmmm.integrators import Integrator


class Plugin(ABC):
    """The abstract plugin base class.

    Attributes:
        _modifieds: A list of names of objects that have been modified
            by the plugin.
        _key: The type of object that the plugin modifies.
    """
    _modifieds: list[str] = []
    _key: str = ""


class CalculatorPlugin(Plugin):
    """The plugin base class for modifying calculator routines.
    """
    _key: str = "calculator"

    @abstractmethod
    def modify(self, calculator: Calculator) -> None:
        """Modify the functionality of a calculator.

        Args:
            calculator: The calculator whose functionality will be
                modified by the plugin.
        """


class CompositeCalculatorPlugin(Plugin):
    """The plugin base class for modifying composite calculator routines.
    """
    _key: str = "calculator"

    @abstractmethod
    def modify(self, calculator: CompositeCalculator) -> None:
        """Modify the functionality of a calculator.

        Args:
            calculator: The composite calculator whose functionality
                will be modified by the plugin.
        """


class PartitionPlugin(CompositeCalculatorPlugin):
    """The plugin base class for modifying partitioning routines.

    Attributes:
        _query: The VMD-like query representing atoms which will be
            evaluated with the partitioning scheme.
    """
    _key: str = "calculator"
    _query: str = ""

    @abstractmethod
    def generate_partition(self) -> None:
        """Perform the system partitioning.
        """


class IntegratorPlugin(Plugin):
    """The plugin base class for modifying integrator routines.
    """
    _key: str = "integrator"

    @abstractmethod
    def modify(self, integrator: Integrator) -> None:
        """Modify the functionality of an integrator.

        Args:
            integrator: The integrator whose functionality will be
                modified by the plugin.
        """
