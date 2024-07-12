"""A utility to manage thread and memory utilization.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydft_qmmm.calculators import InterfaceCalculator


class ResourceManager:
    """A utility to manage thread and memory utilization.

    Args:
        calculators: A list of calculators which interface to external
            software.
    """

    def __init__(self, calculators: list[InterfaceCalculator]) -> None:
        self._calculators = calculators

    def update_threads(self, threads: int) -> None:
        """Set the number of threads that calculators can use.

        Args:
            threads: The number of threads to utilize.
        """
        for calculator in self._calculators:
            calculator.interface.update_threads(threads)

    def update_memory(self, memory: str) -> None:
        """Set the amount of memory that calculators can use.

        Args:
            memory: The amount of memory to utilize.
        """
        for calculator in self._calculators:
            calculator.interface.update_memory(memory)
