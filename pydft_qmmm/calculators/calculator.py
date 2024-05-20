#! /usr/bin/env python3
"""A module defining the :class:`Calculator` base class and derived
non-multiscale classes.
"""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydft_qmmm.plugins.plugin import CalculatorPlugin
    from pydft_qmmm.common import Results
    from pydft_qmmm import System


class Calculator(ABC):
    """An abstract :class:`Calculator` base class for interfacing with
    plugins.
    """
    system: System
    name: str = ""
    _plugins: list[str] = []

    @abstractmethod
    def calculate(
            self,
            return_forces: bool | None = True,
            return_components: bool | None = True,
    ) -> Results:
        """Calculate energies and forces for the :class:`System` with
        the :class:`Calculator`.

        :param return_forces: Whether or not to return forces.
        :param return_components: Whether or not to return
            the components of the energy.
        :return: The energy, forces, and energy components of the
            calculation.
        """

    def register_plugin(self, plugin: CalculatorPlugin) -> None:
        """Register a :class:`Plugin` modifying a :class:`Calculator`
        routine.

        :param plugin: An :class:`CalculatorPlugin` object.
        """
        self._plugins.append(type(plugin).__name__)
        plugin.modify(self)

    def active_plugins(self) -> list[str]:
        """Get the current list of active plugins.

        :return: A list of the active plugins being employed by the
            :class:`Calculator`.
        """
        return self._plugins
