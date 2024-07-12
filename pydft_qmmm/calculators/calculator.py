"""The calculator base class.
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
    """The abstract calculator base class.

    Attributes:
        system: (class attribute) The system whose atom positions, atom
            identities, and geometry will be used to calculate energies
            and forces.
        name: (class attribute) The name of the calculator.
        _plugins: (class attribute) The list of plugin names that have
            been registered to the calculator.
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

    def register_plugin(self, plugin: CalculatorPlugin) -> None:
        """Record plugin name and apply the plugin to the calculator.

        Args:
            plugin: A plugin that will modify the behavior of one or
                more calculator routines.
        """
        self._plugins.append(type(plugin).__name__)
        plugin.modify(self)

    def active_plugins(self) -> list[str]:
        """Get the current list of active plugins.

        Returns:
            A list of the active plugins registered by the calculator.
        """
        return self._plugins
