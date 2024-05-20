#! /usr/bin/env python3
"""A module defining the abstract :class:`Plugin` base class and derived
classes.
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
    """The base class for creating |package| plugins.
    """
    _modifieds: list[str] = []
    _key: str = ""


class CalculatorPlugin(Plugin):
    """The base class for creating a :class:`Plugin` which modifies any
    :class:`Calculator` class.
    """
    _key: str = "calculator"

    @abstractmethod
    def modify(self, calculator: Calculator) -> None:
        """Modify the functionality of any :class:`Calculator`.

        :param calculator: |calculator| to modify with the
            :class:`Plugin`.
        """


class CompositeCalculatorPlugin(Plugin):
    """The base class for creating a :class:`Plugin` which modifies the
    :class:`QMMMCalculator` class.
    """
    _key: str = "calculator"

    @abstractmethod
    def modify(self, calculator: CompositeCalculator) -> None:
        """Modify the functionality of a :class:`QMMMCalculator`
        :class:`QMMMCalculator`.

        :param calculator: The :class:`QMMMCalculator` object to modify
            with the :class:`Plugin`.
        """


class IntegratorPlugin(Plugin):
    """The base class for creating a :class:`Plugin` which modifies any
    :class:`Integrator` class.
    """
    _key: str = "integrator"

    @abstractmethod
    def modify(self, integrator: Integrator) -> None:
        """Modify the functionality of a :class:`Integrator`.

        :param integrator: |integrator| to modify with the
            :class:`Plugin`.
        """
